"""
FastAPI server wrapping the OpenEnv environment.

Run directly:
    python server.py

Or via uvicorn:
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

import os
import uuid
from collections import OrderedDict
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import OpenEnv
from app.models import Observation, StepAction, Reward, StepResponse


# ---------------------------------------------------------------------------
# Session management (LRU dict, not a single global instance)
# ---------------------------------------------------------------------------

_sessions: OrderedDict[str, OpenEnv] = OrderedDict()
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "512"))


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Orion OpenEnv Server",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(body: Optional[ResetRequest] = None):
    """Reset the environment and return the initial observation."""
    session_id = (body.session_id if body else None) or str(uuid.uuid4())
    # Evict oldest session if at capacity
    if len(_sessions) >= MAX_SESSIONS:
        _, old_env = _sessions.popitem(last=False)
        old_env.close()
    api_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    env = OpenEnv(api_key=api_key)
    _sessions[session_id] = env
    try:
        obs_dict = await env.reset(body.difficulty if body else None)
        return {"session_id": session_id, **obs_dict}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/step")
async def step(body: StepRequest):
    """Execute one step in the environment."""
    if body.session_id and body.session_id in _sessions:
        env = _sessions[body.session_id]
    elif _sessions:
        env = next(iter(_sessions.values()))
    else:
        return JSONResponse(status_code=404,
            content={"error": "No active session. Call /reset first."})
    try:
        result = await env.step(body.prompt)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/state")
async def state(session_id: Optional[str] = None):
    """Return the current environment state dict."""
    if session_id and session_id in _sessions:
        env = _sessions[session_id]
    elif _sessions:
        env = next(iter(_sessions.values()))
    else:
        return JSONResponse(status_code=404,
            content={"error": "No active session"})
    return env.get_state()


@app.get("/tasks")
async def get_tasks():
    """Return a list of available tasks."""
    from tasks.task_bank import TASKS
    return [{"name": t.name, "difficulty": t.difficulty, "prompt": t.prompt} for t in TASKS]


@app.get("/openenv.yaml")
async def get_openenv_yaml():
    """Return the openenv.yaml file."""
    from fastapi.responses import FileResponse
    return FileResponse("openenv.yaml", media_type="text/plain")


@app.get("/metadata")
async def metadata():
    """Return environment metadata."""
    from tasks.task_bank import TASKS
    return {
        "name": "orion-cli",
        "version": "1.0.0",
        "description": "RL-optimised agentic coding assistant OpenEnv environment",
        "tasks": [t.name for t in TASKS],
        "action_schema": {"prompt": "str"},
        "observation_fields": [
            "task_name", "task_difficulty", "task_prompt",
            "workspace", "history", "total_reward", "steps",
        ],
    }


@app.get("/schema")
async def schema():
    """Return JSON schemas for the core Action, Observation, and Reward models."""
    return {
        "action": StepAction.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "reward": Reward.model_json_schema(),
    }


class GraderRequest(BaseModel):
    task_name: str
    workspace: str


@app.post("/grader")
async def grader(body: GraderRequest):
    """Re-run a task grader on an existing workspace for trajectory replay."""
    from tasks.task_bank import TASKS
    task_map = {t.name: t for t in TASKS}
    if body.task_name not in task_map:
        return JSONResponse(status_code=404,
            content={"error": f"Unknown task: {body.task_name}"})
    try:
        score = task_map[body.task_name].grader(body.workspace)
        return {"score": score, "task_name": body.task_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/baseline")
async def baseline():
    """Return baseline information."""
    from tasks.task_bank import TASKS
    return {
        "status": "baseline not configured",
        "tasks": [t.name for t in TASKS],
        "note": "run inference.py for baseline scores",
    }


@app.get("/rl/stats")
async def rl_stats():
    """Return runtime statistics for the internal LinUCB Contextual Bandit."""
    from orion.rl.bandit import LinUCBBandit, ACTION_NAMES
    from orion.rl.state_encoder import StateEncoder
    import numpy as np

    bandit = LinUCBBandit()
    encoder = StateEncoder()

    def _best_action(intent: str, complexity: str) -> str:
        # Build hypothetical state vector without mutating bandit metrics
        sv = encoder.encode(intent_type=intent, complexity=complexity)
        state_array = np.array(sv.to_list(), dtype=float)
        scores = [bandit._ucb_score(state_array, a) for a in range(bandit.n_actions)]
        return ACTION_NAMES[int(np.argmax(scores))]

    return {
        "algorithm": "LinUCB Contextual Bandit",
        "n_actions": bandit.n_actions,
        "n_features": bandit.n_features,
        "total_episodes": bandit.total_count,
        "action_counts": {
            ACTION_NAMES[a]: bandit.counts.get(a, 0) for a in range(bandit.n_actions)
        },
        "best_action_per_task": {
            "debug_memory_leak": _best_action("bug_fix", "medium"),
            "fix_retry_logic": _best_action("bug_fix", "high"),
            "implement_circuit_breaker": _best_action("feature", "high")
        },
        "exploration_alpha": bandit.alpha,
        "weights_shape": [bandit.n_actions, bandit.n_features]
    }

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
