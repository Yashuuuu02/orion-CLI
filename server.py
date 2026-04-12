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
from fastapi.responses import JSONResponse, HTMLResponse
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
    task_name: Optional[str] = None
    difficulty: Optional[str] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Optional[dict] = None
    prompt: Optional[str] = None
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
        obs_dict = await env.reset(task_name=body.task_name if body else None, 
                                   difficulty=body.difficulty if body else None)
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
        action_payload = body.action if body.action is not None else body.prompt
        if action_payload is None:
            return JSONResponse(status_code=400, content={"error": "Must provide action or prompt"})
        result = await env.step(action_payload)
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
        raw_score = task_map[body.task_name].grader(body.workspace)
        clamped_score = min(max(float(raw_score), 0.01), 0.99)
        return {"score": clamped_score, "task_name": body.task_name}
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
    bandit.load("/app/bandit_weights.npz")
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
        "total_episodes": sum(bandit.counts.values()),
        "action_counts": {
            ACTION_NAMES[a]: bandit.counts.get(a, 0) for a in range(bandit.n_actions)
        },
        "best_action_per_task": {
            "fix_tenacity_retry": _best_action("bug_fix", "medium"),
            "fix_cachetools_ttl": _best_action("bug_fix", "medium"),
            "implement_pybreaker": _best_action("feature", "high"),
            "fix_async_race": _best_action("bug_fix", "high")
        },
        "exploration_alpha": bandit.alpha,
        "weights_shape": [bandit.n_actions, bandit.n_features]
    }

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OrionCLI Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  body {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    margin: 0;
    padding: 20px;
    display: flex;
    flex-direction: column;
    height: 100vh;
    box-sizing: border-box;
  }
  h1, h2, h3 { color: #58a6ff; margin-top: 0; }
  .accent { color: #00ff88; }
  .container {
    display: flex;
    flex: 1;
    gap: 20px;
    min-height: 0;
  }
  .panel {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .left-panel { flex: 1; }
  .right-panel { flex: 1; }
  .bottom-panel {
    flex: 1;
    margin-top: 20px;
  }
  .badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    background-color: #238636;
    color: #ffffff;
    font-size: 0.8em;
    font-weight: bold;
  }
  .progress-bg {
    background-color: #21262d;
    border-radius: 4px;
    height: 20px;
    width: 100%;
    margin-top: 5px;
  }
  .progress-bar {
    background-color: #00ff88;
    height: 100%;
    border-radius: 4px;
    width: 0%;
    transition: width 0.3s;
  }
  .data-row {
    display: flex;
    justify-content: space-between;
    margin: 10px 0;
    border-bottom: 1px solid #30363d;
    padding-bottom: 5px;
  }
  .log-entry { margin: 5px 0; font-size: 0.9em; }
  .sparkline { display: flex; align-items: flex-end; height: 40px; gap: 2px; margin-top: 10px; }
  .spark-bar { width: 10px; background-color: #00ff88; transition: height 0.3s; }
  .bar-row { display: flex; align-items: center; margin: 5px 0; }
  .bar-label { width: 250px; font-size: 0.8em; }
  .bar-value { background-color: #58a6ff; height: 15px; margin-right: 10px; transition: width 0.3s; min-width: 2px; }
  .bar-number { font-size: 0.8em; }
  .waiting { text-align: center; color: #8b949e; margin-top: 50px; font-style: italic; }
  #log-container { overflow-y: auto; max-height: 100%; }
</style>
</head>
<body>

<div id="content" style="display:none; height:100%; width:100%; flex-direction:column;">
  <div class="container">
    <div class="panel left-panel">
      <h2>🚀 Current Episode</h2>
      <div id="task-info"></div>
      
      <div style="margin-top:20px;">
        <div>Steps: <span id="steps-text"></span></div>
        <div class="progress-bg"><div id="progress-bar" class="progress-bar"></div></div>
      </div>
      
      <div style="margin-top:20px;">
        <div>Best Score:</div>
        <div id="best-score" class="accent" style="font-size: 2.5em; font-weight: bold;">0.00</div>
      </div>
      
      <div style="margin-top:20px;">
        <div>Last Tool Call:</div>
        <pre id="last-tool" style="background:#0d1117; padding:10px; border-radius:4px; max-height:100px; overflow-y:auto; font-size:0.8em; border:1px solid #30363d;">-</pre>
      </div>

      <div style="margin-top:20px;">
        <div>Reward History (last 10)</div>
        <div class="sparkline" id="sparkline"></div>
      </div>
    </div>
    
    <div class="panel right-panel">
      <h2>🤖 RL Bandit Stats</h2>
      <div class="data-row"><span>Algorithm:</span><span id="algo-name" class="accent">-</span></div>
      <div class="data-row"><span>Total Episodes:</span><span id="total-episodes" class="accent">-</span></div>
      <div class="data-row"><span>Exploration α:</span><span id="alpha-val" class="accent">-</span></div>
      
      <h3 style="margin-top:20px;">Best Action per Task</h3>
      <div id="best-actions" style="font-size:0.9em; margin-bottom:20px;"></div>
      
      <h3 style="margin-top:20px;">Action Distribution</h3>
      <div id="action-dist"></div>
    </div>
  </div>
  
  <div class="panel bottom-panel">
    <h2>📋 Live Log</h2>
    <div id="log-container"></div>
  </div>
</div>

<div id="waiting" class="waiting">
  <h2>Waiting for episode...</h2>
  <p>No active sessions found. Run inference.py to start an episode.</p>
</div>

<script>
async function fetchData() {
  try {
    const rlRes = await fetch('/rl/stats');
    if (rlRes.ok) {
      const rlData = await rlRes.json();
      document.getElementById('algo-name').innerText = rlData.algorithm || '-';
      document.getElementById('total-episodes').innerText = rlData.total_episodes || '0';
      document.getElementById('alpha-val').innerText = rlData.exploration_alpha || '-';
      
      let bestHtml = '';
      for (const [task, act] of Object.entries(rlData.best_action_per_task || {})) {
        bestHtml += `<div style="margin:5px 0;"><strong>${task}</strong>: <span class="accent">${act}</span></div>`;
      }
      document.getElementById('best-actions').innerHTML = bestHtml;
      
      let distHtml = '';
      const counts = rlData.action_counts || {};
      const maxCount = Math.max(...Object.values(counts), 1);
      for (const [act, count] of Object.entries(counts)) {
        const w = (count / maxCount) * 100;
        distHtml += `
          <div class="bar-row">
            <div class="bar-label">${act}</div>
            <div class="bar-value" style="width: ${w * 0.6}%"></div>
            <div class="bar-number">${count}</div>
          </div>
        `;
      }
      document.getElementById('action-dist').innerHTML = distHtml;
    }
    
    const stateRes = await fetch('/state');
    if (stateRes.ok) {
        const state = await stateRes.json();
        if (state.error) {
            showWaiting();
            return;
        }
        showContent();
        
        document.getElementById('task-info').innerHTML = `
          <span style="font-weight:bold; font-size:1.2em;">${state.task_name || 'Tasks'}</span> 
          <span class="badge" style="margin-left:10px;">${state.task_difficulty || 'Unknown'}</span>
        `;
        
        const steps = state.steps || 0;
        const budget = 30; // approx max budget
        document.getElementById('steps-text').innerText = steps;
        document.getElementById('progress-bar').style.width = Math.min((steps/budget)*100, 100) + '%';
        
        const best = parseFloat(state.best_score || 0).toFixed(2);
        document.getElementById('best-score').innerText = best;
        
        const hist = state.history || [];
        if (hist.length > 0) {
            const last = hist[hist.length - 1];
            document.getElementById('last-tool').innerText = `[${last.action_type}]\\n${last.tool_result ? last.tool_result.substring(0, 200) : ''}`;
        }
        
        let sparkHtml = '';
        const recentHist = hist.slice(-10);
        for (const h of recentHist) {
            const h2 = Math.max(2, (h.reward * 40)) + 'px';
            const val = parseFloat(h.reward).toFixed(2);
            sparkHtml += `<div class="spark-bar" style="height:${h2}" title="${val}"></div>`;
        }
        document.getElementById('sparkline').innerHTML = sparkHtml;
        
        let logHtml = '';
        const logHist = hist.slice(-5).reverse();
        for (const h of logHist) {
            const val = parseFloat(h.reward).toFixed(2);
            logHtml += `<div class="log-entry"><span style="color:#58a6ff;">Step ${h.step}</span> | Action: <span class="accent">${h.action_type}</span> | Reward: ${val}</div>`;
        }
        document.getElementById('log-container').innerHTML = logHtml;
    } else {
        showWaiting();
    }
  } catch(e) {
    console.error(e);
  }
}

function showWaiting() {
    document.getElementById('waiting').style.display = 'block';
    document.getElementById('content').style.display = 'none';
}

function showContent() {
    document.getElementById('waiting').style.display = 'none';
    document.getElementById('content').style.display = 'flex';
}

fetchData();
setInterval(fetchData, 3000);
</script>
</body>
</html>
"""

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=HTML_CONTENT)

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
