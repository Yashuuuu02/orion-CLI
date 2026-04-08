"""
FastAPI server wrapping the OpenEnv environment.

Run directly:
    python server.py

Or via uvicorn:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env import OpenEnv


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None


class StepRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Lifespan – create / tear-down the global env instance
# ---------------------------------------------------------------------------

_env: Optional[OpenEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    api_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    _env = OpenEnv(api_key=api_key)
    yield
    if _env is not None:
        _env.close()
        _env = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Orion OpenEnv Server",
    version="1.0.0",
    lifespan=lifespan,
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
    difficulty = body.difficulty if body else None
    observation = await _env.reset(difficulty)
    return observation


@app.post("/step")
async def step(body: StepRequest):
    """Execute one step in the environment."""
    state, reward, done = await _env.step(body.prompt)
    return {
        "observation": state,
        "reward": reward,
        "done": done,
    }


@app.get("/state")
async def state():
    """Return the current environment state dict."""
    return _env.get_state()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
