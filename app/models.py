"""
Typed Pydantic v2 models for the OpenEnv environment interface.

These models define the observation, action, and reward schemas
used by the server API and referenced in openenv.yaml.
"""

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Full environment observation returned after reset() or step()."""
    task_name: str
    task_difficulty: str
    task_prompt: str
    workspace: str
    history: list[dict] = Field(default_factory=list)
    total_reward: float = Field(default=0.01, ge=0.01)
    steps: int = Field(default=0, ge=0)
    best_score: float = Field(default=0.01, ge=0.01)


class StepAction(BaseModel):
    """Action submitted by the agent on each step."""
    prompt: str = Field(min_length=1)


class Reward(BaseModel):
    """Structured reward breakdown for a single step."""
    correctness: float = Field(default=0.01, ge=0.01, le=0.99)
    efficiency: float = Field(default=0.01, ge=0.01, le=0.99)
    final_score: float = Field(default=0.01, ge=0.01, le=0.99)


class StepResponse(BaseModel):
    """Response returned by POST /step."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default_factory=dict)
