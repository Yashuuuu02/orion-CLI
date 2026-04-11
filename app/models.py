"""
Typed Pydantic v2 models for the OpenEnv environment interface.

These models define the observation, action, and reward schemas
used by the server API and referenced in openenv.yaml.
"""

from typing import Literal, Union, Annotated, Optional
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
    available_tools: list[str] = Field(
        default=["read_file", "write_file", "run_tests", 
                 "list_files", "submit"],
        description="Tools available to the agent"
    )
    last_tool_result: Optional[str] = Field(
        default=None,
        description="Result of the last tool call"
    )
    files_in_workspace: list[str] = Field(
        default_factory=list,
        description="List of files currently in workspace"
    )


class StepAction(BaseModel):
    """Action submitted by the agent on each step (Legacy)."""
    prompt: str = Field(min_length=1)


class ReadFile(BaseModel):
    """Read a file from the workspace."""
    action_type: Literal["read_file"]
    path: str = Field(description="Relative path to file in workspace")

class WriteFile(BaseModel):
    """Write code to a file in the workspace."""
    action_type: Literal["write_file"]
    path: str = Field(description="Relative path to file in workspace")
    content: str = Field(description="Complete file content to write", min_length=1)

class RunTests(BaseModel):
    """Run the grader tests against current workspace state."""
    action_type: Literal["run_tests"]

class ListFiles(BaseModel):
    """List all files currently in the workspace."""
    action_type: Literal["list_files"]

class Submit(BaseModel):
    """Submit final solution and end the episode."""
    action_type: Literal["submit"]
    explanation: str = Field(
        description="Brief explanation of what you fixed",
        min_length=10
    )

Action = Annotated[
    Union[ReadFile, WriteFile, RunTests, ListFiles, Submit],
    Field(discriminator="action_type")
]

class ToolResponse(BaseModel):
    """What the environment returns after a tool call."""
    action_type: str
    result: str
    success: bool


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
    tool_response: ToolResponse
    info: dict = Field(default_factory=dict)

