import os
import tempfile
import shutil
from typing import Optional
from dataclasses import dataclass, field

from orion.rl.state_encoder import StateEncoder
from orion.rl.bandit import LinUCBBandit, ACTIONS, action_to_pipeline
from orion.pipeline.runner import PipelineRunner
from orion.provider.provider import Provider
from orion.tool.tools import ReadTool, WriteTool, EditTool, GrepTool
from tasks.task_bank import TaskBank, get_task_bank
from app.models import Observation, Reward, StepResponse, ToolResponse


@dataclass
class OpenEnvState:
    task_name: str
    task_difficulty: str
    task_prompt: str
    workspace: str
    history: list = field(default_factory=list)
    total_reward: float = 0.01
    steps: int = 0
    best_score: float = 0.01
    last_submission: str = ""


class OpenEnv:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.state: Optional[OpenEnvState] = None
        self.task_bank = get_task_bank()
        self.state_encoder = StateEncoder()
        self.bandit = LinUCBBandit()
        if not os.path.exists("/app/bandit_weights.npz"):
            try:
                from scripts.preseed_bandit import preseed_bandit
                preseed_bandit()
            except ImportError:
                pass
        self.bandit.load()
        
        self.provider = Provider(api_key=api_key)
        self.tools = {
            "read": ReadTool(),
            "write": WriteTool(),
            "edit": EditTool(),
            "grep": GrepTool(),
        }
        self.runner = PipelineRunner(provider=self.provider, tools=self.tools)

    async def reset(self, task_name: str = None, difficulty: str = None) -> dict:
        if task_name:
            task = self.task_bank.get_by_name(task_name)
            if task is None:
                task = self.task_bank.sample(difficulty=difficulty)
        else:
            task = self.task_bank.sample(difficulty=difficulty)
        
        workspace = tempfile.mkdtemp(prefix="orion_env_")
        
        for filename, content in task.setup_files.items():
            path = os.path.join(workspace, filename)
            with open(path, "w") as f:
                f.write(content)

        self.state = OpenEnvState(
            task_name=task.name,
            task_difficulty=task.difficulty,
            task_prompt=task.prompt,
            workspace=workspace,
        )

        return {
            "task_name": task.name,
            "task_difficulty": task.difficulty,
            "task_prompt": task.prompt,
            "workspace": workspace,
            "history": [],
            "total_reward": 0.01,
            "steps": 0,
            "best_score": 0.01,
        }

    async def step(self, action: dict | str) -> StepResponse:
        """Execute a tool action and return the result."""
        if not self.state:
            raise RuntimeError("Environment not initialized. Call reset() first.")
            
        if isinstance(action, str):
            action = {
                "action_type": "write_file",
                "path": "solution.py",
                "content": action
            }

        action_type = action.get("action_type", "")
        tool_response = None
        grader_score = self.state.best_score
        done = False
        
        # Route to correct tool handler
        if action_type == "read_file":
            path = action.get("path", "")
            full_path = os.path.join(self.state.workspace, path)
            if os.path.exists(full_path):
                content = open(full_path).read()
                tool_response = ToolResponse(
                    action_type="read_file",
                    result=content[:3000],  # cap at 3000 chars
                    success=True
                )
            else:
                tool_response = ToolResponse(
                    action_type="read_file",
                    result=f"File not found: {path}",
                    success=False
                )
            # No grader call on read — reward comes from progress
            reward_val = 0.01
            
        elif action_type == "write_file":
            path = action.get("path", "")
            content = action.get("content", "")
            full_path = os.path.join(self.state.workspace, path)
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content)
                tool_response = ToolResponse(
                    action_type="write_file",
                    result=f"Written {len(content)} bytes to {path}",
                    success=True
                )
                # Run grader after write to check progress
                grader_score = self.task_bank.grade(self.state.workspace)
                grader_score = min(max(float(grader_score), 0.01), 0.99)
            except Exception as e:
                tool_response = ToolResponse(
                    action_type="write_file",
                    result=f"Write failed: {str(e)}",
                    success=False
                )
                grader_score = 0.01
                
        elif action_type == "run_tests":
            grader_score = self.task_bank.grade(self.state.workspace)
            grader_score = min(max(float(grader_score), 0.01), 0.99)
            tool_response = ToolResponse(
                action_type="run_tests",
                result=f"Test score: {grader_score:.2f}. "
                       f"Best so far: {self.state.best_score:.2f}",
                success=True
            )
            reward_val = 0.01  # minimal reward for just running tests
            
        elif action_type == "list_files":
            files = os.listdir(self.state.workspace)
            tool_response = ToolResponse(
                action_type="list_files",
                result="\n".join(files) if files else "Empty workspace",
                success=True
            )
            reward_val = 0.01
            
        elif action_type == "submit":
            grader_score = self.task_bank.grade(self.state.workspace)
            grader_score = min(max(float(grader_score), 0.01), 0.99)
            tool_response = ToolResponse(
                action_type="submit",
                result=f"Submitted. Final score: {grader_score:.2f}",
                success=True
            )
            done = True
        else:
            tool_response = ToolResponse(
                action_type="unknown",
                result=f"Unknown action: {action_type}",
                success=False
            )
            grader_score = 0.01

        # Compute improvement-based reward for write and submit
        if action_type in ("write_file", "submit"):
            base_reward = grader_score - self.state.best_score
            if base_reward <= 0.0:
                base_reward = 0.01
            efficiency_bonus = 0.01
            if grader_score >= 0.95 and self.state.steps <= self.task_bank.current_task.step_budget // 2:
                efficiency_bonus = 0.1
            reward_val = min(max(base_reward + efficiency_bonus, 0.01), 0.99)
        elif 'reward_val' not in locals():
            reward_val = 0.01
            
        if action_type in ("view", "bash", "list_files"):
            step_cost = -0.02
        else:
            step_cost = 0.0
            
        reward_val = max(0.01, reward_val + step_cost)
        
        # Update state
        self.state.steps += 1
        self.state.best_score = max(self.state.best_score, grader_score)
        self.state.total_reward += reward_val
        self.state.history.append({
            "step": self.state.steps,
            "action_type": action_type,
            "tool_result": tool_response.result[:200],
            "reward": reward_val
        })
        
        # Check done conditions
        if self.state.steps >= self.task_bank.current_task.step_budget:
            done = True
        if grader_score >= 0.95:
            done = True
        
        # Build response
        reward_obj = Reward(
            correctness=min(max(grader_score, 0.01), 0.99),
            efficiency=min(max(1.0 - self.state.steps / 
                        self.task_bank.current_task.step_budget, 0.01), 0.99),
            final_score=min(max(reward_val, 0.01), 0.99)
        )
        
        obs = Observation(
            task_name=self.state.task_name,
            task_difficulty=self.state.task_difficulty,
            task_prompt=self.state.task_prompt,
            workspace=self.state.workspace,
            history=self.state.history[-5:],
            total_reward=self.state.total_reward,
            steps=self.state.steps,
            best_score=self.state.best_score,
            available_tools=["read_file", "write_file", 
                            "run_tests", "list_files", "submit"],
            last_tool_result=tool_response.result[:500],
            files_in_workspace=os.listdir(self.state.workspace)
        )
        
        return StepResponse(
            observation=obs,
            reward=reward_obj,
            done=done,
            tool_response=tool_response,
            info={"action_type": action_type}
        )

    def _get_state_dict(self) -> dict:
        if not self.state:
            return {"error": "No active state"}
        
        return {
            "task_name": self.state.task_name,
            "task_difficulty": self.state.task_difficulty,
            "task_prompt": self.state.task_prompt,
            "workspace": self.state.workspace,
            "history": self.state.history[-5:],
            "total_reward": round(self.state.total_reward, 4),
            "steps": self.state.steps,
            "best_score": round(self.state.best_score, 4),
        }

    def get_state(self) -> dict:
        return self._get_state_dict()

    async def state(self) -> dict:
        return self.get_state()

    def close(self):
        if self.state and self.state.workspace:
            try:
                shutil.rmtree(self.state.workspace)
            except Exception:
                pass


_env_instance: Optional[OpenEnv] = None


def get_env(api_key: str) -> OpenEnv:
    global _env_instance
    if _env_instance is None:
        _env_instance = OpenEnv(api_key)
    return _env_instance


async def reset(task_name: str = None, difficulty: str = None) -> dict:
    env = get_env(os.environ.get("NVIDIA_NIM_API_KEY", ""))
    return await env.reset(task_name=task_name, difficulty=difficulty)


async def step(prompt: str) -> StepResponse:
    env = get_env(os.environ.get("NVIDIA_NIM_API_KEY", ""))
    return await env.step(prompt)


async def state() -> dict:
    env = get_env(os.environ.get("NVIDIA_NIM_API_KEY", ""))
    return env.get_state()