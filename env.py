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


@dataclass
class OpenEnvState:
    task_name: str
    task_difficulty: str
    task_prompt: str
    workspace: str
    history: list = field(default_factory=list)
    total_reward: float = 0.0
    steps: int = 0


class OpenEnv:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.state: Optional[OpenEnvState] = None
        self.task_bank = get_task_bank()
        self.state_encoder = StateEncoder()
        self.bandit = LinUCBBandit()
        
        self.provider = Provider(api_key=api_key)
        self.tools = {
            "read": ReadTool(),
            "write": WriteTool(),
            "edit": EditTool(),
            "grep": GrepTool(),
        }
        self.runner = PipelineRunner(provider=self.provider, tools=self.tools)

    async def reset(self, difficulty: Optional[str] = None) -> dict:
        task = self.task_bank.sample(difficulty)
        
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
            "total_reward": 0.0,
            "steps": 0,
        }

    async def step(self, prompt: str) -> tuple[dict, float, bool]:
        if not self.state:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.state.steps += 1

        async def on_stage(msg: str):
            self.state.history.append({"type": "stage", "message": msg})

        async def on_token_delta(chunk: str):
            pass

        conversation_history = [{"role": "user", "content": prompt}]

        action_idx = self.bandit.select([0, 1, 0, 0.5])
        action = self.bandit.get_action(action_idx)
        pipeline_action = action_to_pipeline(action)

        ctx = await self.runner.run(
            prompt=prompt,
            action=pipeline_action,
            on_stage=on_stage,
            on_token_delta=on_token_delta,
            conversation_history=conversation_history,
        )

        reward = ctx.reward if ctx.reward else 0.0
        
        self.state.total_reward += reward
        self.state.history.append({
            "type": "step",
            "step": self.state.steps,
            "prompt": prompt,
            "response": ctx.final_response[:500] if ctx.final_response else "",
            "reward": reward,
            "iisg_rate": ctx.iisg_pass_rate,
            "action": self.bandit.get_action_name(action_idx),
        })

        self.state_encoder.save_history(ctx.iisg_pass_rate)
        
        new_state = self.state_encoder.encode("feature", "medium", prompt, self.state.workspace)
        self.bandit.update(new_state.to_list(), action_idx, reward)

        done = self.state.steps >= 10 or reward >= 0.95

        return self._get_state_dict(), reward, done

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
        }

    async def state(self) -> dict:
        return self._get_state_dict()

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


async def reset(difficulty: Optional[str] = None) -> dict:
    env = get_env(os.environ.get("NVIDIA_NIM_API_KEY", ""))
    return await env.reset(difficulty)


async def step(prompt: str) -> tuple[dict, float, bool]:
    env = get_env(os.environ.get("NVIDIA_NIM_API_KEY", ""))
    return await env.step(prompt)


async def state() -> dict:
    env = get_env(os.environ.get("NVIDIA_NIM_API_KEY", ""))
    return await env.state()