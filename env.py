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
from app.models import Observation, Reward, StepResponse


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

    async def step(self, prompt: str) -> StepResponse:
        if not self.state:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.state.steps += 1

        def on_stage(msg: str):
            self.state.history.append({"type": "stage", "message": msg})

        def on_token_delta(chunk: str):
            pass

        conversation_history = [{"role": "user", "content": prompt}]

        # Build state vector from actual task features
        _DIFFICULTY_TO_COMPLEXITY = {"easy": "low", "medium": "medium", "hard": "high"}
        _TASK_TO_INTENT = {
            "debug_memory_leak": "bug_fix",
            "fix_retry_logic": "bug_fix",
            "implement_circuit_breaker": "feature",
        }
        complexity = _DIFFICULTY_TO_COMPLEXITY.get(self.state.task_difficulty.lower(), "medium")
        intent = _TASK_TO_INTENT.get(self.state.task_name, "feature")
        past_iisg = self.state.total_reward / max(self.state.steps, 1)
        self.state_encoder.save_history(past_iisg)

        state_vec = self.state_encoder.encode(
            intent_type=intent,
            complexity=complexity,
            prompt=prompt,
            cwd=self.state.workspace,
        )
        action_idx = self.bandit.select(state_vec.to_list())
        action = self.bandit.get_action(action_idx)
        pipeline_action = action_to_pipeline(action)

        ctx = await self.runner.run(
            prompt=prompt,
            action=pipeline_action,
            on_stage=on_stage,
            on_token_delta=on_token_delta,
            conversation_history=conversation_history,
        )

        current_task = self.task_bank.get_current()
        step_budget = getattr(current_task, "step_budget", 20)

        # 1. Evaluate actual grader scoring trajectory
        grader_score = self.task_bank.grade(self.state.workspace)
        grader_score = min(max(float(grader_score), 0.01), 0.99)

        # Apply penalties BEFORE the improvement-based reward delta calculation
        # 1. Empty submission penalty
        if not prompt.strip():
            grader_score = 0.01
            
        # 2. Syntax error penalty
        try:
            compile(prompt, "<agent>", "exec")
        except SyntaxError:
            grader_score = min(grader_score, 0.15)

        # 3. Repeated submission penalty
        if prompt.strip() == self.state.last_submission.strip():
            grader_score = max(0.01, grader_score - 0.05)

        # 2. Compute Improvement Reward Mapping
        base_reward = grader_score - self.state.best_score
        if base_reward <= 0.0:
            base_reward = 0.01

        # 3. Apply early-solver efficiency bonus
        efficiency_bonus = 0.01
        if grader_score >= 0.95 and self.state.steps <= step_budget // 2:
            efficiency_bonus = 0.1

        reward_val = base_reward + efficiency_bonus
        if reward_val > 0.99:
            reward_val = 0.99
        reward_val = max(0.01, reward_val)

        # Update environment tracking
        self.state.best_score = max(self.state.best_score, grader_score)
        self.state.total_reward += reward_val
        
        # 4. Multi-step 'done' conditions
        is_stuck = (prompt == self.state.last_submission)
        done = grader_score >= 0.95 or self.state.steps >= step_budget or is_stuck
        
        self.state.last_submission = prompt

        self.state.history.append({
            "type": "step",
            "step": self.state.steps,
            "prompt": prompt,
            "response": ctx.final_response[:500] if ctx.final_response else "",
            "reward": reward_val,
            "iisg_rate": ctx.iisg_pass_rate,
            "action": self.bandit.get_action_name(action_idx),
        })

        self.state_encoder.save_history(ctx.iisg_pass_rate)
        
        new_state = self.state_encoder.encode("feature", "medium", prompt, self.state.workspace)
        self.bandit.update(new_state.to_list(), action_idx, reward_val)
        self.bandit.save()

        # Build token-efficiency score for the efficiency component
        token_budget = 3000
        tokens_used = getattr(ctx, "tokens_used", 0) or 0
        token_efficiency = max(0.01, min(0.99, 1.0 - tokens_used / token_budget))

        observation = Observation(**self._get_state_dict())
        reward_obj = Reward(
            correctness=min(max(grader_score, 0.01), 0.99),
            efficiency=token_efficiency,
            final_score=min(max(reward_val, 0.01), 0.99),
        )

        return StepResponse(
            observation=observation,
            reward=reward_obj,
            done=done,
            info={"action": self.bandit.get_action_name(action_idx)},
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