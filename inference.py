import os
import asyncio
from openai import AsyncOpenAI
from env import OpenEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia_nim/qwen/qwen2.5-coder-32b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 8


def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


async def main():
    api_key = (
        os.environ.get("HF_TOKEN") or 
        os.environ.get("NVIDIA_NIM_API_KEY") or 
        ""
    )
    env = OpenEnv(api_key=api_key)

    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=api_key or HF_TOKEN or "no-key",
    )

    state = await env.reset()
    task_name = state["task_name"]
    task_prompt = state["task_prompt"]

    log_start(task=task_name, env_name="orion", model=MODEL_NAME)

    rewards = []
    steps = 0
    success = False

    try:
        for i in range(1, MAX_STEPS + 1):
            messages = [
                {"role": "system", "content": "You are a coding assistant. Respond with the action to take."},
                {"role": "user", "content": task_prompt},
            ]

            error = None
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
                action_text = response.choices[0].message.content or ""
            except Exception as e:
                action_text = ""
                error = str(e)

            try:
                result_state, reward, done = await env.step(action_text)
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps = i

            log_step(step=i, action=task_name, reward=reward, done=done, error=error)

            if done:
                success = reward >= 0.95
                break
    finally:
        env.close()
        score = sum(rewards) / len(rewards) if rewards else 0.0
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())