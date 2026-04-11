import os
import asyncio
from openai import OpenAI
from env import OpenEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia_nim/qwen/qwen2.5-coder-32b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

MAX_STEPS = 8


def log_start(task, env_name, model):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step, action_text, reward, done, error):
    display_reward = min(max(reward, 0.01), 0.99)
    action_str = action_text.replace('\n', ' ')[:80]
    done_str = "true" if done else "false"
    error_str = "null" if error is None else error
    print(f"[STEP] step={step} action={action_str} reward={display_reward:.2f} done={done_str} error={error_str}", flush=True)


def log_end(task, steps, rewards):
    score = sum(rewards) / len(rewards) if rewards else 0.0
    success_str = "true" if score >= 0.1 else "false"
    clamped = [min(max(r, 0.01), 0.99) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


def call_llm(client, messages):
    """Synchronous LLM call."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    return response.choices[0].message.content or ""


async def main():
    api_key = (
        os.environ.get("HF_TOKEN") or
        os.environ.get("NVIDIA_NIM_API_KEY") or
        ""
    )
    env = OpenEnv(api_key=api_key)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=api_key or HF_TOKEN or "no-key",
    )

    try:
        for target_task in ["debug_memory_leak", "fix_retry_logic", "implement_circuit_breaker"]:
            task_name = target_task
            task_prompt = ""
            rewards = []
            steps = 0
            success = False

            # [START] is emitted BEFORE reset so it is never skipped
            log_start(task=task_name, env_name="orion", model=MODEL_NAME)

            try:
                try:
                    state = await env.reset(task_name=target_task)
                except TypeError:
                    state = await env.reset()

                task_name = state.get("task_name", target_task)
                task_prompt = state.get("task_prompt", "")

                for i in range(1, MAX_STEPS + 1):
                    messages = [
                        {"role": "system", "content": "You are a coding assistant. Respond with the action to take."},
                        {"role": "user", "content": task_prompt},
                    ]

                    error = None
                    try:
                        action_text = call_llm(client, messages)
                    except Exception as e:
                        action_text = ""
                        error = str(e)

                    try:
                        step_result = await env.step(action_text)
                        reward = step_result.reward.final_score
                        done = step_result.done
                    except Exception as e:
                        reward = 0.01
                        done = True
                        error = str(e)

                    rewards.append(reward)
                    steps = i

                    log_step(step=i, action_text=action_text, reward=reward, done=done, error=error)

                    if done:
                        success = reward >= 0.95
                        break
            finally:
                # [END] is ALWAYS emitted, even if reset() or step() throws
                log_end(task=task_name, steps=steps, rewards=rewards)
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())