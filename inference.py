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
    display_reward = min(max(float(reward), 0.01), 0.99)
    action_str = action_text.replace('\n', ' ').replace('\r', '')[:80].strip()
    action_str = action_str if action_str else "empty"
    
    if error is None:
        error_val = "null"
    else:
        error_val = str(error).replace(' ', '_').replace('\n', '').replace('\r', '')[:50]
        if not error_val:
            error_val = "null"

    print(f"[STEP] step={step} action={action_str} reward={display_reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(steps, rewards):
    if not rewards:
        rewards = [0.01]
    clamped = [min(max(float(r), 0.01), 0.99) for r in rewards]
    score = sum(clamped) / len(clamped) if clamped else 0.01
    score = min(max(score, 0.01), 0.99)
    success_str = "true" if score >= 0.1 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in clamped)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


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
        for task_name in ["fix_tenacity_retry", "fix_cachetools_ttl", "implement_pybreaker", "fix_async_race"]:
            try:
                task_prompt = ""
                rewards = []
                steps = 0
                success = False

                # [START] is emitted BEFORE reset so it is never skipped
                log_start(task=task_name, env_name="orion", model=MODEL_NAME)

                try:
                    try:
                        state = await env.reset(task_name=task_name)
                    except TypeError:
                        state = await env.reset()

                    task_name = state.get("task_name", task_name)
                    task_prompt = state.get("task_prompt", "")

                    for i in range(1, MAX_STEPS + 1):
                        messages = [
                            {"role": "system", "content": """You are a coding agent. You must respond 
with ONLY a valid JSON action object. Available actions:

{"action_type": "list_files"}
{"action_type": "read_file", "path": "filename.py"}  
{"action_type": "write_file", "path": "filename.py", "content": "...code..."}
{"action_type": "run_tests"}
{"action_type": "submit", "explanation": "what I fixed"}

Strategy: First list_files, then read_file to see the bug,
then write_file with your fix, then submit.

Respond with ONLY the JSON object, no explanation."""},
                            {"role": "user", "content": task_prompt},
                        ]

                        error = None
                        try:
                            action_text = call_llm(client, messages)
                        except Exception as e:
                            action_text = ""
                            error = str(e)

                        # Parse LLM response as JSON action
                        try:
                            import json
                            action = json.loads(action_text)
                        except json.JSONDecodeError:
                            # fallback: treat as write_file to main task file
                            action = {
                                "action_type": "write_file",
                                "path": "solution.py", 
                                "content": action_text
                            }

                        try:
                            step_result = await env.step(action)
                            reward = step_result.reward.final_score
                            done = step_result.done
                            
                            # Checks if step returned an error string containing 404
                            if step_result.tool_response and step_result.tool_response.success is False and "404" in str(step_result.tool_response.result):
                                raise Exception(step_result.tool_response.result)

                        except Exception as e:
                            if "404" in str(e):
                                action = {
                                    "action_type": "write_file",
                                    "path": "solution.py",
                                    "content": action_text
                                }
                                try:
                                    step_result = await env.step(action)
                                    reward = step_result.reward.final_score
                                    done = step_result.done
                                    error = None
                                except Exception as fallback_e:
                                    reward = 0.01
                                    done = True
                                    error = str(fallback_e)
                            else:
                                reward = 0.01
                                done = True
                                error = str(e)

                        # Update bandit with real episode reward
                        try:
                            if hasattr(env, 'bandit') and hasattr(env, 'state_encoder'):
                                complexity = {"Easy": "low", "Medium": "medium", "Hard": "high"}.get(
                                    env.state.task_difficulty if env.state else "Hard", "medium"
                                )
                                intent = {"fix_tenacity_retry": "bug_fix", "fix_cachetools_ttl": "bug_fix",
                                          "fix_async_race": "bug_fix", "implement_pybreaker": "feature"
                                          }.get(task_name, "feature")
                                state_vec = env.state_encoder.encode(
                                    intent_type=intent, complexity=complexity,
                                    prompt=task_name, cwd="."
                                )
                                last_action_idx = getattr(env.bandit, '_last_selected', 0)
                                env.bandit.update(state_vec.to_list(), last_action_idx, reward)
                                env.bandit.save()
                        except Exception:
                            pass

                        rewards.append(reward)
                        steps = i

                        action_display = action.get("action_type", "unknown")[:80]
                        log_step(step=i, action_text=action_display, reward=reward, done=done, error=error)

                        if done:
                            success = reward >= 0.95
                            break
                finally:
                    # [END] is ALWAYS emitted, even if reset() or step() throws
                    log_end(steps=steps, rewards=rewards)
            except Exception as task_err:
                print(f"[END] success=false steps=0 score=0.01 rewards=0.01", flush=True)
                continue
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())