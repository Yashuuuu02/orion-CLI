---
title: OrionCLI
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# OrionCLI — RL-Optimised Agentic Coding Environment

## Overview
OrionCLI is an advanced OpenEnv submission that pushes the boundaries of standard code generation tasks by incorporating an internal reinforcement learning (RL) optimization loop right into the environment. Unlike static one-shot environments, OrionCLI utilizes a LinUCB Contextual Bandit to dynamically select pipeline execution strategies (e.g., fast versus heavy planning, code reviews). It simulates real-world software engineering with multi-step interactive episodes, sophisticated runtime execution grading, and IISG (Instance, Intent, State, Grader) pass rate tracking. 

## What Makes This Different
- **Adaptive Execution Routing**: Leverages an internal LinUCB Contextual Bandit to route code generation through 8 distinct pipeline configurations based on a dynamic 4-feature state vector (intent, complexity, language, historic pass rate).
- **Multi-Step Persistent Episodes**: Agents can interact iteratively with the environment up to a defined `step_budget`. The environment maintains the workspace state and conversation history across steps.
- **Improvement-Based Reward Model**: Multi-level graders supply localized feedback, and agents are rewarded based on improvement deltas over their historic `best_score` within the episode, with significant bonuses for high token efficiency and early completions.
- **Robust Session Management**: A built-in LRU cache dict efficiently manages up to 512 concurrent simulation sessions directly via HTTP APIs.

## Environment Description
The core OpenEnv interface enables an agent to instantiate a temporary `workspace` populated with task-specific setup files (if any). The episode operates interactively: an agent submits commands or generated code, the environment executes the underlying logic inside a restricted AST-compiled Python Sandbox, and then evaluates the actual state of the workspace using task-specific Graders. The episode terminates when the task is perfectly solved (Grader returns `≥ 0.95`), the agent maxes out the `step_budget`, or the agent duplicates submissions indicating it is stuck.

## Action Space
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `prompt` | `str` | `min_length=1` | The agent's submitted action prompt, code, bash command, or narrative. Submitted per `/step`. |

## Observation Space  
| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Internal identifier of the current task. |
| `task_difficulty` | `str` | Difficulty setting, either `Medium` or `Hard`. |
| `task_prompt` | `str` | The exact prompt outlining requirements provided to the agent. |
| `workspace` | `str` | Absolute path to the isolated temporary directory for the session. |
| `history` | `list[dict]` | Contextual history of recent actions, steps, and execution outputs (last 5 entries). |
| `total_reward` | `float` | Cumulative reward accumulated across all steps in the current episode. |
| `steps` | `int` | Current step index inside the multi-step episode. |
| `best_score` | `float` | The maximum correctness score achieved by the underlying grader so far. |

## Tasks

### debug_memory_leak (Medium · seed=42 · step_budget=15)
- **Prompt**: "Fix the memory leak in cache_manager.py. The _cache dict grows unbounded because expired entries are never evicted. Add TTL-based expiration so entries older than ttl_seconds are removed on access and during cleanup()."
- **Setup**: Drops a `cache_manager.py` file containing a naïve dictionary-based cache implementation lacking TTL checks.
- **Grader breakdown**:
  - `0.99`: Successfully cleans up expired keys explicitly on `cleanup()` *and* implicitly returns `None` on `get()`.
  - `0.75`: Handles `get()` expiration but fails `cleanup()`.
  - `0.5`: Handles `cleanup()` but fails `get()`.
  - `0.25`: Sandbox execution failed after basic AST compile.
  - `0.01`: Syntax Errors or file missing.
- **Difficulty rationale**: The agent must modify two separate methods within a class accurately and manage Python timestamps (`time.time()`).

### fix_retry_logic (Hard · seed=137 · step_budget=20)
- **Prompt**: "Fix the retry decorator in retry_utils.py. It has three bugs:\n1. It retries on ALL exceptions instead of only RetryableError\n2. The backoff multiplier is applied before the first retry (should start at 1x)\n3. It swallows the original exception — should re-raise after max retries"
- **Setup**: Drops a `retry_utils.py` file containing a flawed functional closure `@retry` decorator.
- **Grader breakdown**: 
  - `0.99`: Fixes all 3 distinct bugs.
  - `0.75`: Fixes 2 bugs.
  - `0.5`: Fixes 1 bug.
  - `0.25`: Code executes gracefully but no bugs are corrected.
  - `0.01`: File missing or uncompilable.
- **Difficulty rationale**: Decorators are inherently complex. Fixing all requires detailed understanding of closure variables (`nonlocal`), exception handling nuances (`raise e`), and correct flow control inside tight loops.

### implement_circuit_breaker (Hard · seed=999 · step_budget=25)
- **Prompt**: "Implement a circuit breaker pattern in circuit_breaker.py.\nRequirements:\n- CircuitBreaker(failure_threshold=3, recovery_timeout=30)\n- States: CLOSED (normal), OPEN (failing, reject calls), HALF_OPEN (testing)\n- Transitions: CLOSED→OPEN after failure_threshold failures\n- Transitions: OPEN→HALF_OPEN after recovery_timeout seconds\n- Transitions: HALF_OPEN→CLOSED on success, HALF_OPEN→OPEN on failure\n- call(func, *args) method that enforces the circuit state\n- Raise CircuitBreakerOpen when circuit is OPEN"
- **Setup**: None (pure greenfield implementation).
- **Grader breakdown**:
  - `0.99`: Meets all requirements, executing full `CLOSED -> OPEN -> HALF_OPEN` transitions strictly tested through mocked runtime flows.
  - `0.8`: correctly evaluates the `CLOSED -> OPEN` phase and properly blocks execution returning `CircuitBreakerOpen`.
  - `0.6`: Permits successful code execution when initially `CLOSED`.
  - `0.4`: Class exists and is structurally somewhat valid.
  - `0.01`: Fails basic compilation or file omitted entirely.
- **Difficulty rationale**: Pure from-scratch architecture problem testing State Machine logic. Tricky to manage asynchronous timeouts synchronously during evaluation (`time.sleep` mocking). 

## RL Architecture
- **Algorithm**: `LinUCB Contextual Bandit`
- **State Vector**: Encoded as 4 continuous features:
  - `intent_type` (`bug_fix`, `feature`, `refactor`)
  - `complexity` (`low`, `medium`, `high`)
  - `language` (detected codebase dominant language format integer)
  - `past_iisg` (rolling window of the last 5 average IISG scores)
- **Update Mechanism**: Online weight update. Every episode steps generates a temporal error delta causing `BanditWeights` to adjust exploration variables iteratively, saved locally to `~/.orion/bandit_weights.json`.
- **Visibility**: Hit `/rl/stats` dynamically at runtime to visualize UCB learning coefficients.

| Action Name | Planner Tier | Coder Tier | Reviewer |
|---|---|---|---|
| fast-fast-no-review | fast | fast | False |
| fast-coder-no-review | fast | coder | False |
| fast-coder-balanced-review | fast | coder | True | 
| balanced-coder-balanced-review | balanced | coder | True |
| fast-fast-with-review | fast | fast | True |
| balanced-heavy-with-review | balanced | heavy | True |
| balanced-coder-with-review | balanced | coder | True |
| balanced-heavy-no-review | balanced | heavy | False |

## Reward Model
| Field | Description |
|---|---|
| `correctness` | The absolute evaluation score (`0.0` - `1.0`) obtained straight from the Sandbox Grader. |
| `efficiency` | A computed float `0.0` - `1.0` dynamically balancing tokens used vs general `token_budget` limits. |
| `final_score` | Computes Delta Improvement Rewards against the internal episode `best_score`. Max value is bounded to `0.99`. |

**Efficiency Bonus:** Any step finishing with absolute `correctness >= 0.95` inside half the expected `step_budget` natively earns a `.1` additive early-solver efficiency bonus.  
**Termination Conditions:** `is_done` returns `True` explicitly when max runtime passes, `correctness` goes above `0.95`, or identical successive submission hashes denote loop freezes.

## API Reference
| Method | Endpoint | Description |
|---|---|---|
| **GET** | `/health` | Rapid liveness proxy check for Docker. |
| **POST** | `/reset` | Spawns sandbox, picks task logic, boots `Observation` space session. |
| **POST** | `/step` | Executes AST actions asynchronously returning `StepResponse`. |
| **GET** | `/state` | Returns pure real-time data frame of session internals. |
| **GET** | `/tasks` | Dumps array of seeded `Medium` / `Hard` Tasks. |
| **POST** | `/grader` | Takes offline workspace volumes returning exact validation trajectory output. |
| **POST** | `/baseline` | Dummy endpoint exposing current models base scores. |
| **GET** | `/metadata` | Static metadata variables mapping OpenEnv architecture constraints. |
| **GET** | `/schema` | Dynamic reflection resolving Pydantic v2 underlying JSON schemas. |
| **GET** | `/rl/stats` | JSON payload defining Bandit learning coefficients. |
| **GET** | `/openenv.yaml`| Standard validator compliance requirement file. |

## Baseline Scores
| Task Name | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| debug_memory_leak | Medium | TBD | Currently unmeasured directly by baseline agent. Run `inference.py`. |
| fix_retry_logic | Hard | TBD | Currently unmeasured directly by baseline agent. Run `inference.py`. |
| implement_circuit_breaker | Hard | TBD | Currently unmeasured directly by baseline agent. Run `inference.py`. |

## Setup & Usage

### Environment Variables
| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `"https://integrate.api.nvidia.com/v1"` | LLM Server base URI overwrite for completions. |
| `MODEL_NAME` | No | `"nvidia_nim/qwen/qwen2.5-coder-32b-instruct"` | Name of model deployed in Provider SDK setup. |
| `HF_TOKEN` | Yes | None | HuggingFace Token access. |
| `NVIDIA_NIM_API_KEY` | Yes | (empty) | Default Orion environment API key requirement. |
| `MAX_SESSIONS` | No | `512` | Absolute max quantity of LRU sessions cache. |

### Run with Docker
```bash
docker build -t orion-cli .
docker run -p 7860:7860 -e NVIDIA_NIM_API_KEY="your-api-key" orion-cli
```

### Run Inference Script
```bash
export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="nvidia_nim/qwen/qwen2.5-coder-32b-instruct"
export HF_TOKEN="your-hf-token"
python inference.py
```

### Session Management
Each call to `/reset` independently creates a new globally referenced UUID representing a sandbox namespace. By passing `{"session_id": "YOUR_UUID"}` iteratively to `/step`, the underlying server ensures history, Bandit variables, and workspace contents remain safely separated in the LRU `_sessions` OrderedDict. 

### API Example
```bash
# 1. Start session
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

# 2. Extract given `session_id` from JSON

# 3. Step forward
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"prompt": "print(1)", "session_id": "78b-..."}'

# 4. View environment history
curl "http://localhost:7860/state?session_id=78b-..."
```

## OpenEnv Spec Compliance
- [x] Typed Pydantic v2 models (`Observation`, `StepAction`, `Reward`, `StepResponse`)
- [x] Session management (LRU OrderedDict, 512 max constraint)
- [x] Deterministic validation seeds per task ensuring reproduciability
- [x] Built-in `/grader` offline evaluation endpoint for trajectory replay  
- [x] Environment `openenv validate` compliance passing
- [x] Native `HEALTHCHECK` layer in Dockerfile image
- [x] Multi-step execution episodes returning improvement-based deltas

## Project Structure
```text
orion-CLI/
├── app/
│   └── models.py            # Pydantic v2 Type Definitions for OpenAPI specification
├── orion/
│   ├── pipeline/            # Execution routing handlers
│   ├── provider/            # LLM generation orchestration classes
│   ├── rl/
│   │   ├── bandit.py        # LinUCB stateful learning node & equations
│   │   └── state_encoder.py # Multi-feature state logic compiler
│   └── tool/                # Safe sandbox tools for edit/read actions
├── tasks/
│   └── task_bank.py         # Seeded tasks array, definitions, and restricted Sandbox AST Grader logic 
├── Dockerfile               # Root environment virtualization with HEALTHCHECK logic
├── env.py                   # State environment interface
├── inference.py             # Sandbox baseline evaluator entry script
├── openenv.yaml             # Manifest mapping
└── server.py                # Concurrent FastAPI service gateway
```

## License
Apache 2.0