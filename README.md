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

OrionCLI is a multi-step, tool-based coding environment built to the [OpenEnv](https://github.com/openenv) specification. An RL agent receives a software-engineering task, navigates the workspace through five structured JSON tools, and iteratively repairs code until it passes an automated grader. A **LinUCB contextual bandit** sits above the agent loop, selecting from 8 pipeline configurations to match model capability with task difficulty.

### Key Capabilities

| Capability | Detail |
| :--- | :--- |
| **Multi-step tool workflow** | Agents navigate tasks step-by-step via structured JSON tool calls — not one-shot code dumps |
| **LinUCB contextual bandit** | Dynamically selects from 8 pipeline actions (planner × coder × reviewer) per task context |
| **Sandboxed code execution** | Graders run submitted code in a restricted namespace with a 5-second `exec()` timeout |
| **Production-grade tasks** | TTL cache invalidation, retry decorator bugs, stateful circuit breakers, and syntax repair |
| **OpenEnv compliant** | Typed Pydantic v2 models, session management, deterministic seeds, `openenv.yaml` manifest |

---

## How Episodes Work

```
┌─ reset(task_name) ─────────────────────────────────────────────┐
│  Scaffold workspace · write setup files · return Observation   │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
              ┌─── step loop (up to step_budget) ───┐
              │                                     │
              │   1. list_files  → see workspace    │
              │   2. read_file   → inspect code     │
              │   3. write_file  → apply fix        │
              │   4. run_tests   → check score      │
              │   5. submit      → end episode      │
              │                                     │
              └──────────┬──────────────────────────┘
                         ▼
         done = score ≥ 0.95 | steps ≥ budget | submit
```

- **Observation tools** (`list_files`, `read_file`) return a baseline reward of `0.01`.
- **Mutating tools** (`write_file`, `submit`) trigger the grader and return an improvement-based reward: `grader_score − best_score`.
- `run_tests` executes the grader for informational feedback but returns `0.01` reward.

---

## Available Tools

| Tool | `action_type` | Description | Returns |
| :--- | :--- | :--- | :--- |
| **Read file** | `read_file` | Read a workspace file (capped at 3 000 chars) | File contents |
| **Write file** | `write_file` | Write content to a file, then auto-grade | Bytes written + grader score |
| **Run tests** | `run_tests` | Run the grader without modifying files | Current score + best so far |
| **List files** | `list_files` | List all files in the workspace | Filenames |
| **Submit** | `submit` | Run final grader and end the episode | Final score |

## Action Space

Actions are JSON objects matching a Pydantic v2 [discriminated union](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions) keyed on `action_type`:

```jsonc
// Inspect workspace
{"action_type": "list_files"}

// Read a specific file
{"action_type": "read_file", "path": "retry_utils.py"}

// Write a fix
{"action_type": "write_file", "path": "broken.py", "content": "def add(a, b):\n    return a + b\n"}

// Check current grader score
{"action_type": "run_tests"}

// Submit and end episode
{"action_type": "submit", "explanation": "Fixed missing colon in function definition."}
```

> **Legacy fallback:** If `action` is a plain string instead of a dict, the environment wraps it as a `write_file` to `solution.py`.

## Observation Space

Each `step()` returns a `StepResponse` containing:

| Field | Type | Description |
| :--- | :--- | :--- |
| `task_name` | `str` | Internal identifier for the active task |
| `task_difficulty` | `str` | `Easy`, `Medium`, or `Hard` |
| `task_prompt` | `str` | Natural-language instruction for the agent |
| `workspace` | `str` | Absolute path to the working directory |
| `history` | `list[dict]` | Last 5 step records (`step`, `action_type`, `tool_result`, `reward`) |
| `total_reward` | `float` | Cumulative reward sum across all steps (can exceed 1.0) |
| `steps` | `int` | Number of steps taken so far |
| `best_score` | `float` | Highest grader score achieved this episode |
| `available_tools` | `list[str]` | `["read_file","write_file","run_tests","list_files","submit"]` |
| `last_tool_result` | `str` | Formatted output from the preceding tool (≤ 500 chars) |
| `files_in_workspace` | `list[str]` | Current file listing |

---

## Tasks

### `fix_syntax_error` · Easy · seed 7 · budget 5

**Prompt:** Fix the syntax error in `broken.py` — the function `add(a, b)` is missing its colon.

| Score | Condition |
| :--- | :--- |
| `0.01` | File missing or `SyntaxError` |
| `0.15` | Compiles but `add` function is absent |
| `0.50` | `add` exists but returns wrong result |
| `0.99` | `add(2, 3) == 5` |

**Scoring rationale:** 0.99 = objective achieved; 0.50 = correct structure but wrong semantics; 0.15 = parseable but missing function; 0.01 = floor to prevent zero reward signal to RL agent.

---

### `debug_memory_leak` · Medium · seed 42 · budget 15

**Prompt:** Fix the unbounded `_cache` dict in `cache_manager.py` — add TTL-based expiration on `get()` and in `cleanup()`.

| Score | Condition |
| :--- | :--- |
| `0.01` | Syntax error or failed to load |
| `0.10` | `CacheManager` class missing |
| `0.25` | Class exists but fails validation |
| `0.50` | `cleanup()` correctly removes expired entries |
| `0.75` | `get()` correctly returns `None` for expired entries |
| `0.99` | Full TTL lifecycle — both `get()` expiry and `cleanup()` removal |

**Scoring rationale:** get() and cleanup() are scored independently (0.75 and 0.50) because they are orthogonal fixes — an agent fixing one method still demonstrates partial TTL understanding. Both required for 0.99 because either path alone still leaks.

---

### `fix_retry_logic` · Hard · seed 137 · budget 20

**Prompt:** Fix three bugs in the `retry` decorator in `retry_utils.py`:
1. Catches **all** exceptions — should only retry on `RetryableError`
2. Backoff delay starts at `backoff` instead of `1.0`
3. Swallows the final exception (returns `None`) — should re-raise

| Score | Condition |
| :--- | :--- |
| `0.01` | File missing or `SyntaxError` |
| `0.25` | `retry` / `RetryableError` not found |
| `0.50` | 1 bug fixed |
| `0.75` | 2 bugs fixed |
| `0.99` | All 3 bugs fixed |

**Scoring rationale:** Linear partial credit per bug fixed (0.50/0.75/0.99) because the 3 bugs are independent — any ordering of fixes is valid. Each bug fixed demonstrates a distinct debugging skill.

---

### `implement_circuit_breaker` · Hard · seed 999 · budget 25

**Prompt:** Implement a circuit breaker in `circuit_breaker.py` with `CLOSED → OPEN → HALF_OPEN` state transitions.

| Score | Condition |
| :--- | :--- |
| `0.01` | Class fails to instantiate |
| `0.20` | `call()` method missing |
| `0.40` | Class attributes correctly constructed |
| `0.60` | CLOSED state handles successful calls |
| `0.80` | Transitions to OPEN after `failure_threshold` failures |
| `0.99` | Transitions to HALF_OPEN after `recovery_timeout`, recovers on success |

**Scoring rationale:** Tiers follow state machine implementation depth — instantiation (0.40) → happy path (0.60) → failure detection (0.80) → full recovery path (0.99). Each tier is a meaningful checkpoint in implementing a correct state machine.

---

## Reward Design Philosophy

OrionCLI uses improvement-based delta rewards (`grader_score - best_score`) instead of absolute scores to ensure that agents are rewarded only for progress, preventing models from "hacks" that repeat high-scoring states without further discovery. We clamp rewards to the `(0.01, 0.99)` range to maintain a non-zero gradient for RL training and avoid numerical instability at the limits of the sigmoid/logistic space. An efficiency bonus exists to steer the agent towards the shortest path to completion, rewarding optimal tool usage. Together, these design choices create a dense, informative signal that allows the LinUCB bandit to convergence on optimal pipeline actions across varying task difficulties.

---

## Reward Model

| Component | Formula / Rule |
| :--- | :--- |
| **Base reward** | `grader_score − best_score` (floored at `0.01`) |
| **Efficiency bonus** | `+0.1` if `grader_score ≥ 0.95` and `steps ≤ budget // 2` |
| **Observation-only** | `0.01` for `list_files`, `read_file`, `run_tests` |
| **Clamping** | All rewards clamped to `[0.01, 0.99]` |
| **Termination** | `done = True` when `score ≥ 0.95`, `steps ≥ budget`, or agent calls `submit` |

### Reward Breakdown (per step)

The `Reward` model returned in `StepResponse` has three fields:

| Field | Description |
| :--- | :--- |
| `correctness` | Clamped grader score `[0.01, 0.99]` |
| `efficiency` | `1.0 − (steps / budget)`, clamped `[0.01, 0.99]` |
| `final_score` | The combined reward value for this step |

---

## RL Architecture

**Algorithm:** LinUCB Contextual Bandit (`orion/rl/bandit.py`)

### Pipeline Actions (8 arms)

| Name | Planner | Coder | Reviewer |
| :--- | :--- | :--- | :--- |
| `fast-fast-no-review` | fast | fast | ✗ |
| `fast-coder-no-review` | fast | coder | ✗ |
| `fast-coder-balanced-review` | fast | coder | ✓ |
| `balanced-coder-balanced-review` | balanced | coder | ✓ |
| `fast-fast-with-review` | fast | fast | ✓ |
| `balanced-heavy-with-review` | balanced | heavy | ✓ |
| `balanced-coder-with-review` | balanced | coder | ✓ |
| `balanced-heavy-no-review` | balanced | heavy | ✗ |

### State Vector (4 features)

| Feature | Encoding |
| :--- | :--- |
| `intent_type_int` | `{0: bug_fix, 1: feature, 2: refactor, 3: explain, 4: test}` |
| `complexity_int` | `{0: low, 1: medium, 2: high}` |
| `language_int` | File extension index `0–10` (`.py` = 0, `.js` = 1, …) |
| `past_iisg_avg` | Running average of past correctness scores |

### Learning

- **UCB selection:** `score = w·x + α √(ln(N) / n)` — balances exploitation (weight · state) with exploration.
- **Update rule:** Online SGD with learning rate `1 / (count + 1)`.
- **Weight persistence:** Serialised to `/app/bandit_weights.npz` via `numpy.savez()`. Pre-seeded at Docker build time.

---

## API Reference

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness check → `{"status": "ok"}` |
| `POST` | `/reset` | Create session, scaffold workspace, return initial observation |
| `POST` | `/step` | Execute one tool action, return `StepResponse` |
| `GET` | `/state` | Return current environment state for a session |
| `GET` | `/tasks` | List all available tasks (name, difficulty, prompt) |
| `POST` | `/grader` | Re-run a grader on an existing workspace |
| `POST` | `/baseline` | Placeholder for baseline evaluation info |
| `GET` | `/metadata` | Environment metadata and schema info |
| `GET` | `/schema` | JSON schemas for `Action`, `Observation`, `Reward` models |
| `GET` | `/rl/stats` | LinUCB bandit diagnostics (action counts, best actions, α) |
| `GET` | `/openenv.yaml` | Download the OpenEnv manifest |

---

## Setup & Usage

### Environment Variables

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `HF_TOKEN` | **Yes** | — | Hugging Face token for inference |
| `API_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | LLM inference endpoint |
| `MODEL_NAME` | No | `nvidia_nim/qwen/qwen2.5-coder-32b-instruct` | Model identifier |
| `NVIDIA_NIM_API_KEY` | No | — | Alternative API key for NVIDIA NIM |
| `MAX_SESSIONS` | No | `512` | Maximum concurrent sessions (LRU eviction) |

### Run with Docker

```bash
docker build -t orion-cli .
docker run -p 7860:7860 -e HF_TOKEN="your_token_here" orion-cli
```

### Run Locally

```bash
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python server.py                  # starts on http://0.0.0.0:7860
```

### Run Inference Script

```bash
export HF_TOKEN="your_token_here"
python inference.py
```

The inference script runs all 4 tasks sequentially, logging `[START]`, `[STEP]`, and `[END]` entries for each.

### Example Multi-Step Session

```bash
# 1. Reset — create a new session
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "fix_syntax_error"}'

# 2. List workspace files
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "list_files"}}'

# 3. Read the buggy file
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "read_file", "path": "broken.py"}}'

# 4. Write the fix
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "write_file", "path": "broken.py", "content": "def add(a, b):\n    return a + b\n"}}'

# 5. Submit
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "submit", "explanation": "Added missing colon to function definition."}}'
```

### Session Management

The server maintains an `OrderedDict` of active sessions capped at `MAX_SESSIONS` (default 512). When capacity is reached, the **oldest session** is evicted (LRU). Each `/reset` call creates a fresh `OpenEnv` instance and a new temporary workspace.

---

## OpenEnv Compliance

- [x] Typed Pydantic v2 discriminated union `Action` model
- [x] 5 structured tools with typed `ToolResponse`
- [x] LRU session management (512 max)
- [x] Deterministic seeds per task
- [x] Improvement-based multi-step rewards clamped to `[0.01, 0.99]`
- [x] LinUCB weight persistence across container restarts
- [x] Sandboxed `exec()` with 5-second timeout
- [x] `HEALTHCHECK` in Dockerfile
- [x] Non-root container user (`orion:1000`)
- [x] `openenv.yaml` manifest served at `/openenv.yaml`

---

## Project Structure

```
orion-CLI/
├── app/
│   ├── __init__.py
│   └── models.py              # Pydantic v2 models (Observation, Action, Reward, StepResponse)
├── orion/
│   ├── cli/                   # Terminal UI (Textual)
│   ├── config/                # Default parameters
│   ├── pipeline/              # PipelineRunner — planner/coder/reviewer orchestration
│   ├── provider/              # LLM provider abstraction (NVIDIA NIM, OpenAI-compatible)
│   ├── rl/
│   │   ├── bandit.py          # LinUCB contextual bandit (8 arms, 4-dim state)
│   │   └── state_encoder.py   # Intent/complexity/language → feature vector
│   ├── session/               # Session lifecycle helpers
│   ├── tool/                  # Read, Write, Edit, Grep tool implementations
│   └── utils.py
├── scripts/
│   └── preseed_bandit.py      # Seeds bandit weights at Docker build time
├── server/
│   └── app.py                 # Secondary FastAPI router
├── tasks/
│   ├── __init__.py
│   └── task_bank.py           # 4 tasks + graders + sandboxed exec
├── Dockerfile                 # Python 3.11-slim, non-root, HEALTHCHECK
├── env.py                     # OpenEnv — core reset/step loop
├── inference.py               # Autonomous evaluation across all tasks
├── openenv.yaml               # OpenEnv specification manifest
├── pyproject.toml             # Package metadata (requires Python ≥ 3.11)
├── requirements.txt           # Pinned dependencies
├── server.py                  # Primary FastAPI app — all API endpoints
└── verify_*.py                # Diagnostic / validation scripts
```

## License

Apache 2.0