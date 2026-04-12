---
title: OrionCLI
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# OrionCLI — RL-Optimised Agentic Coding Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1%20Spec-green?style=for-the-badge)]
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=for-the-badge)]
[![Tests](https://img.shields.io/badge/Tests-6%20passing-brightgreen?style=for-the-badge)]
[![Apache](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)]
[![HF Space](https://img.shields.io/badge/HF%20Space-Live-yellow?style=for-the-badge)](https://huggingface.co/spaces/YASHUUU8/OrionCLI)

*Built for the Meta × Hugging Face OpenEnv Hackathon*

> 🔴 **Live Dashboard:** View real-time episode data and RL bandit stats at [/dashboard](https://yashuuu8-orioncli.hf.space/dashboard)

## Overview

OrionCLI is not just an OpenEnv submission — it's a complete RL-optimised agentic coding assistant built in 3 phases:
- **Phase 1: Production TUI** (Textual, SQLite sessions, streaming)
- **Phase 2: Agentic pipeline** (C01 intent → IISG validation → real file execution)
- **Phase 3: RL layer + OpenEnv interface**

Each `/step` call runs the complete Phase 2 agentic pipeline internally — intent classification, multi-model agentic loop, real file execution, and IISG validation — not a single LLM call.

The bandit learns from real usage — not synthetic data:
- `bug_fix` + `low complexity` → fast pipeline, skip reviewer
- `feature` + `high complexity` → full pipeline, heavy model
- `explain` tasks → cheapest config, reviewer irrelevant

The 8 actions represent real pipeline configurations that affect actual LLM model tier, planner usage, and review step.

## 🚫 The Problem → ✅ The Solution

Current LLM coding agents fail in production because:
- They make one-shot code dumps without reading the codebase
- They have no mechanism to learn which approach works per task type  
- They treat all bugs equally — a race condition needs different 
  reasoning than a retry logic bug

**OrionCLI** is the training environment that closes this gap. 
A standardised RL environment where agents learn to *investigate*, 
*plan*, and *fix* production bugs through structured tool calls, 
while a LinUCB bandit learns which pipeline configuration produces 
the best fixes per task category.

### Key Capabilities

| Capability | Detail |
| :--- | :--- |
| **Multi-step tool workflow** | Agents navigate tasks step-by-step via structured JSON tool calls — not one-shot code dumps |
| **Sandboxed code execution** | Graders run submitted code in a restricted namespace with a 5-second `exec()` timeout |
| **Production-grade tasks** | Grounded against real libraries (tenacity, cachetools, pybreaker, asyncio) |
| **OpenEnv compliant** | Typed Pydantic v2 models, session management, deterministic seeds, `openenv.yaml` manifest |

## Real-World Grounding

OrionCLI tasks are modeled after bugs and patterns from 
production Python libraries:

| Task | Based On | Real Library |
|------|----------|--------------|
| fix_tenacity_retry | Retry decorator bugs | [tenacity](https://github.com/jd/tenacity) |
| fix_cachetools_ttl | TTL cache expiry bugs | [cachetools](https://github.com/tkem/cachetools) |
| implement_pybreaker | Circuit breaker pattern | [pybreaker](https://github.com/danielfm/pybreaker) |
| fix_async_race | asyncio race conditions | Production async services |

This grounds the environment in real failure modes that 
production agents would actually encounter.



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

### `fix_tenacity_retry` · Medium · seed 42 · budget 15

**Prompt:** Fix the retry decorator in `retry_utils.py`. It has two bugs:
1. It does not check `retry_on_exception` before retrying.
2. It returns `None` instead of raising `RetryError` after max attempts.

| Score | Condition |
| :--- | :--- |
| `0.01` | File missing or `SyntaxError` |
| `0.25` | Class exists but both bugs present |
| `0.50` | Returns None but retry checks `retry_on_exception` |
| `0.75` | `RetryError` raised but `retry_on_exception` check missed |
| `0.99` | All bugs fixed |

**Scoring rationale:** Linear partial credit because bugs can be fixed independently.

---

### `fix_cachetools_ttl` · Medium · seed 137 · budget 20

**Prompt:** Fix the TTL cache in `cache_manager.py`. It has two bugs:
1. `__getitem__` never checks if the entry has expired.
2. `expire()` is broken/empty but should remove expired entries.

| Score | Condition |
| :--- | :--- |
| `0.01` | File missing or `SyntaxError` |
| `0.25` | Class exists but both bugs present |
| `0.50` | `expire()` works but `__getitem__` doesn't check TTL |
| `0.75` | `__getitem__` handles expiry but `expire()` is missing |
| `0.99` | All bugs fixed |

**Scoring rationale:** `__getitem__` and `expire()` are independent fixes.

---

### `implement_pybreaker` · Hard · seed 999 · budget 25

**Prompt:** Implement a circuit breaker compatible with the pybreaker library interface in `circuit_breaker.py` with `CLOSED → OPEN → HALF_OPEN` state transitions.

| Score | Condition |
| :--- | :--- |
| `0.01` | Class fails to instantiate |
| `0.20` | `call()` method missing |
| `0.40` | Class attributes correctly constructed |
| `0.60` | CLOSED state handles successful calls |
| `0.80` | Transitions to OPEN after `failure_threshold` failures |
| `0.99` | Transitions to HALF_OPEN after `recovery_timeout`, recovers on success |

**Scoring rationale:** Tiers follow state machine implementation depth — instantiation (0.40) → happy path (0.60) → failure detection (0.80) → full recovery path (0.99). Each tier is a meaningful checkpoint in implementing a correct state machine.

> ⚠️ **Why agents struggle:** State machines require tracking 
> transitions across multiple calls. A one-shot agent writes 
> CLOSED→OPEN correctly but forgets HALF_OPEN recovery. 
> Multi-step tool workflow forces the agent to test each 
> transition before submitting.

---

### `fix_async_race` · Hard · seed 777 · budget 30

**Prompt:** Fix the read-modify-write race condition in `SharedCounter.increment()` within `async_worker.py`.

| Score | Condition |
| :--- | :--- |
| `0.01` | File missing or `SyntaxError` |
| `0.25` | Class exists but race condition persists |
| `0.50` | Code runs but counter is lost/corrupted |
| `0.99` | Race condition fixed (e.g. using `asyncio.Lock()`) |

**Scoring rationale:** Async bugs require explicit locks or atomic updates; running with partial locks usually fails completely (0.50) or succeeds fully (0.99).

> ⚠️ **Why agents struggle:** asyncio.sleep(0) as a yield point 
> is non-obvious. Most agents add a Lock but put it in the wrong 
> place. The run_tests tool lets agents verify their fix actually 
> prevents the race before submitting.

---

## 📊 Baseline Scores

Measured from inference.py first-attempt results using 
`nvidia_nim/qwen/qwen2.5-coder-32b-instruct`:

| Task | Difficulty | Zero-Shot Score | Optimal Score | Training Gap |
|------|-----------|----------------|---------------|--------------|
| `fix_tenacity_retry` | Medium | 0.25 | 0.99 | **0.74** |
| `fix_cachetools_ttl` | Medium | 0.25 | 0.99 | **0.74** |
| `implement_pybreaker` | Hard | 0.20 | 0.99 | **0.79** |
| `fix_async_race` | Hard | 0.50 | 0.99 | **0.49** |

> **Training gap** = how much room exists for RL improvement.
> A zero-shot score of 0.20 on `implement_pybreaker` means 
> the agent gets the class structure right but misses the 
> state machine transitions — exactly the multi-step reasoning 
> that RL training improves.
> 
> `fix_async_race` scores highest zero-shot (0.50) because 
> agents recognize Lock patterns but often miss that 
> `asyncio.sleep(0)` is the yield point causing the race.

---

## Reward Design Philosophy

OrionCLI uses improvement-based delta rewards (`grader_score - best_score`) instead of absolute scores to ensure that agents are rewarded only for progress, preventing models from "hacks" that repeat high-scoring states without further discovery. We clamp rewards to the `(0.01, 0.99)` range to maintain a non-zero gradient for RL training and avoid numerical instability at the limits of the sigmoid/logistic space. An efficiency bonus exists to steer the agent towards the shortest path to completion, rewarding optimal tool usage. Together, these design choices create a dense, informative signal that allows the LinUCB bandit to convergence on optimal pipeline actions across varying task difficulties.

## Why This Design

**Why improvement-based delta rewards?**
Absolute scores don't signal progress. An agent scoring 0.50 
twice learns nothing. Delta rewards mean every improvement 
gets positive signal, every regression gets none.

**Why (0.01, 0.99) range?**
Unlike classification tasks where 0.0 and 1.0 are valid, 
automated code graders cannot claim perfect certainty. 
0.01 ensures the RL agent always receives gradient signal. 
0.99 prevents false "solved" termination.

**Why real library tasks (tenacity, cachetools, pybreaker)?**
Synthetic bugs are easy to pattern-match. Real library bugs 
require understanding the intended behavior, not just the 
syntax. An agent that fixes tenacity's retry logic has learned 
something transferable to production codebases.

**Why 8 pipeline actions?**
The action space spans the meaningful dimensions of LLM 
pipeline configuration: model capability (fast/balanced/heavy), 
planning depth (with/without planner), and output quality 
(with/without reviewer). 8 actions = 2³ combinations of 
these three binary choices.

**Why LinUCB over simpler bandits?**
LinUCB uses the task context (intent, complexity, language) 
to generalize across tasks. A simpler ε-greedy bandit would 
treat every task independently. LinUCB learns that 
"bug_fix + high complexity" always benefits from heavy+review 
regardless of which specific bug it is.

## Why These Reward Weights?

The reward formula R = 0.8 × iisg_pass_rate + 0.1 × syntax_valid 
+ 0.1 × token_efficiency is derived from production code generation 
priorities:

- **0.8 weight on IISG**: Intent-Instruction-Solution-Grade 
  validation measures whether the generated code actually solves 
  the described problem — the most important signal
- **0.1 weight on syntax**: Syntactically valid code is a 
  prerequisite, but a low bar — weighted low intentionally
- **0.1 weight on token efficiency**: Production systems care 
  about cost — a solution using 3000 tokens when 500 suffice 
  is objectively worse

This formula emerged from real usage patterns, not arbitrary 
assignment. It matches how senior engineers evaluate code: 
does it work (IISG) > does it compile (syntax) > is it efficient 
(tokens).

## Grader Design Philosophy

The grader scoring tiers are not arbitrary — they follow a 
principled partial-credit system inspired by real code review:

- **0.99 (near-perfect)**: Objective achieved — solution works 
  in all tested cases. Not 1.0 because no automated grader can 
  claim perfect certainty.
- **0.75**: Partial fix — one dimension correct, one missing. 
  Like a PR that fixes the happy path but misses edge cases.
- **0.50**: Structural fix — correct approach, wrong semantics. 
  Like code that compiles and runs but returns wrong values.
- **0.25**: Minimal signal — evidence of understanding without 
  correct execution.
- **0.01**: Floor signal — prevents zero reward, ensures RL 
  agent always gets gradient signal.

The (0.01, 0.99) range is deliberately open — 0.0 would mean 
"no information" and 1.0 would mean "perfect certainty", 
neither of which is appropriate for automated graders.

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
  -d '{"task_name": "fix_tenacity_retry"}'

# 2. List workspace files
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "list_files"}}'

# 3. Read the buggy file
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "read_file", "path": "retry_utils.py"}}'

# 4. Write the fix
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "write_file", "path": "retry_utils.py", "content": "def add(a, b):\n    return a + b\n"}}'

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
- [x] 6 golden trajectory tests (tests/test_graders.py)
- [x] Bandit weights pre-seeded at Docker build time (200 episodes)
- [x] Real library grounding (tenacity, cachetools, pybreaker)
- [x] Step cost pressure on observation actions

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