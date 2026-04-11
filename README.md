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
OrionCLI is a multi-step tool-based coding environment where an RL agent learns to route tasks through optimal LLM pipeline configs using a LinUCB contextual bandit. The environment provides agents with 5 structured tools to inspect, repair, and verify code inside real workspaces to solve 4 production-grade software engineering tasks. The integrated bandit differentiator allows the system to continuously balance exploration and exploitation across 8 pipeline variants, ensuring high efficiency over iterative execution.

## What Makes This Different
- **Multi-step tool workflow:** Agents navigate tasks sequentially via structured JSON tool calls instead of producing one-shot code dumps.
- **LinUCB contextual bandit:** The system dynamically selects from 8 explicit pipeline actions to align capability with task difficulty.
- **Real code execution:** Fixes operate against physical temporary workspace files and are immediately tested against Python interpreters.
- **Production-grade tasks:** Problems emulate real-world architectures, including TTL cache invalidation, rigid retry decorators, stateful circuit breakers, and core syntax repair.

## How Episodes Work
The agent interacts with the environment through a persistent step-by-step loop:
1. Agent calls `list_files` to inspect the current state of the workspace.
2. Agent calls `read_file` to review specific file logic and identify bugs.
3. Agent calls `write_file` to write corrected code to the filesystem.
4. Agent calls `run_tests` to execute graders and verify correctness.
5. Agent calls `submit` to declare success and terminate the episode.

The environment only recalculates and issues non-minimal rewards when the agent executes `write_file` or `submit`. The reward specifically mirrors the improvement delta over the episode (`grader_score - best_score`). Pure observation actions like `list_files` or `read_file` return a baseline reward of `0.01`.

## Available Tools

| Tool | Action Type | Description | Returns |
| :--- | :--- | :--- | :--- |
| `read_file` | `read_file` | Read workspace file | File contents |
| `write_file` | `write_file` | Write fix to file | Bytes written + grader score |
| `run_tests` | `run_tests` | Check current score | Score + best so far |
| `list_files` | `list_files` | List workspace files | Filenames |
| `submit` | `submit` | End episode | Final score |

## Action Space
The environment processes actions strictly modeled as a discriminated JSON union matching 5 valid models.

```json
// Example: list_files
{
  "action_type": "list_files"
}

// Example: read_file
{
  "action_type": "read_file",
  "path": "retry_utils.py"
}

// Example: write_file
{
  "action_type": "write_file",
  "path": "solution.py",
  "content": "def add(a, b):\n    return a + b\n"
}

// Example: run_tests
{
  "action_type": "run_tests"
}

// Example: submit
{
  "action_type": "submit",
  "explanation": "Applied fixed bounds to TTL timestamps matching expiration logic."
}
```

## Observation Space
The environment returns observations carrying complete session context back to the agent.

| Field | Description | Type |
| :--- | :--- | :--- |
| `task_name` | The internal identifier for the current assigned problem | `str` |
| `task_difficulty` | String complexity representation (`easy`, `medium`, `hard`) | `str` |
| `task_prompt` | Instruction provided to the agent explaining the objective | `str` |
| `workspace` | Absolute path locating the active working directory | `str` |
| `history` | List mapping previous step executions and responses | `list[dict]` |
| `total_reward` | Aggregated episodic reward accumulated globally | `float` |
| `steps` | Running count of tools actioned incrementally by the agent | `int` |
| `best_score` | Peak verification score achieved historically within the active session | `float` |
| `available_tools` | Permissible JSON action string types | `list[str]` |
| `last_tool_result` | The formatted execution output returned from the preceding tool | `str` |
| `files_in_workspace` | Current directory tree populated automatically | `list[str]` |

## Tasks

### fix_syntax_error (Easy · seed=7 · budget=5)
- **Prompt:** "Fix the syntax error in broken.py. The file contains a function with a missing colon after the def statement. The function should be named add(a, b) and return a + b."
- **Setup file:** `broken.py` instantiated with a missing trailing colon entirely blocking compilation. 
- **Grader breakdown:** 
  - `0.01`: File missing or Python syntax error
  - `0.15`: `add` function missing entirely
  - `0.5`: Code compiles but output is incorrect
  - `0.99`: Correct execution `add(2,3) == 5`
- **Why it's a good warm-up:** Introduces the agent to the core file manipulation flow natively without presenting complex implementation obstacles.

### debug_memory_leak (Medium · seed=42 · budget=15)
- **Prompt:** "Fix the memory leak in cache_manager.py. The _cache dict grows unbounded because expired entries are never evicted. Add TTL-based expiration so entries older than ttl_seconds are removed on access and during cleanup()." 
- **Setup file:** `cache_manager.py` deployed with traditional `set()` and `get()` definitions omitting eviction tracking logic. 
- **Grader breakdown:**
  - `0.01`: Syntax error or failed to load namespace
  - `0.1`: Missing `CacheManager` class
  - `0.25`: Class defined but fails validation tests natively
  - `0.5`: Successfully implements `cleanup()` removal logic
  - `0.75`: Successfully implements implicit `get()` expiry logic only
  - `0.99`: Implements full TTL-based cache lifecycle correctly
- **What makes it tricky:** Agents cannot mock dictionary sizes; gradients must implement actual chronological timestamp tracking validated via backend temporal mocking.

### fix_retry_logic (Hard · seed=137 · budget=20)
- **Prompt:** "Fix the retry decorator in retry_utils.py. Contains bugs with exception filtering, initial backoff value, and swallowing original exceptions."
- **Setup file:** A decorator that recklessly masks all Exceptions, applies multifold backoff immediately prior to first runs, and masks terminal failures.
- **Grader breakdown:**
  - `0.01`: File completely absent or corrupted
  - `0.25`: Fallback failure with components existing
  - `0.5`: 1 bug successfully resolved
  - `0.75`: 2 bugs successfully resolved
  - `0.99`: 3 bugs successfully resolved
- **Why frontier models struggle:** Meta-programming logic inherently abstracts sequence execution tracking effectively pushing architectural abstraction handling to the extreme.

### implement_circuit_breaker (Hard · seed=999 · budget=25)
- **Prompt:** "Implement a circuit breaker pattern with CLOSED, OPEN, HALF_OPEN states and specific transition rules."
- **Setup file:** Intentionally left blank.
- **Grader breakdown:** 
  - `0.01`: Class fails instantiation
  - `0.2`: `call` method not built natively
  - `0.4`: Class attributes constructed successfully
  - `0.6`: Maintains standard CLOSED state handling correctly
  - `0.8`: Accurately fails calls breaching thresholds tracking OPEN logic
  - `0.99`: Smoothly executes timeout transition delays verifying HALF_OPEN tracking
- **Why it requires multi-step reasoning:** The agent constructs entirely fresh robust state architectures while simultaneously deploying sequences dynamically tested incrementally.

## RL Architecture
- **Algorithm:** LinUCB Contextual Bandit.
- **8 pipeline actions:**
| name | planner_tier | coder_tier | reviewer |
| :--- | :--- | :--- | :--- |
| `fast-fast-no-review` | `fast` | `fast` | `False` |
| `fast-coder-no-review` | `fast` | `coder` | `False` |
| `fast-coder-balanced-review` | `fast` | `coder` | `True` |
| `balanced-coder-balanced-review` | `balanced` | `coder` | `True` |
| `fast-fast-with-review` | `fast` | `fast` | `True` |
| `balanced-heavy-with-review` | `balanced` | `heavy` | `True` |
| `balanced-coder-with-review` | `balanced` | `coder` | `True` |
| `balanced-heavy-no-review` | `balanced` | `heavy` | `False` |

- **State vector:**
  - `intent_type_int`: Categorical mapping describing action mapping bounds (`[0: bug_fix, 1: feature, 2: refactor, 3: explain, 4: test]`).
  - `complexity_int`: Direct assessment scalar `[0: low, 1: medium, 2: high]`.
  - `language_int`: Extracted target extension index. Bounds spanning `[0-10]` (`0: .py`, `1: .js`, etc).
  - `past_iisg_avg`: Floating point calculation mirroring average correctness limits across past iterations.
- **How learning works:** The contextual bandit evaluates pipeline action likelihood predicting potential limits prior to execution. Rewards naturally map to covariance matrices continuously minimizing expectation error live online step-over-step based on specific learning steps boundaries.
- **Weight persistence:** Global context counts properly serialize state representations out into `/app/bandit_weights.npz` exactly via explicit `.save()` tracking outputs.
- **GET /rl/stats endpoint:** Actively serves diagnostic algorithm dimensions indicating internal matrices directly over REST bounds.

## Reward Model
- **Improvement-based:** Validations dynamically yield gradients constrained via the delta equation tracking historical logic: `reward = grader_score - best_score`
- **Efficiency bonus:** Allocates `+0.1` explicitly if the grader validation verifies execution solved successfully traversing `<= step_budget // 2` steps.
- **Penalty logic:**
  - Empty submission: Overrides score instantly mapping to `0.01`.
  - Syntax error: Max output artificially capped statically to `0.15`.
  - Repeated submission: Grader natively scaled backward shifting output limits `-0.05`.
- **Clamping:** Outputs actively bounded exclusively across floating allocations `[0.01, 0.99]`.
- **Termination conditions:** Returns `done` signals if score scales `>=0.95`, internal checks bound strict steps maximums `steps >= budget`, or agent naturally invokes `submit`.

## API Reference
| Method | Path | Description | Key Request/Response Fields |
| :--- | :--- | :--- | :--- |
| **GET** | `/health` | Indicates operational liveness bounds | `None` -> `{"status": "ok"}` |
| **POST** | `/reset` | Scaffolds environment resetting task contexts | `ResetRequest` -> Yields `Observation` Dictionary |
| **POST** | `/step` | Forwards tool dictionary inputs verifying internal environment | `StepRequest` -> Yields `StepResponse` tracking observation limits |
| **GET** | `/state` | Queries internal environment history configurations | `<session_id>` -> Dictionaries serializing state mappings |
| **GET** | `/tasks` | Dumps available internal task lists configurations | `None` -> Model configuration definitions |
| **POST** | `/grader` | Validates grading scripts retroactively locally | `GraderRequest` -> Bound Float logic validations |
| **POST** | `/baseline` | Maps placeholder diagnostic validation requirements | `None` -> Object bounds logic fallbacks |
| **GET** | `/metadata` | Reflects formal internal schemas environments boundaries | `None` -> Structured definition constraints |
| **GET** | `/schema` | Enumerates native Pydantic limits format | `None` -> Maps definitions formats natively |
| **GET** | `/rl/stats` | Fetches active local contextual bounds execution parameters | `None` -> Algorithmic exploration properties |
| **GET** | `/openenv.yaml` | Downloads the compliant open configuration specifications | `None` -> Physical `.yaml` document formatting |

## Baseline Scores
| Task | Difficulty | Expected Score | Notes |
| :--- | :--- | :--- | :--- |
| `fix_syntax_error` | Easy | TBD | |
| `debug_memory_leak` | Medium | TBD | |
| `fix_retry_logic` | Hard | TBD | |
| `implement_circuit_breaker` | Hard | TBD | |

## Setup & Usage

### Environment Variables
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `API_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | Explicit inference gateway definition binding. |
| `MODEL_NAME` | No | `nvidia_nim/qwen/qwen2.5-coder-32b-instruct` | Generative framework string identifier string limits. |
| `HF_TOKEN` | **Yes** | `<None>` | Dedicated huggingface token bearer limits mapping explicit operations. |
| `NVIDIA_NIM_API_KEY` | No | `<None>` | Provider integrations mapping active API deployments. |
| `MAX_SESSIONS` | No | `512` | Strict count evicting legacy connection environments. |

### Run with Docker
```bash
docker build -t orion-cli .
docker run -p 7860:7860 -e HF_TOKEN="your_token_here" orion-cli
```

### Run Inference Script
```bash
export HF_TOKEN="your_token_here"
python inference.py
```

### Example Multi-Step Session
Proper evaluation sequentially relies on dict integrations handling step tracking properly natively via JSON bindings:
```bash
# 1. Start the simulation
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"

# 2. Check architecture layouts
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": {"action_type": "list_files"}}'

# 3. Assess local problems
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": {"action_type": "read_file", "path": "broken.py"}}'

# 4. Integrate physical updates dynamically
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": {"action_type": "write_file", "path": "broken.py", "content": "def add(a, b):\n    return a + b\n"}}'

# 5. Lock submission metrics formally
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action": {"action_type": "submit", "explanation": "Appended fixed syntax boundaries correctly."}}'
```

### Session Management
The environment actively enforces LRU eviction constraints mapping memory thresholds dynamically across simultaneous connections sequentially. Deployments safely preserve execution scaling accurately capping context lengths against fixed boundaries at 512 connections.

## OpenEnv Spec Compliance
- [x] Typed Pydantic v2 discriminated union Action model
- [x] 5 structured tools with typed responses
- [x] Session management (LRU, 512 sessions)
- [x] Deterministic seeds per task
- [x] Improvement-based multi-step rewards
- [x] Weight persistence across restarts
- [x] `exec()` timeout protection (5 seconds)
- [x] `HEALTHCHECK` in Dockerfile
- [x] `openenv validate` passes

## Project Structure
```text
orion-CLI/
├── app/                  # Application specific model constraints
│   ├── __init__.py       # Package definition initializer
│   └── models.py         # PyDantic action and schema logic
├── orion/                # Core RL and environment handler library
│   ├── __init__.py       # Shared module entry point constraints
│   ├── cli/              # Generic presentation handlers
│   ├── config/           # General parameter defaults
│   ├── pipeline/         # Agentic model orchestration variants
│   ├── provider/         # Language model binding mappings
│   ├── rl/               # LinUCB mapping architecture definitions
│   ├── session/          # Environment interaction managers
│   ├── tool/             # Legacy standard interaction utilities
│   └── utils.py          # Helper function logic
├── server/               # API endpoint deployment architectures
│   └── app.py            # FastAPI secondary routing file
├── tasks/                # Evaluated execution graders
│   ├── __init__.py       # Package definition mappings
│   └── task_bank.py      # Structured assignment bounds lists
├── Dockerfile            # Container deployment specification 
├── env.py                # Structural OpenEnv system bounds mapping  
├── inference.py          # Independent diagnostic loop tracking    
├── openenv.yaml          # Formal definition boundary artifact        
├── pyproject.toml        # Dependency control mapping      
├── requirements.txt      # Executable module imports limits        
├── server.py             # Primary FastAPI routing instance          
└── verify_*.py           # Collection diagnostics validators  
```

## License
Apache 2.0