# OrionCLI — Architecture Document

> **Version:** 1.0.0  
> **Last Updated:** 2026-04-12  
> **Python:** ≥ 3.11  

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Core Subsystems](#4-core-subsystems)
   - 4.1 [OpenEnv Environment (`env.py`)](#41-openenv-environment-envpy)
   - 4.2 [FastAPI Server (`server.py`)](#42-fastapi-server-serverpy)
   - 4.3 [Inference Script (`inference.py`)](#43-inference-script-inferencepy)
   - 4.4 [Task Bank (`tasks/`)](#44-task-bank-tasks)
   - 4.5 [Pipeline Engine (`orion/pipeline/`)](#45-pipeline-engine-orionpipeline)
   - 4.6 [Reinforcement Learning (`orion/rl/`)](#46-reinforcement-learning-orionrl)
   - 4.7 [LLM Provider (`orion/provider/`)](#47-llm-provider-orionprovider)
   - 4.8 [Tool System (`orion/tool/`)](#48-tool-system-oriontool)
   - 4.9 [Session & Config (`orion/session/`, `orion/config/`)](#49-session--config)
   - 4.10 [Terminal UI (`orion/cli/`)](#410-terminal-ui-orioncli)
   - 4.11 [Data Models (`app/models.py`)](#411-data-models-appmodelspy)
5. [Data Flow](#5-data-flow)
   - 5.1 [Episode Lifecycle](#51-episode-lifecycle)
   - 5.2 [Reward Computation](#52-reward-computation)
   - 5.3 [Bandit Learning Loop](#53-bandit-learning-loop)
6. [Deployment Architecture](#6-deployment-architecture)
7. [OpenEnv Compliance Contract](#7-openenv-compliance-contract)
8. [Dependency Graph](#8-dependency-graph)

---

## 1. Project Identity

OrionCLI is a **dual-purpose** system:

| Role | Description |
|---|---|
| **Agentic Coding Assistant** | A Textual-based terminal UI that lets developers interact with LLMs through a tool-augmented pipeline (read/write/edit/grep files) to perform coding tasks. |
| **OpenEnv RL Environment** | A standardised environment exposing `reset()`/`step()` semantics — designed for training and evaluating autonomous coding agents on real-world bug-fixing and feature-implementation tasks. |

The environment is built around **real-world failure modes** derived from production libraries (`tenacity`, `cachetools`, `pybreaker`, `asyncio`), making it a grounded benchmark rather than a synthetic toy.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL AGENTS                              │
│           (inference.py / third-party evaluators)                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │  HTTP (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (server.py)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ /reset   │  │ /step    │  │ /state   │  │/dashboard│            │
│  │ /grader  │  │ /tasks   │  │ /schema  │  │/rl/stats │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘            │
│       │              │             │                                 │
│       └──────────────┼─────────────┘                                │
│                      ▼                                              │
│            ┌─────────────────┐       ┌──────────────────┐           │
│            │   OpenEnv       │◄─────►│   TaskBank       │           │
│            │   (env.py)      │       │   (tasks/)       │           │
│            └───────┬─────────┘       └──────────────────┘           │
│                    │                                                │
│         ┌──────────┼──────────┐                                     │
│         ▼          ▼          ▼                                     │
│  ┌───────────┐ ┌────────┐ ┌──────────────┐                         │
│  │ LinUCB    │ │Provider│ │PipelineRunner│                          │
│  │ Bandit    │ │(LLM)   │ │(orion/       │                         │
│  │(orion/rl/)│ │        │ │  pipeline/)  │                          │
│  └───────────┘ └────────┘ └──────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │  NVIDIA NIM API │
              │  (LiteLLM)      │
              └─────────────────┘
```

---

## 3. Repository Layout

```
orion-CLI/
├── env.py                  # Core OpenEnv environment (reset/step/state)
├── server.py               # FastAPI server wrapping env.py
├── inference.py            # Baseline agent — multi-task evaluation loop
├── openenv.yaml            # OpenEnv manifest declaration
├── Dockerfile              # Production container image
├── pyproject.toml          # Package metadata & entry-points
├── requirements.txt        # Pinned runtime dependencies
│
├── orion/                  # Main library package
│   ├── __init__.py         # Package version (0.1.0)
│   ├── utils.py            # Shared utilities
│   │
│   ├── cli/                # Textual TUI application
│   │   ├── app.py          # OrionApp entry-point (summon-orion)
│   │   ├── screens/        # TUI screens (splash, setup, main, help, history)
│   │   └── widgets/        # TUI widgets (chat_panel, file_tree, info_panel, input_bar)
│   │
│   ├── config/             # Configuration management
│   │   └── config.py       # TOML-backed Config dataclass (~/.orion/config.toml)
│   │
│   ├── session/            # Session & message persistence
│   │   └── session.py      # SQLite-backed SessionManager (~/.orion/sessions.db)
│   │
│   ├── provider/           # LLM abstraction layer
│   │   └── provider.py     # Multi-tier Provider (fast/coder/balanced/heavy via LiteLLM)
│   │
│   ├── pipeline/           # Multi-stage agentic pipeline
│   │   ├── models.py       # Dataclasses: IntentResult, ToolCall, AgentResult, PipelineContext
│   │   ├── context.py      # System prompt builder + message assembly
│   │   ├── c01_intent.py   # Stage C01 — Intent classification
│   │   ├── agentic_loop.py # Stage C02–C08 — Tool-calling agentic loop (max 3 iterations)
│   │   ├── c09_validation.py # Stage C09 — IISG safety/quality validation
│   │   └── runner.py       # PipelineRunner — orchestrates intent → agentic loop → validation
│   │
│   ├── rl/                 # Reinforcement learning subsystem
│   │   ├── bandit.py       # LinUCB Contextual Bandit (8 actions, 4-dim state)
│   │   └── state_encoder.py # State vector encoder (intent, complexity, language, past IISG)
│   │
│   └── tool/               # File-system tools for the agentic pipeline
│       └── tools.py        # ReadTool, WriteTool, EditTool, GrepTool
│
├── tasks/                  # Task definitions & graders
│   ├── __init__.py
│   └── task_bank.py        # 4 production-grounded tasks + sandboxed graders + TaskBank
│
├── app/                    # Pydantic v2 API models
│   ├── __init__.py
│   └── models.py           # Observation, Action (discriminated union), Reward, StepResponse
│
├── server/                 # Alternate server entry-point
│   └── app.py              # Re-exports server:app with uvicorn runner
│
├── scripts/                # Build-time scripts
│   └── preseed_bandit.py   # Pre-trains the bandit with 100 simulated episodes
│
└── tests/                  # Test suite
    └── test_graders.py     # Golden trajectory tests (16 test cases across 4 graders)
```

---

## 4. Core Subsystems

### 4.1 OpenEnv Environment (`env.py`)

The central class `OpenEnv` implements the standard **reset → step → done** RL loop.

#### State Management

```python
@dataclass
class OpenEnvState:
    task_name: str
    task_difficulty: str
    task_prompt: str
    workspace: str          # Temp directory for this episode
    history: list           # Step-by-step action log
    total_reward: float     # Cumulative reward (starts at 0.01)
    steps: int              # Current step count
    best_score: float       # Highest grader score seen
    last_submission: str
```

#### `reset(task_name?, difficulty?) → dict`

1. Selects a task from `TaskBank` (by name or by random sampling).
2. Creates a temporary workspace directory.
3. Writes the task's `setup_files` into the workspace (buggy source files).
4. Returns the initial observation.

#### `step(action: dict | str) → StepResponse`

Accepts a **discriminated-union tool action** with `action_type` routing:

| `action_type` | Behaviour | Grader Called? |
|---|---|---|
| `read_file` | Returns file contents (capped at 3000 chars) | No |
| `write_file` | Writes content to workspace, then runs grader | **Yes** |
| `run_tests` | Runs grader on current workspace state | **Yes** |
| `list_files` | Lists workspace directory | No |
| `submit` | Final submission — runs grader, marks episode done | **Yes** |

**Backward compatibility:** If `action` is a plain string, it's auto-wrapped as a `write_file` to `solution.py`.

#### Reward Logic

- **Improvement-based:** `reward = grader_score - best_score` (floor at 0.01)
- **Efficiency bonus:** +0.1 if score ≥ 0.95 within half the step budget
- **Observation penalty:** Reading/listing/testing deducts 0.02 from cumulative `total_reward`
- **All values clamped:** `[0.01, 0.99]` — prevents validator edge-case failures

#### Termination Conditions

- `steps >= step_budget` (per-task, ranges 15–30)
- `grader_score >= 0.95` (early success)
- Agent explicitly calls `submit`

#### Singleton Pattern

A module-level `get_env(api_key)` function provides a cached singleton for module-level `reset()`/`step()`/`state()` convenience functions.

---

### 4.2 FastAPI Server (`server.py`)

Wraps `OpenEnv` as a stateless HTTP service compliant with the OpenEnv specification.

#### Session Management

- Uses an `OrderedDict` as an **LRU session cache** (max 512 sessions by default).
- Each `/reset` creates a new `OpenEnv` instance keyed by `session_id`.
- Oldest sessions are evicted when the pool is full.

#### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute one tool action |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | List all available tasks |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | JSON schemas (Action, Observation, Reward) |
| `POST` | `/grader` | Re-run a grader on an existing workspace |
| `POST` | `/baseline` | Baseline info stub |
| `GET` | `/rl/stats` | LinUCB bandit runtime statistics |
| `GET` | `/dashboard` | Live HTML monitoring dashboard |
| `GET` | `/openenv.yaml` | Serve the OpenEnv manifest |

#### Dashboard

An embedded single-page HTML dashboard (served from `/dashboard`) that auto-refreshes every 3 seconds. It displays:
- Current episode state (task, steps, best score, reward sparkline)
- LinUCB bandit statistics (action distribution, best action per task)
- Live step log

---

### 4.3 Inference Script (`inference.py`)

The **baseline agent** — a self-contained evaluation loop demonstrating how an LLM agent interacts with the environment.

#### Architecture

```
                ┌──────────────┐
                │  OpenAI SDK  │───► NVIDIA NIM API
                └──────┬───────┘
                       │ LLM responses (JSON tool actions)
                       ▼
┌──────────────────────────────────────────┐
│              inference.py                 │
│                                           │
│  for each task in task_bank:              │
│    1. log_start(task)                     │
│    2. env.reset(task_name)                │
│    3. for step in range(MAX_STEPS=8):     │
│       a. call_llm(conversation) → JSON    │
│       b. env.step(action)                 │
│       c. append result to conversation    │
│       d. update bandit weights            │
│       e. log_step(...)                    │
│    4. log_end(steps, rewards)             │
└──────────────────────────────────────────┘
```

#### Key Design Decisions

- **Conversational memory:** Maintains a full `conversation` list so the LLM remembers previous tool results.
- **JSON-native tool calling:** System prompt instructs the LLM to respond with only valid JSON action objects.
- **Fallback resilience:** If JSON parsing fails, the response is wrapped as a `write_file` to `solution.py`. If a 404 error occurs during `step()`, it retries with the fallback action.
- **Bandit integration:** After each step, the bandit is updated with the real episode reward.
- **Structured logging:** Emits `[START]`, `[STEP]`, and `[END]` lines in a format parseable by the OpenEnv validator.

#### Logging Format

```
[START] task=fix_tenacity_retry env=orion model=nvidia_nim/qwen/qwen2.5-coder-32b-instruct
[STEP] step=1 action=list_files reward=0.01 done=false error=null
[STEP] step=2 action=read_file reward=0.01 done=false error=null
[STEP] step=3 action=write_file reward=0.75 done=false error=null
[END] success=true steps=3 score=0.75 rewards=0.01,0.01,0.75
```

---

### 4.4 Task Bank (`tasks/`)

Contains **4 production-grounded coding tasks** and their deterministic graders.

#### Task Schema

```python
@dataclass
class Task:
    name: str            # Unique identifier
    difficulty: str      # "Medium" or "Hard"
    prompt: str          # Natural language problem description
    setup_files: dict    # {filename: buggy_source_code}
    grader: callable     # workspace → float ∈ [0.01, 0.99]
    seed: int            # Deterministic seed
    step_budget: int     # Max steps before forced termination
```

#### Task Inventory

| # | Task | Difficulty | Target File | Bug Type | Budget |
|---|---|---|---|---|---|
| 1 | `fix_tenacity_retry` | Medium | `retry_utils.py` | Missing exception filter + wrong error handling | 15 |
| 2 | `fix_cachetools_ttl` | Medium | `cache_manager.py` | Missing TTL check on get + broken expire() | 20 |
| 3 | `implement_pybreaker` | Hard | `circuit_breaker.py` | Full state-machine implementation needed | 25 |
| 4 | `fix_async_race` | Hard | `async_worker.py` | asyncio read-modify-write race condition | 30 |

#### Grading Architecture

Each grader follows a consistent pattern:

1. **File existence check** — returns 0.01 if target file is missing.
2. **Syntax validation** — `compile()` to catch parse errors.
3. **Sandboxed execution** — `_safe_exec()` runs student code in a restricted namespace:
   - Stripped builtins (no `eval`, `exec`, `open`, `__import__`)
   - Whitelisted stdlib imports only
   - 5-second timeout via `ThreadPoolExecutor`
4. **Behavioural testing** — exercises the student's code with specific inputs.
5. **Partial credit scoring** — returns intermediate scores (0.25, 0.50, 0.75) based on which bugs are fixed.

**Score scale:** `0.01 → 0.25 → 0.50 → 0.75 → 0.99` (never 0.0 or 1.0, by design).

#### TaskBank Class

```python
class TaskBank:
    sample(difficulty?) → Task      # Random task selection
    get_by_name(name) → Task        # Direct lookup
    grade(workspace) → float        # Run current task's grader
    reset() → dict                  # Clear current task
```

---

### 4.5 Pipeline Engine (`orion/pipeline/`)

The pipeline is the **brain of the TUI coding assistant** — a multi-stage agentic system that processes user prompts into code changes.

#### Stage Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────────────────┐
│  C01 — Intent Classification        │   LLM call (fast tier)
│  → IntentResult(type, complexity)   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Bandit Action Selection            │   LinUCB selects pipeline config
│  → {planner_tier, coder_tier,       │   (which model tiers to use)
│     reviewer_tier}                  │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  C02–C08 — Agentic Tool Loop       │   Up to 3 iterations
│  ┌─────────────────────────────┐    │
│  │ LLM generates response     │    │   LLM call (coder tier)
│  │ Parse <tool>XML</tool>     │    │
│  │ Execute tool calls         │    │
│  │ Run C09 validation on      │    │
│  │   writes                   │    │
│  │ If pass_rate < 1.0:        │    │
│  │   feed failures back       │    │
│  │   and retry                │    │
│  └─────────────────────────────┘    │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  C09 — Final IISG Validation        │   Regex-based safety checks
│  → ValidationResult(pass_rate,      │
│     syntax_valid, clause_results)   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Reward Calculation                  │
│  = 0.8 × IISG + 0.1 × syntax       │
│    + 0.1 × token_efficiency         │
└─────────────────────────────────────┘
```

#### C01 — Intent Classification (`c01_intent.py`)

- Sends user prompt to a lightweight LLM (`fast` tier).
- Parses a structured two-line response: `INTENT: <type>` and `COMPLEXITY: <level>`.
- Valid intents: `bug_fix`, `feature`, `refactor`, `explain`, `test`.
- Valid complexities: `low`, `medium`, `high`.
- **Fail-safe:** defaults to `feature / medium` on any parse error.

#### Agentic Tool Loop (`agentic_loop.py`)

- Runs up to `MAX_ITERATIONS = 3` coding attempts.
- Parses XML-formatted tool calls from LLM output:
  ```xml
  <tool>WriteTool</tool>
  <path>src/main.py</path>
  <content>
  ... complete file content ...
  </content>
  ```
- **Auto-retry on format failure:** If the LLM outputs 300+ chars but no valid tool blocks, it sends a correction prompt demanding XML format.
- **Read-result feedback:** Read results are injected back into conversation history for subsequent iterations.
- After each iteration, C09 runs on the writes. If `pass_rate == 1.0`, the loop exits early.

#### C09 — IISG Validation (`c09_validation.py`)

Validates all `WriteTool`/`EditTool` outputs against safety clauses:

| Clause | What It Catches |
|---|---|
| `no_hardcoded_secrets` | API keys, passwords, tokens embedded in code |
| `no_dangerous_patterns` | `os.system()`, `eval()`, `exec()`, `subprocess(..., shell=True)` |
| `no_todos_left` | `TODO`, `FIXME`, `HACK`, `XXX` markers |
| `no_absolute_paths` | Hardcoded paths like `/home/`, `C:\\` |
| `syntax_valid` | Python syntax check via `compile()` |

**Pass rate** = (passed clauses) / (total clauses).

#### Pipeline Context (`models.py`)

```python
@dataclass
class PipelineContext:
    prompt: str
    intent: IntentResult
    agent: AgentResult
    validation: ValidationResult
    iisg_pass_rate: float
    syntax_valid: bool
    tokens_used: int
    reward: float               # Composite scalar
    action_name: str            # Bandit action name
    final_response: str         # Cleaned response (tool blocks stripped)
    time_taken: float
    error: str
```

#### PipelineRunner (`runner.py`)

Orchestrates the full pipeline:

1. Resets token counters.
2. Runs C01 intent classification.
3. Optionally selects a bandit action (model tier configuration).
4. Runs the agentic loop with the chosen tiers.
5. Runs C09 final validation.
6. Computes composite reward: `0.8 × IISG + 0.1 × syntax + 0.1 × token_efficiency`.
7. Updates the bandit with the reward signal.
8. Strips tool XML blocks from the final response.

---

### 4.6 Reinforcement Learning (`orion/rl/`)

#### LinUCB Contextual Bandit (`bandit.py`)

An **online contextual bandit** that learns which LLM model-tier configuration works best for each type of coding task.

**Algorithm:** Linear Upper Confidence Bound (LinUCB)

- **State space:** 4-dimensional vector `[intent_type, complexity, language, past_iisg_avg]`
- **Action space:** 8 discrete pipeline configurations:

| Index | Action Name | Planner | Coder | Reviewer |
|---|---|---|---|---|
| 0 | `fast-fast-no-review` | fast | fast | ✗ |
| 1 | `fast-coder-no-review` | fast | coder | ✗ |
| 2 | `fast-coder-balanced-review` | fast | coder | ✓ |
| 3 | `balanced-coder-balanced-review` | balanced | coder | ✓ |
| 4 | `fast-fast-with-review` | fast | fast | ✓ |
| 5 | `balanced-heavy-with-review` | balanced | heavy | ✓ |
| 6 | `balanced-coder-with-review` | balanced | coder | ✓ |
| 7 | `balanced-heavy-no-review` | balanced | heavy | ✗ |

**UCB Score:**
```
score(a) = w_a · x + α × √(ln(N+1) / (n_a+1))
```
Where `w_a` = learned weight vector for action `a`, `x` = state vector, `α` = exploration parameter (default 1.0), `N` = total episodes, `n_a` = times action `a` was selected.

**Update Rule:** Online linear regression with decaying learning rate:
```
error = reward - w_a · x
w_a += (1 / (n_a + 1)) × error × x
```

**Persistence:** Dual storage — JSON (`~/.orion/bandit_weights.json`) and NumPy `.npz` (`/app/bandit_weights.npz`).

#### State Encoder (`state_encoder.py`)

Transforms raw context into a numeric feature vector:

| Feature | Encoding | Range |
|---|---|---|
| Intent type | Ordinal: `{bug_fix:0, feature:1, refactor:2, explain:3, test:4}` | [0, 4] |
| Complexity | Ordinal: `{low:0, medium:1, high:2}` | [0, 2] |
| Language | Detected from prompt/workspace: `{.py:0, .js:1, .ts:2, .go:3, ...}` | [0, 10] |
| Past IISG avg | Rolling average of last 5 IISG pass rates | [0, 1] |

IISG history is persisted to `~/.orion/iisg_history.json`.

#### Preseed Script (`scripts/preseed_bandit.py`)

Runs **100 simulated episodes** at Docker build time to give the bandit non-trivial prior weights, so it doesn't start from pure exploration in production.

---

### 4.7 LLM Provider (`orion/provider/`)

#### Multi-Tier Model Routing

```python
MODELS = {
    "fast":     "nvidia_nim/meta/llama-3.1-8b-instruct",
    "coder":    "nvidia_nim/qwen/qwen2.5-coder-32b-instruct",
    "balanced": "nvidia_nim/meta/llama-3.3-70b-instruct",
    "heavy":    "nvidia_nim/deepseek/deepseek-v3",
}
```

All calls go through [LiteLLM](https://github.com/BerriAI/litellm) for unified API access to NVIDIA NIM endpoints.

#### Features

- **Streaming support:** First agentic iteration is streamed for responsiveness; subsequent iterations use batch completion.
- **Token tracking:** Accumulates `prompt_tokens` and `completion_tokens` across a pipeline run.
- **Rate-limit retry:** Auto-retries once on 429/529 responses after a 2-second backoff.

---

### 4.8 Tool System (`orion/tool/`)

Four file-system tools for the agentic pipeline:

| Tool | Operation | Notes |
|---|---|---|
| `ReadTool` | Read file contents | Truncates at 50,000 chars |
| `WriteTool` | Create/overwrite file | Creates parent directories |
| `EditTool` | Overwrite existing file | Identical implementation to WriteTool |
| `GrepTool` | Regex search across files | Walks directory tree, capped at 100 matches |

All tools operate on the **workspace** (either the user's CWD in TUI mode, or the temp directory in OpenEnv mode).

**GrepTool** ignores: `.git`, `node_modules`, `__pycache__`, `.venv`, `.env`, `dist`, `build`, `orion_cli.egg-info`.

---

### 4.9 Session & Config

#### Configuration (`orion/config/config.py`)

- TOML-backed dataclass stored at `~/.orion/config.toml`.
- Fields: `nim_api_key`, `default_model`, `theme`, `cwd`.
- Environment variable `NVIDIA_NIM_API_KEY` overrides the stored key at runtime (never persisted).

#### Session Manager (`orion/session/session.py`)

- **SQLite-backed** persistence at `~/.orion/sessions.db`.
- Two tables: `sessions` (id, name, created_at, cwd) and `messages` (id, session_id, role, content, created_at).
- Sessions get random two-word names (e.g., "brave-falcon", "quiet-ember").
- WAL journal mode for concurrent access.

---

### 4.10 Terminal UI (`orion/cli/`)

Built with [Textual](https://textual.textualize.io/) for a rich terminal experience.

#### Entry Point

```bash
summon-orion    # pyproject.toml console_script
```

#### Screen Flow

```
OrionApp.on_mount()
    │
    ├── Config not set? → SetupScreen (API key entry)
    │
    └── Config OK? → SplashScreen → MainScreen
                                        │
                                        ├── ChatPanel (conversation view)
                                        ├── InfoPanel (task/session info)
                                        ├── FileTree (workspace files)
                                        ├── InputBar (prompt input)
                                        └── HelpScreen (key bindings)
```

---

### 4.11 Data Models (`app/models.py`)

Pydantic v2 models defining the OpenEnv API contract:

#### Observation

```python
class Observation(BaseModel):
    task_name: str
    task_difficulty: str
    task_prompt: str
    workspace: str
    history: list[dict]
    total_reward: float          # ∈ [0.01, ∞)
    steps: int
    best_score: float            # ∈ [0.01, ∞)
    available_tools: list[str]   # ["read_file", "write_file", "run_tests", "list_files", "submit"]
    last_tool_result: str | None
    files_in_workspace: list[str]
```

#### Action (Discriminated Union)

```python
Action = Union[ReadFile, WriteFile, RunTests, ListFiles, Submit]
# Discriminated by action_type field
```

#### Reward

```python
class Reward(BaseModel):
    correctness: float   # ∈ [0.01, 0.99] — grader score
    efficiency: float    # ∈ [0.01, 0.99] — 1 - (steps / budget)
    final_score: float   # ∈ [0.01, 0.99] — composite reward for this step
```

---

## 5. Data Flow

### 5.1 Episode Lifecycle

```
Agent                     Server                     OpenEnv                    TaskBank
  │                         │                          │                          │
  │──POST /reset───────────►│                          │                          │
  │                         │──env.reset(task)────────►│                          │
  │                         │                          │──get_by_name(task)──────►│
  │                         │                          │◄──Task(setup_files)──────│
  │                         │                          │──mkdir(workspace)         │
  │                         │                          │──write(setup_files)       │
  │◄─observation────────────│◄─dict──────────────────-─│                          │
  │                         │                          │                          │
  │──POST /step {action}───►│                          │                          │
  │                         │──env.step(action)───────►│                          │
  │                         │                          │──route(action_type)       │
  │                         │                          │──[write_file?]────────────│
  │                         │                          │   grade(workspace)───────►│
  │                         │                          │◄──score──────────────────│
  │                         │                          │──compute reward           │
  │                         │                          │──check done               │
  │◄─StepResponse──────────│◄─StepResponse────────────│                          │
  │                         │                          │                          │
  │  ... repeat until done  │                          │                          │
```

### 5.2 Reward Computation

```
                    ┌──────────────────┐
                    │  Grader Score     │  float ∈ [0.01, 0.99]
                    │  (correctness)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Improvement     │  base = score - best_score
                    │  Calculation     │  if base ≤ 0 → 0.01
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Efficiency      │  +0.1 if score ≥ 0.95 AND
                    │  Bonus           │  steps ≤ budget/2
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Observation     │  -0.02 from total_reward for
                    │  Penalty         │  read_file, list_files, run_tests
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Clamping        │  final ∈ [0.01, 0.99]
                    └──────────────────┘
```

### 5.3 Bandit Learning Loop

```
                      ┌──────────────┐
          encode()    │ StateEncoder │
     ────────────────►│ (4-dim vec)  │
                      └──────┬───────┘
                             │ state_vector
                      ┌──────▼───────┐
          select()    │  LinUCB      │
     ────────────────►│  Bandit      │──► action_idx (0–7)
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
     action_to_       │  Pipeline    │
     pipeline()       │  Config      │──► {planner, coder, reviewer}
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
                      │  Execute     │
                      │  Pipeline    │──► reward
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
          update()    │  LinUCB      │
     ────────────────►│  Bandit      │──► updated weights
                      └──────────────┘
```

---

## 6. Deployment Architecture

### Docker Container

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY orion/ tasks/ app/ env.py inference.py server.py server/ openenv.yaml scripts/ .

# Pre-train bandit at build time
RUN python -c "from scripts.preseed_bandit import preseed_bandit; preseed_bandit()"

# Security: non-root user
USER orion

# Healthcheck
HEALTHCHECK --interval=30s CMD curl -fsS http://127.0.0.1:7860/health

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `NVIDIA_NIM_API_KEY` | API key for NVIDIA NIM LLM access | (required) |
| `HF_TOKEN` | HuggingFace token (used by inference.py) | (required for inference) |
| `API_BASE_URL` | LLM API endpoint | `https://integrate.api.nvidia.com/v1` |
| `MODEL_NAME` | Model for inference script | `nvidia_nim/qwen/qwen2.5-coder-32b-instruct` |
| `MAX_SESSIONS` | Server session pool size | `512` |
| `BANDIT_WEIGHTS_PATH` | Path to bandit weights file | `/app/bandit_weights.npz` |

---

## 7. OpenEnv Compliance Contract

OrionCLI conforms to the [OpenEnv specification](https://openenv.dev) as declared in `openenv.yaml`:

| Requirement | Implementation |
|---|---|
| **Manifest** | `openenv.yaml` at repo root with tasks, obs/action spaces, endpoints |
| **REST API** | FastAPI server on port 7860 with `/reset`, `/step`, `/state` |
| **Typed schemas** | Pydantic v2 models: `Observation`, `Action`, `Reward` |
| **Reward bounds** | All rewards strictly clamped to `[0.01, 0.99]` — never 0 or 1 |
| **Deterministic tasks** | Each task has a fixed `seed` and `step_budget` |
| **Grader isolation** | Sandboxed execution with restricted builtins and 5s timeout |
| **Baseline agent** | `inference.py` runs all 4 tasks with structured logging |
| **Docker support** | Production `Dockerfile` with healthcheck and non-root user |
| **Structured logging** | `[START]`/`[STEP]`/`[END]` format for validator parsing |

---

## 8. Dependency Graph

```
┌─────────────────────────────────────────────────────┐
│                  External Services                   │
│                                                     │
│         NVIDIA NIM API (LLM inference)              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               Python Dependencies                    │
│                                                     │
│  litellm ──────── LLM abstraction layer             │
│  openai ───────── SDK for inference.py              │
│  fastapi ──────── HTTP server framework             │
│  uvicorn ──────── ASGI server                       │
│  pydantic ─────── Data validation (v2)              │
│  numpy ────────── LinUCB bandit math                │
│  textual ──────── Terminal UI framework             │
│  tomli-w ──────── TOML config writing               │
│  httpx ────────── Async HTTP client                 │
│  slowapi ──────── Rate limiting                     │
│  pyyaml ───────── YAML parsing                      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Internal Module Graph                   │
│                                                     │
│  server.py ─────► env.py ─────► orion/rl/           │
│      │               │              │                │
│      │               ├─────► orion/pipeline/         │
│      │               │              │                │
│      │               ├─────► orion/provider/         │
│      │               │                               │
│      │               ├─────► orion/tool/             │
│      │               │                               │
│      │               └─────► tasks/task_bank.py      │
│      │                                               │
│      └──────────► app/models.py                     │
│                                                     │
│  orion/cli/app.py ──► orion/config/                 │
│        │              orion/session/                 │
│        └────────────► orion/pipeline/               │
│                       orion/provider/               │
└─────────────────────────────────────────────────────┘
```

---

*This document was auto-generated from a full codebase analysis of OrionCLI v0.1.0.*
