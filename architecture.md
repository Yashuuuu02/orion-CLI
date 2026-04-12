# OrionCLI — System Architecture

> **Version:** 1.0.0  
> **Last Updated:** 2026-04-12  
> **Python:** ≥ 3.11  

---

## Table of Contents

1. [Project Identity & Overview](#1-project-identity--overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Phase 1 — TUI Shell](#4-phase-1--tui-shell)
5. [Phase 2 — AI Pipeline](#5-phase-2--ai-pipeline)
6. [Phase 3 — RL Layer + OpenEnv](#6-phase-3--rl-layer--openenv)
7. [Data Flow](#7-data-flow)
8. [Configuration & Deployment](#8-configuration--deployment)
9. [OpenEnv Compliance Contract](#9-openenv-compliance-contract)
10. [Build Order & Test Protocol](#10-build-order--test-protocol)
11. [Known Limitations & Roadmap](#11-known-limitations--roadmap)
12. [Dependency Graph](#12-dependency-graph)

---

## 1. Project Identity & Overview

**Orion CLI is a terminal-native, self-validating, RL-optimised coding agent.** Users describe coding tasks in plain English. Orion classifies the intent, plans, writes real files to disk, validates output with deterministic checks, and learns which model configuration to use over time via a contextual bandit.

The CLI ships with an **embedded OpenEnv RL environment** as an integrated subsystem. This environment exposes the same core pipeline, tool system, and LLM provider that power the CLI, but packages them behind a standardised `reset()`/`step()` API for training and evaluating autonomous coding agents.

### 1.1 What Orion Does

| Capability | Description |
|---|---|
| **Intent Classification** | Classifies every prompt into one of 5 intents (`bug_fix`, `feature`, `refactor`, `explain`, `test`) and 3 complexity levels using a lightweight fast model. |
| **Agentic Loop** | Iterative coder (max 3 attempts) with tool execution. Reads, writes, and edits files on disk. Auto-retries when tool format fails. |
| **Deterministic Validation** | C09 runs 5 pure-Python IISG clauses after every write. No LLM involved. Score is always reproducible. |
| **Live Streaming** | Tokens stream live into the TUI as the coder model generates them. |
| **Session Persistence** | Every conversation stored in SQLite. Full history across restarts. |
| **RL Routing (Phase 3)** | LinUCB bandit learns optimal model tier per task type. Gets cheaper and better over time. |
| **OpenEnv Interface (Phase 3)** | Exposes `reset()` / `step()` / `state()` for hackathon judge evaluation. |

### 1.2 How It Compares

| Feature | Orion CLI | GitHub Copilot CLI | OpenCode |
|---|---|---|---|
| **Runs in terminal** | Yes | Yes | Yes |
| **Writes files to disk** | Yes | No | Yes |
| **Deterministic validation** | Yes (C09) | No | No |
| **RL model routing** | Yes (Phase 3) | No | No |
| **Session persistence** | Yes (SQLite) | No | Yes |
| **Live token streaming** | Yes | No | Yes |
| **Cost tracking** | Yes | No | Partial |
| **Open source** | Yes | No | Yes |

---

## 2. High-Level Architecture

Orion is structured as three clean layers: the **TUI shell** (Phase 1), the **AI pipeline** (Phase 2), and the **RL + OpenEnv layer** (Phase 3). Each layer is independently testable and replaceable.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL AGENTS                              │
│           (inference.py / third-party evaluators)                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │  HTTP (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (server.py)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ /reset   │  │ /step    │  │ /state   │  │/dashboard│             │
│  │ /grader  │  │ /tasks   │  │ /schema  │  │/rl/stats │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘             │
│       │              │             │                                │
│       └──────────────┼─────────────┘                                │
│                      ▼                                              │
│            ┌─────────────────┐       ┌──────────────────┐           │
│            │   OpenEnv       │◄─────►│   TaskBank       │           │
│            │   (env.py)      │       │   (tasks/)       │           │
│            └───────┬─────────┘       └──────────────────┘           │
│                    │                                                │
│         ┌──────────┼──────────┐                                     │
│         ▼          ▼          ▼                                     │
│  ┌───────────┐ ┌────────┐ ┌──────────────┐                          │
│  │ LinUCB    │ │Provider│ │PipelineRunner│   ◄─── Textual TUI       │
│  │ Bandit    │ │(LLM)   │ │(orion/       │      (summon-orion)      │
│  │(Phase 3)  │ │(Phase 2│ │ pipeline/)   │        (Phase 1)         │
│  └───────────┘ └────────┘ └──────────────┘                          │
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

```text
orion-CLI/
├── env.py                  # Phase 3: Core OpenEnv environment (reset/step/state)
├── server.py               # Phase 3: FastAPI server wrapping env.py
├── inference.py            # Phase 3: Hackathon mandatory file — baseline agent
├── openenv.yaml            # Phase 3: Environment spec & manifest
├── Dockerfile              # Phase 3: Container deployment
├── pyproject.toml          # Package metadata & entry-points
├── requirements.txt        # Pinned runtime dependencies
│
├── orion/                  # Main library package
│   ├── __init__.py
│   ├── utils.py            # Shared utilities
│   │
│   ├── cli/                # Phase 1: Terminal UI (Textual)
│   │   ├── app.py          # OrionApp (Textual application entry point `summon-orion`)
│   │   ├── screens/        # splash.py, main.py
│   │   └── widgets/        # chat_panel.py, input_bar.py, info_panel.py, file_tree.py
│   │
│   ├── config/             # Phase 1: Config & Env var resolution
│   │   └── config.py       # TOML config loader
│   │
│   ├── session/            # Phase 1: Session & message persistence
│   │   └── manager.py      # SQLite-backed SessionManager
│   │
│   ├── provider/           # Phase 2: LLM abstraction layer
│   │   └── provider.py     # LiteLLM → NVIDIA NIM wrapper
│   │
│   ├── pipeline/           # Phase 2: Agentic Pipeline Engine
│   │   ├── models.py       # Pure dataclasses (no I/O)
│   │   ├── c01_intent.py   # Intent classification (1 LLM call)
│   │   ├── c09_validation.py # IISG checker (pure Python, 0 LLM calls)
│   │   ├── context.py      # Context window builder (sliding window)
│   │   ├── agentic_loop.py # Core agentic loop (up to 3 iterations)
│   │   └── runner.py       # Chains all pipeline stages
│   │
│   ├── rl/                 # Phase 3: Reinforcement learning subsystem
│   │   ├── bandit.py       # LinUCB Contextual Bandit
│   │   └── state_encoder.py # State vector encoder
│   │
│   └── tool/               # Phase 2: File-system tools
│       └── tools.py        # ReadTool, WriteTool, EditTool, GrepTool
│
├── tasks/                  # Phase 3: Evaluation Tasks
│   └── task_bank.py        # Real-world tasks + sandboxed graders
│
├── app/                    # Phase 3: Pydantic v2 API models
│   └── models.py           # Observation, Action, Reward, StepResponse
│
├── server/
│   └── app.py              # Alternate server entry
│
├── scripts/
│   └── preseed_bandit.py   # Pre-trains bandit with 100 simulated episodes
│
└── tests/
    └── test_graders.py     # Golden trajectory tests
```

---

## 4. Phase 1 — TUI Shell

Phase 1 built the complete terminal user interface without any LLM integration. The TUI is fully functional as a standalone shell — sessions, config, navigation.

### 4.1 What Was Built
- **SplashScreen** — animated startup with project name and version.
- **MainScreen** — primary chat interface with three-panel layout.
- **ChatPanel** — scrollable message history with code block rendering (`RichLog`) and thinking indicator.
- **InputBar** — multi-line `TextArea`, Enter to submit, Shift+Enter for newline.
- **InfoPanel sidebar** — model name, token counts, cost, IISG rate, RL action display. Uses Textual reactive properties (`update_pipeline(iisg_rate, rl_action)`) that smoothly accept Phase 2 output.
- **Session management** — SQLite-backed (`~/.orion/sessions.db`), Ctrl+N for new session, Ctrl+H for history.
- **Config system** — TOML config file (`~/.orion/config.toml`) + env var overrides (`NVIDIA_NIM_API_KEY`).
- **Entry point** — `summon-orion` command via `pyproject.toml`.

### 4.2 Key Technical Decisions

| Decision | Rationale |
|---|---|
| **Textual 8.2.1** | Async-native TUI framework. `@work` decorator allows async pipeline execution without blocking UI. |
| **SQLite for sessions** | Zero-dependency persistence. Full message history queryable with standard SQL. |
| **TOML config** | Human-readable, supports nested structure. |
| **Three-panel layout** | Chat (center) + sidebar (right) + input (bottom). Matches developer mental model. |
| **Placeholder replies** | Allowed TUI to be tested and demoed before pipeline was built. |

---

## 5. Phase 2 — AI Pipeline

Phase 2 replaced the placeholder reply with a real agentic pipeline, wiring seven new modules into the working TUI via a single `@work` process `_run_pipeline()`.

### 5.1 Provider — NVIDIA NIM via LiteLLM
`orion/provider/provider.py` wraps LiteLLM's `acompletion()` for NVIDIA NIM, with four model tiers:
- **fast**: `nvidia_nim/meta/llama-3.1-8b-instruct`
- **coder**: `nvidia_nim/qwen/qwen2.5-coder-32b-instruct`
- **balanced**: `nvidia_nim/meta/llama-3.3-70b-instruct`
- **heavy**: `nvidia_nim/deepseek/deepseek-v3`

**Details:**
- **Streaming:** `on_token_delta(chunk)` fires per token chunk for live TUI updates.
- **Token Tracking:** Accumulates throughout the pipeline, resetting on each `PipelineRunner.run()`.
- **Rate Limit Retry:** Waits 2 seconds and retries once on 429/529 errors.

### 5.2 Tools
Four pure-Python sync file I/O classes in `orion/tool/tools.py`:
- **ReadTool**: Reads file relative to `cwd`. Caps at 50,000 chars.
- **WriteTool**: Writes file, creating parent directories. Returns `True`/`False`.
- **EditTool**: Full file overwrite. Semantically distinct from `WriteTool`.
- **GrepTool**: Regex search across files. Caps at 100 matches. Ignores `.git`, `node_modules`, etc.

### 5.3 AI Pipeline Architecture

`orion/pipeline/models.py` defines the pure dataclasses (`IntentResult`, `LoopIteration`, `PipelineContext`, etc.) that flow through the pipeline.

#### C01 — Intent Classification
Takes ~1s using the `fast` tier to output `INTENT: <type>` and `COMPLEXITY: <level>`.
- Valid intents: `bug_fix`, `feature`, `refactor`, `explain`, `test`.
- Valid complexities: `low`, `medium`, `high`.
- **Fail-safe:** Defaults to `feature / medium` if parsing fails. Forms the start state vector for the Phase 3 bandit.

#### Context Builder
Implements a sliding window methodology to prevent context bloat:
- **System prompt:** Current dir, platform, intent, 2-level file tree, XML tool instructions.
- **Sliding Window:** Sends only last 10 messages from SQLite.
- **Truncation:** Clips messages > 2000 chars.

#### Agentic Loop
Max 3 iterations per task.
- Iteration involves: Build context ➔ Call model ➔ Parse `<tool>` blocks ➔ Execute tools ➔ Inject `ReadTool` output into context ➔ Run C09.
- **Auto-retry:** If model writes >300 chars without XML tags, loops back immediately with explicit format instructions.

#### C09 — IISG Validation
Pure Python, deterministic validation running on all files written.
- Checks: `no_hardcoded_secrets`, `no_dangerous_patterns` (`eval`, `exec`), `no_todos_left`, `no_absolute_paths`, `syntax_valid` (`compile()`).
- Returns pass rate (0.0 to 1.0). If rate < 1.0, injects failure string as feedback message and triggers a retry.

#### Pipeline Runner
Chains it all together, returning the `PipelineContext` up to the UI. The overall reward directly feeds Phase 3:
`R = 0.8 × iisg_pass_rate + 0.1 × syntax_valid + 0.1 × token_efficiency`

---

## 6. Phase 3 — RL Layer + OpenEnv

Phase 3 wraps the Agentic loop with an online contextual bandit and exposes the entire stack via standard programmatic APIs (OpenEnv).

### 6.1 Reinforcement Learning (`orion/rl/`)
The Phase 2 `DEFAULT_ACTION` is replaced by RL.
- **State Encoder:** Transforms C01 outputs into a 4D vector: `[intent_type_int, complexity_int, language_int, past_iisg_avg]`.
- **Action Space:** 8 discrete pipeline configurations deciding combinations of models (fast vs balanced vs heavy vs coder) and dropping reviewer layers.
- **LinUCB Bandit:** Uses Upper Confidence Bound to balance exploration vs. exploitation entirely locally (NumPy only). Persists weights via `.npz` and `.json`.

### 6.2 Task Bank (`tasks/`)
4 production-grounded python bugs (`tenacity` retry bugs, `cachetools` TTL issues, `pybreaker` state machine, `asyncio` race conditions).
- Deterministic sandboxed python evaluation (`_safe_exec`) prevents external execution.
- Scales partial credits: `0.01 ➔ 0.25 ➔ 0.50 ➔ 0.75 ➔ 0.99`.

### 6.3 OpenEnv Wrapper (`env.py`)
Exposes `reset()`, `step()`, `state()` for agents. Step parses a discriminated union action payload:
`ReadFile`, `WriteFile`, `RunTests`, `ListFiles`, `Submit`.

### 6.4 Server & Inference
- **FastAPI Server:** Maps `env.py` into HTTP endpoints (`/reset`, `/step`, `/grader`, `/dashboard`). Tracks sessions via an LRU cache.
- **Inference Script:** A baseline OpenAI-compatible agent that interacts via JSON payload tasks, handling network 404s, using fallback mechanisms, and generating `[START]`, `[STEP]`, `[END]` logs. 

---

## 7. Data Flow

### 7.1 Single Request Data Flow (Phase 1 & 2 integration)
When a user hits Enter in the TUI:
1. `InputBar.Submitted` event fires; `MainScreen` captures.
2. `_handle_user_message()` displays text, disables input, shows typing indicator.
3. `_run_pipeline(text)` (Textual `@work`) initiates. Retrieves 10 prior messages from SQLite.
4. `PipelineRunner.run()` sets token counters.
5. **C01 Intent** uses fast model to classify problem size.
6. **AgenticLoop Itr 1**: Builds context; streams `on_token_delta` directly to `chat_panel.add_streaming_chunk()`.
7. **Tool Execution**: Regex parse XML `<tool>`. Modifies file system.
8. **C09 Validation**: Validates file outputs.
9. **Iterations 2–3**: Triggered if C09 pass < 1.0. Injects failing reasons back to model context silently.
10. **Reward Computed**: Math combining passes & tokens. Feedback fed to RL LinUCB.
11. **TUI Update**: `finalize_streaming()`, updates side panel RL stats, frees input.
12. **Session Persist**: SQLite receives updated dialog tree.

### 7.2 OpenEnv Evaluator Data Flow
```text
Agent                     Server                     OpenEnv                    TaskBank
  │──POST /reset───────────►│──env.reset(task)────────►│──get_by_name(task)──────►│
  │                         │                          │◄──Task(setup_files)──────│
  │                         │                          │──mkdir(workspace)        │
  │                         │                          │──write(setup_files)      │
  │◄─observation────────────│◄─dict────────────────────│                          │
  │                         │                          │                          │
  │──POST /step {action}───►│──env.step(action)───────►│──[write_file?]───────────│
  │                         │                          │   grade(workspace)──────►│
  │                         │                          │◄──score──────────────────│
  │                         │                          │──compute reward          │
  │◄─StepResponse───────────│◄─StepResponse────────────│                          │
```

---

## 8. Configuration & Deployment

### 8.1 Configuration
Stored locally in `~/.orion/config.toml`:
```toml
[orion]
nim_api_key = ""         # or set NVIDIA_NIM_API_KEY env var
default_model = "qwen2.5-coder-32b"
theme = "dark"
```
**Database:** Queries can be tested via `sqlite3 ~/.orion/sessions.db`.

### 8.2 Deployment (OpenEnv)
The hackathon mandatory runtime is defined by a `Dockerfile` using `python:3.11-slim`:
- Installs dependencies & runs script `scripts.preseed_bandit` to initialize weights.
- Deploys as limited-user.
- Exposes port 7860.

Env vars for OpenEnv/Inference usage: `NVIDIA_NIM_API_KEY`, `HF_TOKEN`, `MODEL_NAME`, `MAX_SESSIONS`.

---

## 9. OpenEnv Compliance Contract

The embedded OpenEnv component conforms strictly to the [OpenEnv specification](https://openenv.dev) via `openenv.yaml`:

| Requirement | Implementation |
|---|---|
| **Manifest** | `openenv.yaml` with tasks, obs/action spaces, endpoints. |
| **REST API** | FastAPI server on port 7860 with `/reset`, `/step`, `/state`. |
| **Typed schemas** | Pydantic v2 models: `Observation`, `Action`, `Reward`. |
| **Reward bounds** | All rewards strictly clamped to `[0.01, 0.99]`. |
| **Deterministic tasks** | Each task has a fixed `seed` and `step_budget`. |
| **Grader isolation** | Sandboxed execution with restricted builtins and 5s timeout. |
| **Structured logging** | `[START]`/`[STEP]`/`[END]` formatting directly parseable. |

---

## 10. Build Order & Test Protocol

Each module must follow strict test verification protocols:

### Deployment Build Order
- P3-1 State encoder ➔ P3-2 Bandit ➔ P3-3 Task Bank ➔ P3-4 Env wrapper ➔ P3-5 Inference script ➔ P3-6 openenv.yaml ➔ P3-7 Dockerfile ➔ P3-8 Runner Wiring ➔ P3-9 Launch.

*(Windows Note: Execute tests as scripts, avoid PowerShell `cmd /c set` or trailing quote misinterpretations).*

---

## 11. Known Limitations & Roadmap

Two months of focused work post-hackathon outline the trajectory to surpass tools like OpenCode.

### Current Limitations
- **No LSP integration:** Outputting blind without compiler diagnostics.
- **XML parsing:** ~85% reliability without native JSON function calling.
- **No bash tool:** Cannot run output to detect actual execution traces.
- **Context ceiling:** Large tasks eventually overflow the 10 message history queue.
- **RL Isolation:** Bandit does not yet share user experiences globally.

### Roadmap (Phases 4-8)

| Phase | Description | Key Advantage Gained |
|---|---|---|
| **Phase 4: Tool Gap** (1w) | Move to native tool JSON schema via LiteLLM; Introduction of a `bash` sandbox tool & glob navigation. | `bash` converts Orion from a generator to a live verifier. |
| **Phase 5: Context Intel** (1w) | Incorporate `tree-sitter` AST for non-LLM based whole-project code-maps, plus a Rolling Compaction AI to distill history. | Prevents API bloat, removes raw dependency on full file parsing. |
| **Phase 6: LSP Right** (1w) | Subprocess `pyright/pylsp` to catch post-write diagnostics and inject as errors back to context. | Always deterministic compilation data. |
| **Phase 7: Multi-Agent** (2w) | Introduce separate Explorer, Planner, and Reviewer subsystems. Matrix bandit to `8x8=64` branches. | Model matches problem difficulty dynamically. |
| **Phase 8: Uniqueness** (1m) | Global Bandit aggregation via Supabase; pluggable project-scoped validation; output auto-tuning on high IISG scores. | Agent actively trains better base foundational capabilities over time. |

---

## 12. Dependency Graph

```text
┌─────────────────────────────────────────────────────┐
│                  External Services                  │
│                                                     │
│         NVIDIA NIM API (LLM inference)              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               Python Dependencies                   │
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
│              Internal Module Graph                  │
│                                                     │
│  server.py ─────► env.py ─────► orion/rl/           │
│      │               │              │               │
│      │               ├─────► orion/pipeline/        │
│      │               │              │               │
│      │               ├─────► orion/provider/        │
│      │               │                              │
│      │               ├─────► orion/tool/            │
│      │               │                              │
│      │               └─────► tasks/task_bank.py     │
│      │                                              │
│      └──────────► app/models.py                     │
│                                                     │
│  orion/cli/app.py ──► orion/config/                 │
│        │              orion/session/                │
│        └────────────► orion/pipeline/               │
│                       orion/provider/               │
└─────────────────────────────────────────────────────┘
```

---

*This document was auto-generated from a full codebase analysis of OrionCLI v1.0.0 and hackathon release phases.*
