---
title: OrionCLI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# OrionCLI — RL-Optimised Agentic Coding Environment

An OpenEnv-compliant environment where an RL agent learns to route coding tasks through optimal LLM pipeline configurations.

## What Makes This Different
- LinUCB contextual bandit selects optimal pipeline config per task type
- 8 distinct pipeline actions (planner/coder/reviewer tier combinations)
- IISG (Intent-Instruction-Solution-Grade) validation pipeline
- Real algorithmic tasks that require multi-step reasoning

## Environment Description
OrionCLI is an OpenEnv-compliant reinforcement learning wrapper around a conversational LLM pipeline. In an episode, an external LLM agent interacts with the environment by proposing textual prompt actions. Internally, the environment passes the prompts to a `PipelineRunner`. An embedded LinUCB contextual bandit sits in the middle, dynamically optimizing the specific execution pipeline (choosing between 8 combinations of planner, coder, and reviewer configurations) based on the task's complexity, category, and historical pass rates. The workspace lives in an isolated temporary directory dynamically created per session where files are created, modified, and executed.

## Action Space
The environment expects the external agent to provide a `StepAction` at each step:
- `prompt` (`str`): The instruction or response from the agent. This field uses Pydantic's constraint logic ensuring `min_length=1`.

## Observation Space
Every step or reset operation yields an `Observation` Pydantic model:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Unique identifier for the current task being attempted. |
| `task_difficulty` | `str` | Human-readable string denoting task difficulty (e.g., Medium, Hard). |
| `task_prompt` | `str` | The initial task problem statement and constraints. |
| `workspace` | `str` | Absolute path to the isolated temporary workspace directory. |
| `history` | `list[dict]` | Contextual log of the past actions, pipeline executions, and rewards. |
| `total_reward` | `float` | Cumulative reward score obtained across the current episode. |
| `steps` | `int` | Integer tracking the number of steps taken in the episode. |

## Tasks
The environment features demanding programming tasks with granular scoring to evaluate nuanced capabilities of frontier LLMs.

### debug_off_by_one (Medium, seed=42, budget=10)
- **Description**: Fix binary search off-by-one error that misses the last array element.
- **What makes it hard**: The agent must detect a subtle boundary indexing error (`len(arr) - 2`) inside a well-known algorithmic loop without hallucinating unnecessary refactors.
- **Grader Scoring**: 
  - `0.99`: Proper correction implemented, passes all test cases.
  - `0.7`: Fixes the original edge case but introduces regression elsewhere (3+ tests pass).
  - `0.4`: Function executes and returns an int, but core algorithmic logic is flawed.
  - `0.1`: Function executes but raises an exception.
  - `0.01`: Syntax error or missing target file.

### fix_race_condition (Hard, seed=137, budget=20)
- **Description**: Fix async counter race condition by adding `asyncio.Lock` to read-modify-write pattern.
- **What makes it hard**: Concurrency flaws can be elusive. The asynchronous race condition only triggers during context yields (`await asyncio.sleep(0)`). The agent must infer the lack of thread-safety in an async block and properly provision locking primitives.
- **Grader Scoring**:
  - `0.99`: Correct lock implementation, 100-coroutine concurrent stress test succeeds.
  - `0.7`: Locking primitive implemented but the concurrent state validation fails over time.
  - `0.4`: Core class logic is present but lacking asynchronous synchronization mechanisms.
  - `0.1`: Syntax execution fails due to improper async semantics.
  - `0.01`: Target task files are non-existent.

### implement_lru_cache (Hard, seed=999, budget=20)
- **Description**: Implement O(1) LRU cache with `get(key)` and `put(key, value)` supporting eviction.
- **What makes it hard**: Realizing algorithmic rigor; an optimal solution requires `collections.OrderedDict` or a doubly linked list coupled with a hash map to achieve strict `O(1)` performance guarantees during operations and eviction.
- **Grader Scoring**:
  - `0.99`: Full functional correctness passing 5 behavioral tests.
  - `0.8`: Degraded logic (4/5 cases passing).
  - `0.6`: Boundary mismanagement (3/5 cases passing).
  - `0.4`: Minimal functional correctness (2/5 cases passing).
  - `0.2`: Poor functional correctness (1/5 cases passing).
  - `0.1`: Class loads but core constraints violate expected behavior across all tests.
  - `0.01`: Structural errors or omission of expected files.

## RL Architecture
The internal architecture relies on a **LinUCB (Linear Upper Confidence Bound)** contextual bandit. The bandit optimizes execution runtimes by assigning an incoming request to one of 8 heterogeneous operational pipelines.
- **State Vector Features**: It derives a state representation by analyzing intent (`"bug_fix"`, `"feature"`), static complexity constraints, detected programming language footprints, and a historical running average of the Intent-Instruction-Solution-Grade (IISG) validation rate.
- **Learning**: Post-execution, the model calculates the derived step rewards to recursively adjust covariance properties and pipeline feature weights online, shifting compute allocation progressively towards efficient pipeline strategies over multiple episodes.

## Reward Model
Step evaluations utilize the structured `Reward` model ensuring normalized output bounding `[0.0, 1.0]`:
- **`correctness`**: A float indicating the functional validity of an intermediate action or a granular unit test execution.
- **`efficiency`**: Inverse token-efficiency penalty calculated dynamically based on a token execution budget.
- **`final_score`**: Output evaluation scalar aggregating correctness validation against the targeted intent. 

## API Reference
The application is wrapped natively via FastAPI for continuous environment interaction.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Simple liveness check |
| POST | `/reset` | Initializes a new session and returns the initial observation |
| POST | `/step` | Steps the environment via `StepAction` returning observation and reward |
| GET | `/state` | Returns the raw internal state dictionary of the active session |
| GET | `/tasks` | Lists available tasks and descriptions |
| POST | `/grader` | Re-run a task grader against an arbitrary existing workspace |
| POST | `/baseline` | Fetch static baseline configuration information |
| GET | `/metadata` | Environment properties, schema formats, and supported fields |
| GET | `/schema` | Core entity representation JSON schemas |
| GET | `/openenv.yaml`| Returns the static OpenEnv configuration structure |

## Baseline Scores
The following reflect expected behavior trajectories obtained via local execution. 

| Task Name | Difficulty | Expected Baseline | Notes |
|-----------|------------|-------------------|-------|
| `debug_off_by_one` | Medium | `> 0.80` | Consult `inference.py` baseline output |
| `fix_race_condition` | Hard | `~ 0.50` | Frequently encounters syntax degradation |
| `implement_lru_cache` | Hard | `~ 0.40` | Requires complex O(1) constraints inference |

## Setup & Usage

### Environment Variables
| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | Provider URL path. Default: `https://integrate.api.nvidia.com/v1` |
| `MODEL_NAME` | Active inference target. Default: `nvidia_nim/qwen/qwen2.5-coder-32b-instruct` |
| `HF_TOKEN` | HuggingFace Token, mandatory for local inference endpoints. |
| `NVIDIA_NIM_API_KEY` | Secondary authentication key for internal agent capabilities mapping. |

### Run with Docker
Compile and expose the server process reliably.
```bash
docker build -t orion-cli .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN orion-cli
```

### Run Inference Script
Execute continuous evaluations locally against the bandit configuration.
```bash
python inference.py
```

### API Usage Example with session_id
Session architectures require tracking session state.

```bash
# 1. Initialize environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard"}'

# 2. Advance Episode Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Construct the async lock constraints", "session_id": "c62fb..."}'
```

## OpenEnv Spec Compliance
The implementation completely fulfills standard OpenEnv benchmarks:
- **Pydantic Validation**: All Action/Observation/Reward schema types utilize rigorously bounded Pydantic definitions natively.
- **Session Control**: Concurrent request interactions leverage LRU-evicting `OrderedDict` UUID dictionaries scaling natively up to `MAX_SESSIONS`.
- **Reproducible Evaluation**: Graders apply fixed deterministic testing seeds mapping correctly to individual subtasks.
- **API Parity**: Exposes dedicated configuration endpoints including `/grader`, `/metadata`, `/schema`, `/openenv.yaml` matching standard specifications. 

## Project Structure
```text
orion-CLI/
├── app/
│   ├── __init__.py
│   └── models.py
├── orion/                     # Framework Logic
│   ├── rl/                    # Bandit & State Management
│   ├── pipeline/
│   ├── provider/
│   └── cli/
├── tasks/
│   ├── __init__.py
│   └── task_bank.py           # Multi-tier grading implementations
├── Dockerfile                 # Healthcheck injected runner config
├── env.py                     # Environment wrapper encapsulation
├── inference.py               # Main baseline trajectory executor
├── openenv.yaml               # Metadata mapping configuration
├── pyproject.toml
├── requirements.txt
├── server.py                  # UUID Session based FastAPI core
└── README.md
```