---
title: OrionCLI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# OrionCLI — AI Coding Agent OpenEnv Environment

OrionCLI is an OpenEnv-compliant environment where an AI agent solves real-world Python coding tasks. The agent receives a coding prompt, writes code to a temporary workspace, and is graded by an automated grader that executes the code and checks correctness. It is designed to evaluate and train LLM-based coding agents on progressively harder software engineering tasks.

## Environment Description

The agent operates in an isolated temporary workspace. Each episode presents a coding task — from writing a simple function to refactoring a multi-function pipeline. The agent submits Python code as its action, and the environment executes and grades it, returning a reward in [0.0, 1.0] based on correctness.

## Action Space

Actions are plain strings containing Python code. The agent submits code via the `step()` endpoint.

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | Python code string to be written and executed in the workspace |

## Observation Space

Each `reset()` and `step()` call returns an observation dict with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Name of the current task |
| `task_difficulty` | `str` | Easy / Medium / Hard |
| `task_prompt` | `str` | The task instruction shown to the agent |
| `workspace` | `str` | Path to the temporary working directory |
| `history` | `list` | Last 5 steps taken in the episode |
| `total_reward` | `float` | Cumulative reward so far (0.0 on reset) |
| `steps` | `int` | Number of steps taken (0 on reset) |

## Tasks

### easy_add — Easy
Write an `add(a, b)` function that returns the sum of two numbers.
- **Grader**: Returns `1.0` if `add(2,3)==5`, `0.5` if wrong value returned, `0.25` if function exists but errors, `0.0` if file missing or syntax error.

### medium_fix — Medium
Fix a broken `add(a, b)` function that currently returns `None`.
- **Grader**: Returns `1.0` if `add(2,3)==5`, `0.5` if wrong non-None result, `0.25` if function missing, `0.0` if file missing or syntax error.

### hard_refactor — Hard
Implement three functions — `parse_input`, `process_data`, `format_output` — that chain together to form a data processing pipeline producing a string output.
- **Grader**: Returns `1.0` if all three chain successfully and produce a string, `0.7` / `0.5` / `0.3` for partial success, `0.0` if file missing or syntax error.

## Baseline Scores

| Task | Difficulty | Baseline Score |
|------|------------|----------------|
| easy_add | Easy | 1.00 |
| medium_fix | Medium | 0.50 |
| hard_refactor | Hard | 0.30 |

## Setup & Usage

### Requirements
- Docker
- Python 3.11+
- `openenv-core` (`pip install openenv-core`)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://integrate.api.nvidia.com/v1` |
| `MODEL_NAME` | Model identifier | `nvidia_nim/qwen/qwen2.5-coder-32b-instruct` |
| `HF_TOKEN` | Hugging Face / API key | required |

### Run with Docker

```bash
docker build -t orion-cli .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://integrate.api.nvidia.com/v1 \
  -e MODEL_NAME=nvidia_nim/qwen/qwen2.5-coder-32b-instruct \
  -e HF_TOKEN=your_token \
  orion-cli
```

### Run Inference

```bash
export API_BASE_URL=https://integrate.api.nvidia.com/v1
export MODEL_NAME=nvidia_nim/qwen/qwen2.5-coder-32b-instruct
export HF_TOKEN=your_token
python inference.py
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode, returns initial observation |
| `/step` | POST | Submit an action, returns reward + next observation |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |

## OpenEnv Spec Compliance

- `reset()` returns clean state with all observation fields
- `step()` returns reward in [0.0, 1.0], done flag, and next observation
- `get_state()` returns current state dict
- `openenv.yaml` defines the environment spec
- Dockerfile builds and runs on 2vCPU / 8GB RAM within 20 minutes