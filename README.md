---
title: Priority Hire Env
emoji: "💼"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Priority Hire Env

Priority Hire Env is a real-world-style OpenEnv benchmark for interview scheduling. An agent interacts with the environment by scheduling, deferring, and submitting candidate plans while balancing candidate priority, urgency, specialization match, fit score, and deadline pressure.

It is designed for agent evaluation, local experimentation, and deployment as a Hugging Face Docker Space.

## What This Project Contains

This repository includes:

- a FastAPI server that exposes the environment over HTTP
- the core scheduling environment and task definitions
- typed action, observation, and state models
- a Python client for interacting with the environment
- an inference script for running an LLM baseline against the benchmark
- Docker and OpenEnv metadata for deployment

## Core Idea

Each task presents:

- a queue of pending candidates
- a pool of interviewers
- a limited set of available interview slots
- a scenario-specific hiring objective

The agent must choose one of three actions on each step:

- `schedule` to assign a candidate to an interviewer slot
- `defer` to postpone a candidate
- `submit` to finish the plan and receive the final score

The environment rewards good scheduling decisions and penalizes poor coverage, weak specialization alignment, and leaving urgent candidates pending.

## Folder Structure

```text
priority-hire-env/
|-- server/
|   |-- app.py
|   |-- environment.py
|   `-- __init__.py
|-- client.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- requirements.txt
|-- Dockerfile
|-- uv.lock
|-- README.md
`-- venv/
```

## File Guide

### `server/`

#### `server/app.py`

Creates the FastAPI application using OpenEnv's `create_fastapi_app` helper and exposes:

- `/`
- `/tasks`
- `/grader`
- `/baseline`
- standard OpenEnv endpoints such as `/reset`, `/step`, and `/state`

#### `server/environment.py`

This is the main environment implementation. It contains:

- all benchmark tasks in the `TASKS` dictionary
- the assignment scoring logic
- task-specific graders
- environment state transitions for `reset()` and `step()`
- support for concurrent sessions

This file is the heart of the benchmark.

#### `server/__init__.py`

Marks `server` as a Python package.

### Top-level Python Files

#### `models.py`

Defines the typed contracts used by the environment:

- `PriorityHireAction`
- `PriorityHireObservation`
- `PriorityHireState`

It also provides helper constructors such as:

- `PriorityHireAction.schedule(...)`
- `PriorityHireAction.defer(...)`
- `PriorityHireAction.submit(...)`

#### `client.py`

Provides `PriorityHireEnv`, a typed Python client built on `openenv.core.env_client.EnvClient`. It converts server payloads into typed observation and state objects.

#### `inference.py`

Runs a baseline agent against the environment using an OpenAI-compatible client. It:

- builds prompts from the current observation
- sends them to the configured LLM
- parses JSON action output
- executes actions in the environment
- logs per-step and final scores

It supports environment variables for model and API configuration.

### Config and Packaging Files

#### `openenv.yaml`

Declares the benchmark metadata, task list, action space, observation space, reward range, endpoints, and runtime expectations.

#### `requirements.txt`

Lists runtime dependencies for local development and Docker builds.

#### `pyproject.toml`

Defines packaging metadata and the Python version requirement (`>=3.11`).

#### `Dockerfile`

Builds a Docker image for the environment and serves the app on port `7860`, which is the standard port for Hugging Face Spaces Docker apps.

#### `uv.lock`

Lockfile for reproducible dependency resolution when using `uv`.

#### `venv/`

Local virtual environment directory. This is part of your local workspace setup, not core benchmark logic.

## Tasks

The benchmark currently contains 5 tasks:

| Task ID | Difficulty | Description | Max Attempts |
| --- | --- | --- | --- |
| `easy_critical_backend` | easy | Prioritize urgent backend candidates while keeping interviewer fit high. | 10 |
| `medium_scarce_ml_specialist` | medium | Allocate scarce ML specialist slots to maximize match quality under pressure. | 12 |
| `hard_multi_tradeoff` | hard | Balance competing constraints across leadership, platform, and product priorities. | 12 |
| `medium_deadline_pressure` | medium | Handle near-term deadlines; late scheduling heavily impacts score. | 10 |
| `hard_conflicting_priorities` | hard | Resolve conflicts where top-priority candidates compete for overlapping specialist slots. | 14 |

## Task Themes

Each task stresses a different planning behavior:

- `easy_critical_backend`: prioritize urgent backend talent early
- `medium_scarce_ml_specialist`: avoid wasting specialist ML capacity
- `hard_multi_tradeoff`: balance fit, specialization, and business tradeoffs
- `medium_deadline_pressure`: place urgent candidates into the earliest viable slots
- `hard_conflicting_priorities`: resolve competition among top-priority candidates

## Observation Space

Each step returns an observation with these important fields:

- `pending_candidates_queue`: remaining candidates to handle
- `interviewer_pool`: interviewers and currently available slots
- `global_context`: company and task-specific objective context
- `task_description`: natural-language description of the active task
- `task_id`: current task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `attempt_number`: current step count
- `max_attempts`: max actions allowed before auto-submit
- `feedback`: grader/environment feedback from the previous action

## Action Space

The environment supports the following action schema:

```json
{
  "action_type": "schedule | defer | submit",
  "candidate_id": "required for schedule/defer",
  "interviewer_id": "required for schedule",
  "slot_id": "required for schedule",
  "explanation": "optional"
}
```

### Action Semantics

- `schedule`: assigns a candidate to an interviewer and slot
- `defer`: removes a candidate from the pending queue without scheduling them
- `submit`: finishes the episode and triggers the final grader

## Scoring

Scores are continuous and clamped to a strict open interval between `0.1001` and `0.9899`.

The base score combines:

- candidate priority
- urgency
- interviewer fit score
- specialization match
- deadline pressure
- overall scheduling coverage

The environment also applies:

- penalties for deferring urgent candidates
- penalties for leaving important candidates pending
- task-specific modifiers based on the scenario

### Task-Specific Grading Adjustments

- `easy_critical_backend`: rewards scheduling `c_backend_hotfix` early
- `medium_scarce_ml_specialist`: penalizes wasting ML specialist slots on non-ML candidates
- `hard_multi_tradeoff`: rewards broad specialization alignment
- `medium_deadline_pressure`: rewards urgency-to-slot alignment
- `hard_conflicting_priorities`: strongly rewards placing top-priority candidates

### Pass Threshold

A final score of `0.85` or higher is treated as a pass/completed run.

## API Endpoints

The app exposes the following endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Basic service metadata |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger/OpenAPI UI |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit the next action |
| `/state` | GET | Inspect current episode state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Evaluate a supplied action plan |
| `/baseline` | GET | Run the built-in oracle baseline |

## Local Setup

### Requirements

- Python 3.11
- `pip`

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you are using a virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Server

### Option 1: Run the app module directly

```bash
python -m server.app
```

### Option 2: Run with Uvicorn

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4
```

Once the server is running locally, the app is typically available at:

```text
http://127.0.0.1:7860
```

Useful URLs:

- `/docs`
- `/tasks`
- `/baseline`
- `/health`

## Using the Grader

You can grade a proposed plan by sending a task ID and action list to `/grader`.

Example action list:

```json
[
  {
    "action_type": "schedule",
    "candidate_id": "c_backend_hotfix",
    "interviewer_id": "i_backend_1",
    "slot_id": "b1_morning"
  },
  {
    "action_type": "submit"
  }
]
```

If a plan does not end with `submit`, the grader auto-submits it.

## Running the Baseline Inference Script

`inference.py` uses an OpenAI-compatible API client and can be pointed at Hugging Face Router or another compatible backend.

### Supported Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `LOCAL_IMAGE_NAME` | Local image label for logging | unset |
| `IMAGE_NAME` | Fallback image label if `LOCAL_IMAGE_NAME` is unset | unset |
| `API_KEY` | Required proxy API token injected by the evaluator | unset |
| `API_BASE_URL` | Required proxy base URL injected by the evaluator | unset |
| `MODEL_NAME` | Model used by the baseline agent | `Qwen/Qwen2.5-72B-Instruct` |
| `PRIORITY_HIRE_TASK` | Task to run | `easy_critical_backend` |
| `PRIORITY_HIRE_BENCHMARK` | Benchmark name used in logs | `priority_hire` |
| `SPACE_URL` | Environment server URL | `https://priorityhire-env.hf.space` |

### Example

```powershell
$env:API_BASE_URL="https://your-proxy.example/v1"
$env:API_KEY="your_proxy_key"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:PRIORITY_HIRE_TASK="medium_scarce_ml_specialist"
python inference.py
```

## Deployment Notes

This repository is set up for Docker deployment.

The Docker image:

- uses `python:3.11-slim`
- installs dependencies from `requirements.txt`
- serves the API with Uvicorn
- exposes port `7860`
- includes a health check against `/health`

This makes it suitable for Hugging Face Docker Spaces.

## OpenEnv Metadata

`openenv.yaml` includes:

- benchmark name and version
- tags and description
- the task catalog
- observation space
- action space
- reward range and shaping notes
- endpoint descriptions
- runtime metadata such as Python version and Docker/HF Spaces support

## Notes and Gotchas

- Use `pip install -r requirements.txt`, not `pip install requirements.txt`
- for benchmark submissions, use the injected `API_BASE_URL` and `API_KEY` exactly as provided
- `submit` is required for final grading, but the grader auto-submits if needed
- the environment may auto-submit when `max_attempts` is reached
- task scores are intentionally bounded below `1.0`
- `PRIORITY_HIRE_TASK` currently defaults to a single task in `inference.py`

## Summary

Priority Hire Env is a compact but expressive benchmark for testing agentic planning in a constrained scheduling problem. If you want to modify task difficulty, add new scenarios, or change grading behavior, `server/environment.py` is the main place to work.
