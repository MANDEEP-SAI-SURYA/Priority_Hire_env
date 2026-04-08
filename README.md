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

Priority Hire Env is an OpenEnv-based interview scheduling environment for training and evaluating AI agents. The agent must schedule candidates into interviewer slots while balancing priority, urgency, specialization match, fit, and deadline pressure.

## Features

- FastAPI server built on OpenEnv
- Five tasks across easy, medium, and hard difficulty levels
- Continuous reward scoring
- Baseline grading endpoint for deterministic evaluation
- Docker-ready deployment for Hugging Face Spaces

## Requirements

- Python 3.11
- `pip`

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the API server with:

```bash
python -m server.app
```

Or run it with Uvicorn:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4
```

Once running, visit:

- `/docs` for the Swagger UI
- `/health` for a health check
- `/tasks` to inspect available tasks
- `/baseline` to run the oracle baseline

## API Endpoints

- `POST /reset` - start a new episode
- `POST /step` - submit the next action
- `GET /state` - inspect episode metadata
- `GET /tasks` - list tasks and action schema
- `POST /grader` - evaluate a proposed plan
- `GET /baseline` - run the built-in oracle baseline
- `GET /health` - service health status

## Project Metadata

Environment metadata is defined in `openenv.yaml`, including:

- task IDs and difficulty levels
- observation and action spaces
- reward shaping criteria
- runtime configuration for Docker and Hugging Face Spaces

## Docker

This repo includes a `Dockerfile` configured to expose the app on port `7860`, which matches Hugging Face Spaces defaults for Docker deployments.
