---
title: PriorityHireEnv
emoji: "📅"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---


# PriorityHireEnv

`PriorityHireEnv` is a production-style OpenEnv RL environment for dynamic interview scheduling. The agent chooses which candidate to schedule, which interviewer to assign, and which slot to use while balancing business priority, urgency, fit, specialization correctness, slot scarcity, and deadline pressure.

## Included

- Typed Pydantic observation, action, reward, transition, and state models
- `reset()`, `step()`, and `state()` environment methods
- FastAPI server exposing `/`, `/health`, `/tasks`, `/reset`, `/step`, and `/state`
- Three deterministic tasks: easy, medium, and hard
- Deterministic normalized grading in `[0.0, 1.0]`
- Four baselines: FIFO, priority-only, fit-only, and priority+fit
- Root `inference.py` with structured stdout logging
- Dockerfile for container deployment

## Environment Variables

- `API_BASE_URL`: base URL for the OpenAI-compatible API
- `MODEL_NAME`: model used by `inference.py`
- `HF_TOKEN`: Hugging Face token or compatible API key
- `LOCAL_IMAGE_NAME`: optional local image name if a harness expects one

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_baselines.py
python inference.py
uvicorn priority_hire_env.server:app --host 0.0.0.0 --port 7860
```
