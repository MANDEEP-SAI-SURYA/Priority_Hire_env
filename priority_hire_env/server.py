from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import TypeAdapter

from .env import PriorityHireEnv
from .models import (
    ActionModel,
    HealthResponse,
    MetadataResponse,
    Observation,
    ResetRequest,
    SchemaResponse,
    StateResponse,
    Transition,
)
from .tasks import list_task_names


APP_TITLE = "PriorityHireEnv"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Dynamic interview scheduling environment for priority-aware hiring batches."
app = FastAPI(title=APP_TITLE, version=APP_VERSION, description=APP_DESCRIPTION)
_env = PriorityHireEnv(task_name=os.getenv("PRIORITY_HIRE_TASK", "easy_critical_backend"))


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return HealthResponse()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        name=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        benchmark=PriorityHireEnv.benchmark_name,
        tasks=list_task_names(),
    )


@app.get("/schema", response_model=SchemaResponse)
async def schema() -> SchemaResponse:
    return SchemaResponse(
        action=TypeAdapter(ActionModel).json_schema(),
        observation=Observation.model_json_schema(),
        state=StateResponse.model_json_schema(),
    )


@app.post("/mcp")
async def mcp(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("id")
    method = payload.get("method")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": APP_TITLE, "version": APP_VERSION},
                "capabilities": {},
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


@app.get("/tasks")
async def tasks() -> dict:
    return {"tasks": list_task_names()}


@app.post("/reset", response_model=StateResponse)
async def reset(request: Optional[ResetRequest] = None) -> StateResponse:
    task_name = None if request is None else request.task_name
    await _env.areset(task_name=task_name)
    return await _env.astate()


@app.post("/step", response_model=Transition)
async def step(action: ActionModel) -> Transition:
    return await _env.astep(action)


@app.get("/state", response_model=StateResponse)
async def state() -> StateResponse:
    return await _env.astate()
