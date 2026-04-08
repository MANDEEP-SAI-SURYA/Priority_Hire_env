from __future__ import annotations

import os
import sys

from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import PriorityHireAction, PriorityHireObservation
from server.environment import PriorityHireEnvironment


app = create_fastapi_app(PriorityHireEnvironment, PriorityHireAction, PriorityHireObservation)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        content={
            "name": "PriorityHireEnv",
            "version": "0.1.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "tasks": "/tasks",
                "grader": "/grader",
                "reset": "/reset",
                "step": "/step",
                "state": "/state",
            },
        }
    )


@app.get("/tasks")
def get_tasks() -> JSONResponse:
    tasks = PriorityHireEnvironment.list_tasks()
    return JSONResponse(content={"tasks": tasks, "total": len(tasks)})


@app.get("/grader")
def run_grader(task_name: str) -> JSONResponse:
    return JSONResponse(content=PriorityHireEnvironment.run_grader(task_name))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
