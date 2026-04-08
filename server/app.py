"""
server/app.py - FastAPI server for the PriorityHire interview scheduling environment.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "server"))

from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app

from models import PriorityHireAction, PriorityHireObservation
from server.environment import PriorityHireEnvironment, TASKS


app = create_fastapi_app(PriorityHireEnvironment, PriorityHireAction, PriorityHireObservation)


@app.get("/")
def root():
    return JSONResponse(content={
        "name": "PriorityHireEnv",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "tasks": "/tasks",
            "grader": "/grader",
            "baseline": "/baseline",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
        }
    })


@app.get("/tasks", tags=["Competition"])
def get_tasks():
    return JSONResponse(content={
        "tasks": PriorityHireEnvironment.list_tasks(),
        "total": len(TASKS),
        "action_schema": {
            "action_type": "string - schedule | defer | submit",
            "candidate_id": "string (required for schedule/defer)",
            "interviewer_id": "string (required for schedule)",
            "slot_id": "string (required for schedule)",
            "explanation": "string (optional) - Agent reasoning",
        },
    })


@app.post("/grader", tags=["Competition"])
def run_grader(task_id: str, plan_json: str = "[]"):
    result = PriorityHireEnvironment.run_grader(task_id, plan_json)
    return JSONResponse(content=result)


@app.get("/baseline", tags=["Competition"])
def run_baseline():
    baseline_scores = {}
    for task_id, task in TASKS.items():
        result = PriorityHireEnvironment.run_grader(task_id, json.dumps(task["oracle_plan"]))
        baseline_scores[task_id] = {
            "score": result["score"],
            "passed": result["passed"],
            "feedback": result["feedback"],
        }
    avg = sum(v["score"] for v in baseline_scores.values()) / len(baseline_scores)
    return JSONResponse(content={
        "baseline_agent": "oracle (submits deterministic scheduling plan)",
        "results": baseline_scores,
        "average_score": round(avg, 4),
    })


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", 4))
    uvicorn.run("server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
