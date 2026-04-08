from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import Action as OpenEnvAction
from openenv.core.env_server import Observation as OpenEnvObservation
from openenv.core.env_server import State as OpenEnvState
from pydantic import Field


class PriorityHireAction(OpenEnvAction):
    kind: Literal["schedule", "defer", "submit"]
    candidate_id: Optional[str] = None
    interviewer_id: Optional[str] = None
    slot_id: Optional[str] = None


class PriorityHireObservation(OpenEnvObservation):
    pending_candidates_queue: List[Dict[str, Any]]
    interviewer_pool: List[Dict[str, Any]]
    global_context: Dict[str, Any]
    last_action_error: Optional[str] = None
    last_action_summary: Optional[str] = None
    task_name: str = ""
    score: float = 0.0
    done_reason: str = "in_progress"
    schedule_log: List[Dict[str, Any]] = Field(default_factory=list)


class PriorityHireState(OpenEnvState):
    task_name: str = ""
    max_steps: int = 0
    score: float = 0.0
    done: bool = False
    done_reason: str = "not_started"
    schedule_log: List[Dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "PriorityHireAction",
    "PriorityHireObservation",
    "PriorityHireState",
]
