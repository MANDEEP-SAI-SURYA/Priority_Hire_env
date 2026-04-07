from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class CandidateObservation(BaseModel):
    candidate_id: str
    role: str
    priority: int = Field(ge=1, le=5)
    urgency: int = Field(ge=1, le=5)
    fit_score: float = Field(ge=0.0, le=1.0)
    deadline: int
    available_slots: List[str]
    required_specialization: str
    status: Literal["pending", "scheduled", "deferred", "missed"]


class InterviewerObservation(BaseModel):
    interviewer_id: str
    specialization: str
    available_slots: List[str]
    max_capacity: int = Field(ge=1)
    scheduled_count: int = Field(ge=0)


class GlobalContext(BaseModel):
    remaining_slots: int = Field(ge=0)
    remaining_candidates: int = Field(ge=0)
    remaining_critical_candidates: int = Field(ge=0)
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    task_name: str
    benchmark: str = "priority_hire"


class Observation(BaseModel):
    pending_candidates_queue: List[CandidateObservation]
    interviewer_pool: List[InterviewerObservation]
    global_context: GlobalContext
    last_action_error: Optional[str] = None
    last_action_summary: Optional[str] = None


class ScheduleAction(BaseModel):
    kind: Literal["schedule"] = "schedule"
    candidate_id: str
    interviewer_id: str
    slot_id: str


class DeferAction(BaseModel):
    kind: Literal["defer"] = "defer"
    candidate_id: str


class SubmitAction(BaseModel):
    kind: Literal["submit"] = "submit"


ActionModel = Annotated[
    Union[ScheduleAction, DeferAction, SubmitAction],
    Field(discriminator="kind"),
]


class RewardModel(BaseModel):
    reward: float
    components: Dict[str, float]
    normalized_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reason: str


class Transition(BaseModel):
    observation: Observation
    reward: RewardModel
    done: bool
    info: Dict[str, object]


class HealthResponse(BaseModel):
    ok: bool = True
    env: str = "PriorityHireEnv"
    status: Literal["healthy"] = "healthy"


class MetadataResponse(BaseModel):
    name: str
    description: str
    version: str
    benchmark: str
    tasks: List[str]


class SchemaResponse(BaseModel):
    action: Dict[str, object]
    observation: Dict[str, object]
    state: Dict[str, object]


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


class StateResponse(BaseModel):
    observation: Observation
    done: bool
    info: Dict[str, object]
