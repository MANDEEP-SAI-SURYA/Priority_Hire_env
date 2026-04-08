"""
models.py - Type-safe data contracts for the PriorityHire interview scheduling environment.
"""

from typing import Dict, List
from openenv.core.env_server import Action, Observation, State


class PriorityHireAction(Action):
    """
    Supported actions:
    - schedule(candidate_id, interviewer_id, slot_id)
    - defer(candidate_id)
    - submit()
    """

    action_type: str = "submit"
    candidate_id: str = ""
    interviewer_id: str = ""
    slot_id: str = ""
    explanation: str = ""

    @classmethod
    def schedule(
        cls,
        candidate_id: str,
        interviewer_id: str,
        slot_id: str,
        explanation: str = "",
    ) -> "PriorityHireAction":
        return cls(
            action_type="schedule",
            candidate_id=candidate_id,
            interviewer_id=interviewer_id,
            slot_id=slot_id,
            explanation=explanation,
        )

    @classmethod
    def defer(cls, candidate_id: str, explanation: str = "") -> "PriorityHireAction":
        return cls(action_type="defer", candidate_id=candidate_id, explanation=explanation)

    @classmethod
    def submit(cls, explanation: str = "") -> "PriorityHireAction":
        return cls(action_type="submit", explanation=explanation)


class PriorityHireObservation(Observation):
    pending_candidates_queue: List[Dict] = []
    interviewer_pool: List[Dict] = []
    global_context: Dict = {}
    task_description: str = ""
    task_id: str = ""
    difficulty: str = ""
    attempt_number: int = 0
    max_attempts: int = 12
    feedback: str = ""


class PriorityHireState(State):
    task_id: str = ""
    difficulty: str = ""
    max_attempts: int = 12
    last_score: float = 0.0
    completed: bool = False
    scheduled_candidates: List[str] = []
    deferred_candidates: List[str] = []
