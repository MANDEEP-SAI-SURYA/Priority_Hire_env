from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import PriorityHireAction, PriorityHireObservation, PriorityHireState


class PriorityHireEnv(EnvClient[PriorityHireAction, PriorityHireObservation, PriorityHireState]):
    def _step_payload(self, action: PriorityHireAction) -> dict:
        payload = {"kind": action.kind}
        if action.candidate_id:
            payload["candidate_id"] = action.candidate_id
        if action.interviewer_id:
            payload["interviewer_id"] = action.interviewer_id
        if action.slot_id:
            payload["slot_id"] = action.slot_id
        return payload

    def _parse_result(self, payload: dict) -> StepResult[PriorityHireObservation]:
        observation = PriorityHireObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict) -> PriorityHireState:
        return PriorityHireState.model_validate(payload)


__all__ = [
    "PriorityHireAction",
    "PriorityHireEnv",
    "PriorityHireObservation",
    "PriorityHireState",
]
