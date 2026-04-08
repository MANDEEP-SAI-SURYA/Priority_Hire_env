"""
client.py - Python client for the PriorityHire interview scheduling environment.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import PriorityHireAction, PriorityHireObservation, PriorityHireState


class PriorityHireEnv(EnvClient[PriorityHireAction, PriorityHireObservation, PriorityHireState]):
    """
    Client for the PriorityHire interview scheduling environment.
    """

    def _step_payload(self, action: PriorityHireAction) -> dict:
        return {
            "action_type": action.action_type,
            "candidate_id": action.candidate_id,
            "interviewer_id": action.interviewer_id,
            "slot_id": action.slot_id,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        reward = payload.get("reward")

        return StepResult(
            observation=PriorityHireObservation(
                done=done,
                reward=reward,
                pending_candidates_queue=obs_data.get("pending_candidates_queue", []),
                interviewer_pool=obs_data.get("interviewer_pool", []),
                global_context=obs_data.get("global_context", {}),
                task_description=obs_data.get("task_description", ""),
                task_id=obs_data.get("task_id", ""),
                difficulty=obs_data.get("difficulty", ""),
                attempt_number=obs_data.get("attempt_number", 0),
                max_attempts=obs_data.get("max_attempts", 12),
                feedback=obs_data.get("feedback", ""),
            ),
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> PriorityHireState:
        return PriorityHireState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", ""),
            max_attempts=payload.get("max_attempts", 12),
            last_score=payload.get("last_score", 0.0),
            completed=payload.get("completed", False),
            scheduled_candidates=payload.get("scheduled_candidates", []),
            deferred_candidates=payload.get("deferred_candidates", []),
        )
