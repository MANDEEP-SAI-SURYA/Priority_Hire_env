from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from .models import (
    ActionModel,
    CandidateObservation,
    GlobalContext,
    InterviewerObservation,
    Observation,
    RewardModel,
    ScheduleAction,
    StateResponse,
    Transition,
)
from .tasks import TASKS, TaskDefinition


class PriorityHireEnv:
    benchmark_name = "priority_hire"
    min_score = 0.1001
    max_score = 0.9899

    def __init__(self, task_name: str = "easy_critical_backend"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task_name: {task_name}")
        self.task_name = task_name
        self.task_definition: TaskDefinition = TASKS[task_name]
        self.max_steps = self.task_definition["max_steps"]
        self._done = False
        self._last_action_error: Optional[str] = None
        self._last_action_summary: Optional[str] = None
        self.reset()

    @classmethod
    async def from_docker_image(
        cls, image_name: Optional[str] = None, task_name: Optional[str] = None
    ) -> "PriorityHireEnv":
        del image_name
        return cls(task_name=task_name or "easy_critical_backend")

    @classmethod
    async def from_local(cls, task_name: Optional[str] = None) -> "PriorityHireEnv":
        return cls(task_name=task_name or "easy_critical_backend")

    def reset(self, task_name: Optional[str] = None) -> Observation:
        if task_name:
            if task_name not in TASKS:
                raise ValueError(f"Unknown task_name: {task_name}")
            self.task_name = task_name
            self.task_definition = TASKS[task_name]
            self.max_steps = self.task_definition["max_steps"]

        self._candidates: Dict[str, Dict[str, object]] = {
            item["candidate_id"]: {
                **deepcopy(item),
                "status": "pending",
                "scheduled_slot": None,
                "scheduled_interviewer": None,
            }
            for item in self.task_definition["candidates"]
        }
        self._interviewers: Dict[str, Dict[str, object]] = {
            item["interviewer_id"]: {**deepcopy(item), "scheduled_count": 0}
            for item in self.task_definition["interviewers"]
        }
        self._used_slots: set[Tuple[str, str]] = set()
        self._schedule_log: List[Dict[str, object]] = []
        self._step_count = 0
        self._done = False
        self._last_action_error = None
        self._last_action_summary = "Environment reset"
        return self._build_observation()

    async def areset(self, task_name: Optional[str] = None) -> Observation:
        return self.reset(task_name=task_name)

    def state(self) -> StateResponse:
        return StateResponse(observation=self._build_observation(), done=self._done, info=self._build_info())

    async def astate(self) -> StateResponse:
        return self.state()

    def step(self, action: ActionModel) -> Tuple[Observation, RewardModel, bool, Dict[str, object]]:
        if self._done:
            reward = RewardModel(
                reward=-0.25,
                components={"extra_action_penalty": -0.25},
                normalized_score=self.compute_score(),
                reason="Episode already finished",
            )
            self._last_action_error = "episode_already_done"
            self._last_action_summary = reward.reason
            return self._build_observation(), reward, True, self._build_info()

        self._step_count += 1
        reward_value = 0.0
        components: Dict[str, float] = {}
        self._last_action_error = None

        if isinstance(action, ScheduleAction):
            reward_value, components = self._handle_schedule(action)
        elif action.kind == "defer":
            reward_value, components = self._handle_defer(action.candidate_id)
        else:
            reward_value, components = 0.0, {"submit": 0.0}
            self._done = True
            self._last_action_summary = "Scheduling batch submitted"

        missed_penalty = self._advance_deadlines()
        if missed_penalty:
            reward_value += missed_penalty
            components["deadline_miss_penalty"] = missed_penalty

        if self._step_count >= self.max_steps:
            self._done = True
        if self._all_candidates_terminal() or self._all_capacity_consumed():
            self._done = True

        score = self.compute_score()
        reward = RewardModel(
            reward=round(reward_value, 4),
            components={key: round(value, 4) for key, value in components.items()},
            normalized_score=score,
            reason=self._last_action_summary or "step_completed",
        )
        return self._build_observation(), reward, self._done, self._build_info()

    async def astep(self, action: ActionModel) -> Transition:
        observation, reward, done, info = self.step(action)
        return Transition(observation=observation, reward=reward, done=done, info=info)

    async def close(self) -> None:
        return None

    def compute_score(self) -> float:
        candidates = list(self._candidates.values())
        scheduled = [candidate for candidate in candidates if candidate["status"] == "scheduled"]
        unresolved = [candidate for candidate in candidates if candidate["status"] in {"pending", "missed"}]

        total_priority_mass = sum(self._priority_mass(candidate) for candidate in candidates) or 1.0
        scheduled_priority_mass = sum(self._priority_mass(candidate) for candidate in scheduled)
        priority_correctness = scheduled_priority_mass / total_priority_mass

        ideal_fit = sum(sorted((candidate["fit_score"] for candidate in candidates), reverse=True)[: len(scheduled)]) or 1.0
        achieved_fit = sum(candidate["fit_score"] for candidate in scheduled)
        fit_utilization = achieved_fit / ideal_fit if scheduled else 0.0

        specialization_match = 1.0
        if scheduled:
            specialization_match = sum(
                1.0
                for candidate in scheduled
                if self._scheduled_specialization(candidate["candidate_id"]) == candidate["required_specialization"]
            ) / len(scheduled)

        slot_validity = 1.0
        if self._schedule_log:
            slot_validity = sum(1.0 for item in self._schedule_log if item["valid"]) / len(self._schedule_log)

        missed_mass = sum(self._priority_mass(candidate) for candidate in unresolved if candidate["status"] == "missed")
        deadline_handling = max(0.0, 1.0 - (missed_mass / total_priority_mass))
        scarce_slot_preservation = max(0.0, 1.0 - self._scarce_slot_waste_ratio())

        efficiency = min(1.0, len(scheduled) / max(1, self._step_count))
        if self._done and self._all_candidates_terminal():
            efficiency = min(1.0, efficiency + 0.1)

        score = (
            0.22 * priority_correctness
            + 0.16 * fit_utilization
            + 0.16 * specialization_match
            + 0.12 * slot_validity
            + 0.16 * deadline_handling
            + 0.10 * scarce_slot_preservation
            + 0.08 * efficiency
        )
        return round(min(max(score, self.min_score), self.max_score), 4)

    def _handle_schedule(self, action: ScheduleAction) -> Tuple[float, Dict[str, float]]:
        candidate = self._candidates.get(action.candidate_id)
        interviewer = self._interviewers.get(action.interviewer_id)
        if candidate is None or interviewer is None:
            self._last_action_error = "unknown_candidate_or_interviewer"
            self._last_action_summary = "Invalid schedule action"
            self._schedule_log.append({"valid": False, "reason": self._last_action_error})
            return -1.2, {"invalid_scheduling": -1.2}

        if candidate["status"] != "pending":
            self._last_action_error = "candidate_not_pending"
            self._last_action_summary = "Candidate was already processed"
            self._schedule_log.append({"valid": False, "reason": self._last_action_error})
            return -1.0, {"repeated_action_penalty": -1.0}

        valid, error = self._is_valid_schedule(candidate, interviewer, action.slot_id)
        if not valid:
            self._last_action_error = error
            self._last_action_summary = "Schedule action rejected"
            self._schedule_log.append({"valid": False, "reason": error})
            return -1.5, {"invalid_scheduling": -1.5}

        candidate["status"] = "scheduled"
        candidate["scheduled_slot"] = action.slot_id
        candidate["scheduled_interviewer"] = action.interviewer_id
        interviewer["scheduled_count"] = int(interviewer["scheduled_count"]) + 1
        self._used_slots.add((action.interviewer_id, action.slot_id))

        components = {
            "schedule_success": 0.6,
            "priority_reward": 0.1 * int(candidate["priority"]),
            "urgency_reward": 0.06 * int(candidate["urgency"]),
            "fit_reward": 0.5 * float(candidate["fit_score"]),
            "specialization_match": 0.35,
            "slot_validity": 0.2,
            "deadline_risk_reduction": self._deadline_reward(candidate),
        }

        scarce_penalty = self._scarce_slot_penalty(candidate, interviewer, action.slot_id)
        if scarce_penalty:
            components["scarce_slot_penalty"] = scarce_penalty

        ordering_penalty = self._ordering_penalty(candidate, interviewer, action.slot_id)
        if ordering_penalty:
            components["ordering_penalty"] = ordering_penalty

        components["slot_efficiency"] = self._slot_efficiency_bonus(interviewer)
        self._last_action_summary = f"Scheduled {action.candidate_id} with {action.interviewer_id} at {action.slot_id}"
        self._schedule_log.append(
            {
                "valid": True,
                "candidate_id": action.candidate_id,
                "interviewer_id": action.interviewer_id,
                "slot_id": action.slot_id,
            }
        )
        return sum(components.values()), components

    def _handle_defer(self, candidate_id: str) -> Tuple[float, Dict[str, float]]:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            self._last_action_error = "unknown_candidate"
            self._last_action_summary = "Unknown candidate could not be deferred"
            return -0.8, {"invalid_defer": -0.8}

        if candidate["status"] != "pending":
            self._last_action_error = "candidate_not_pending"
            self._last_action_summary = "Candidate was already processed"
            return -0.6, {"repeated_action_penalty": -0.6}

        schedulable = self._candidate_has_any_valid_schedule(candidate_id)
        penalty = -0.15
        if schedulable and self._priority_mass(candidate) >= 16:
            penalty = -0.9
        elif schedulable and self._priority_mass(candidate) >= 10:
            penalty = -0.5
        self._last_action_summary = f"Deferred candidate {candidate_id}"
        return penalty, {"defer_penalty": penalty}

    def _advance_deadlines(self) -> float:
        penalty = 0.0
        for candidate in self._candidates.values():
            if candidate["status"] == "pending":
                candidate["deadline"] = int(candidate["deadline"]) - 1
                if int(candidate["deadline"]) < 0:
                    candidate["status"] = "missed"
                    penalty -= 1.0 + 0.12 * int(candidate["priority"]) + 0.08 * int(candidate["urgency"])
        return penalty

    def _build_observation(self) -> Observation:
        pending_candidates = [
            CandidateObservation(
                candidate_id=item["candidate_id"],
                role=item["role"],
                priority=int(item["priority"]),
                urgency=int(item["urgency"]),
                fit_score=float(item["fit_score"]),
                deadline=int(item["deadline"]),
                available_slots=list(item["available_slots"]),
                required_specialization=item["required_specialization"],
                status=item["status"],
            )
            for item in self._candidates.values()
            if item["status"] == "pending"
        ]
        pending_candidates.sort(
            key=lambda candidate: (
                -(candidate.priority * 10 + candidate.urgency * 3 + candidate.fit_score),
                candidate.deadline,
                candidate.candidate_id,
            )
        )

        interviewer_pool = [
            InterviewerObservation(
                interviewer_id=item["interviewer_id"],
                specialization=item["specialization"],
                available_slots=list(item["available_slots"]),
                max_capacity=int(item["max_capacity"]),
                scheduled_count=int(item["scheduled_count"]),
            )
            for item in self._interviewers.values()
        ]
        interviewer_pool.sort(key=lambda interviewer: interviewer.interviewer_id)

        return Observation(
            pending_candidates_queue=pending_candidates,
            interviewer_pool=interviewer_pool,
            global_context=GlobalContext(
                remaining_slots=self._remaining_capacity(),
                remaining_candidates=len(pending_candidates),
                remaining_critical_candidates=sum(
                    1 for candidate in pending_candidates if (candidate.priority >= 4 or candidate.urgency >= 4)
                ),
                step_count=self._step_count,
                max_steps=self.max_steps,
                task_name=self.task_name,
            ),
            last_action_error=self._last_action_error,
            last_action_summary=self._last_action_summary,
        )

    def _build_info(self) -> Dict[str, object]:
        return {
            "task_name": self.task_name,
            "benchmark": self.benchmark_name,
            "score": self.compute_score(),
            "done_reason": self._done_reason(),
            "schedule_log": deepcopy(self._schedule_log),
        }

    def _done_reason(self) -> str:
        if not self._done:
            return "in_progress"
        if self._step_count >= self.max_steps:
            return "max_steps"
        if self._all_candidates_terminal():
            return "all_candidates_processed"
        if self._all_capacity_consumed():
            return "all_slots_filled"
        return "submitted"

    def _is_valid_schedule(
        self, candidate: Dict[str, object], interviewer: Dict[str, object], slot_id: str
    ) -> Tuple[bool, Optional[str]]:
        if slot_id not in candidate["available_slots"]:
            return False, "candidate_unavailable_for_slot"
        if slot_id not in interviewer["available_slots"]:
            return False, "interviewer_unavailable_for_slot"
        if interviewer["specialization"] != candidate["required_specialization"]:
            return False, "specialization_mismatch"
        if int(interviewer["scheduled_count"]) >= int(interviewer["max_capacity"]):
            return False, "interviewer_at_capacity"
        if (interviewer["interviewer_id"], slot_id) in self._used_slots:
            return False, "slot_already_used"
        return True, None

    def _candidate_has_any_valid_schedule(self, candidate_id: str) -> bool:
        candidate = self._candidates[candidate_id]
        for interviewer in self._interviewers.values():
            for slot_id in candidate["available_slots"]:
                valid, _ = self._is_valid_schedule(candidate, interviewer, slot_id)
                if valid:
                    return True
        return False

    def _priority_mass(self, candidate: Dict[str, object]) -> float:
        return float(int(candidate["priority"]) * 2 + int(candidate["urgency"]) * 1.5 + float(candidate["fit_score"]) * 2)

    def _deadline_reward(self, candidate: Dict[str, object]) -> float:
        deadline = int(candidate["deadline"])
        if deadline <= 0:
            return 0.55
        if deadline == 1:
            return 0.35
        if deadline == 2:
            return 0.15
        return 0.05

    def _scarce_slot_penalty(
        self, candidate: Dict[str, object], interviewer: Dict[str, object], slot_id: str
    ) -> float:
        if not self._is_interviewer_scarce(interviewer["interviewer_id"]):
            return 0.0
        higher_priority_waiting = any(
            other["status"] == "pending"
            and other["required_specialization"] == interviewer["specialization"]
            and self._priority_mass(other) > self._priority_mass(candidate)
            and slot_id in other["available_slots"]
            for other in self._candidates.values()
            if other["candidate_id"] != candidate["candidate_id"]
        )
        if higher_priority_waiting and int(candidate["priority"]) <= 2:
            return -0.9
        if higher_priority_waiting:
            return -0.4
        return 0.0

    def _ordering_penalty(
        self, candidate: Dict[str, object], interviewer: Dict[str, object], slot_id: str
    ) -> float:
        current_mass = self._priority_mass(candidate)
        for other in self._candidates.values():
            if other["status"] != "pending" or other["candidate_id"] == candidate["candidate_id"]:
                continue
            if other["required_specialization"] != interviewer["specialization"]:
                continue
            if slot_id not in other["available_slots"]:
                continue
            if self._priority_mass(other) >= current_mass + 4:
                return -0.7
        return 0.0

    def _slot_efficiency_bonus(self, interviewer: Dict[str, object]) -> float:
        remaining_after = int(interviewer["max_capacity"]) - int(interviewer["scheduled_count"])
        return 0.08 if remaining_after == 0 else 0.02

    def _scheduled_specialization(self, candidate_id: str) -> Optional[str]:
        interviewer_id = self._candidates[candidate_id]["scheduled_interviewer"]
        if not interviewer_id:
            return None
        interviewer = self._interviewers.get(str(interviewer_id))
        return None if interviewer is None else str(interviewer["specialization"])

    def _scarce_slot_waste_ratio(self) -> float:
        scarce_interviewers = [item for item in self._interviewers if self._is_interviewer_scarce(item)]
        if not scarce_interviewers:
            return 0.0

        waste = 0.0
        total = 0.0
        for interviewer_id in scarce_interviewers:
            interviewer = self._interviewers[interviewer_id]
            specialization = interviewer["specialization"]
            relevant = [
                candidate
                for candidate in self._candidates.values()
                if candidate["required_specialization"] == specialization
            ]
            if not relevant:
                continue
            total += 1.0
            scheduled = [
                candidate for candidate in relevant if candidate["scheduled_interviewer"] == interviewer_id
            ]
            if scheduled:
                best_mass = max(self._priority_mass(candidate) for candidate in relevant)
                chosen_mass = max(self._priority_mass(candidate) for candidate in scheduled)
                waste += max(0.0, (best_mass - chosen_mass) / max(best_mass, 1.0))
        return waste / total if total else 0.0

    def _is_interviewer_scarce(self, interviewer_id: str) -> bool:
        interviewer = self._interviewers[interviewer_id]
        same_specialty = [
            other for other in self._interviewers.values() if other["specialization"] == interviewer["specialization"]
        ]
        total_capacity = sum(int(other["max_capacity"]) for other in same_specialty)
        total_demand = sum(
            1
            for candidate in self._candidates.values()
            if candidate["required_specialization"] == interviewer["specialization"]
        )
        return total_capacity <= total_demand

    def _remaining_capacity(self) -> int:
        return sum(
            max(0, int(item["max_capacity"]) - int(item["scheduled_count"]))
            for item in self._interviewers.values()
        )

    def _all_candidates_terminal(self) -> bool:
        return all(item["status"] in {"scheduled", "missed"} for item in self._candidates.values())

    def _all_capacity_consumed(self) -> bool:
        return self._remaining_capacity() <= 0
