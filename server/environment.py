from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import PriorityHireAction, PriorityHireObservation, PriorityHireState


MIN_SCORE = 0.1001
MAX_SCORE = 0.9899


class CandidateSeed(TypedDict):
    candidate_id: str
    role: str
    priority: int
    urgency: int
    fit_score: float
    deadline: int
    available_slots: List[str]
    required_specialization: str


class InterviewerSeed(TypedDict):
    interviewer_id: str
    specialization: str
    available_slots: List[str]
    max_capacity: int


class TaskDefinition(TypedDict):
    name: str
    difficulty: str
    description: str
    max_steps: int
    candidates: List[CandidateSeed]
    interviewers: List[InterviewerSeed]


TASKS: Dict[str, TaskDefinition] = {
    "easy_critical_backend": {
        "name": "easy_critical_backend",
        "difficulty": "easy",
        "description": "Enough capacity exists, but a single backend candidate is clearly the most urgent.",
        "max_steps": 8,
        "candidates": [
            {"candidate_id": "cand_backend_1", "role": "Senior Backend Engineer", "priority": 5, "urgency": 5, "fit_score": 0.93, "deadline": 2, "available_slots": ["s1", "s2"], "required_specialization": "backend"},
            {"candidate_id": "cand_frontend_1", "role": "Frontend Engineer", "priority": 2, "urgency": 2, "fit_score": 0.74, "deadline": 4, "available_slots": ["s2", "s3"], "required_specialization": "frontend"},
            {"candidate_id": "cand_data_1", "role": "Data Analyst", "priority": 3, "urgency": 2, "fit_score": 0.81, "deadline": 3, "available_slots": ["s1", "s3"], "required_specialization": "data"},
        ],
        "interviewers": [
            {"interviewer_id": "int_backend_a", "specialization": "backend", "available_slots": ["s1", "s2"], "max_capacity": 2},
            {"interviewer_id": "int_frontend_a", "specialization": "frontend", "available_slots": ["s2", "s3"], "max_capacity": 1},
            {"interviewer_id": "int_data_a", "specialization": "data", "available_slots": ["s1", "s3"], "max_capacity": 1},
        ],
    },
    "medium_scarce_ml_specialist": {
        "name": "medium_scarce_ml_specialist",
        "difficulty": "medium",
        "description": "A scarce ML specialist must be preserved for the urgent candidate rather than a lower-priority high-fit profile.",
        "max_steps": 9,
        "candidates": [
            {"candidate_id": "cand_ml_urgent", "role": "ML Platform Engineer", "priority": 5, "urgency": 4, "fit_score": 0.82, "deadline": 1, "available_slots": ["s1"], "required_specialization": "ml"},
            {"candidate_id": "cand_ml_nice", "role": "ML Research Engineer", "priority": 2, "urgency": 2, "fit_score": 0.94, "deadline": 3, "available_slots": ["s1", "s2"], "required_specialization": "ml"},
            {"candidate_id": "cand_general_backend", "role": "Backend Engineer", "priority": 4, "urgency": 3, "fit_score": 0.86, "deadline": 2, "available_slots": ["s2", "s3"], "required_specialization": "backend"},
        ],
        "interviewers": [
            {"interviewer_id": "int_ml_only", "specialization": "ml", "available_slots": ["s1"], "max_capacity": 1},
            {"interviewer_id": "int_backend_b", "specialization": "backend", "available_slots": ["s2", "s3"], "max_capacity": 2},
        ],
    },
    "hard_multi_tradeoff": {
        "name": "hard_multi_tradeoff",
        "difficulty": "hard",
        "description": "Competing deadlines, scarce security slots, and conflicting fit-vs-urgency decisions require careful scheduling.",
        "max_steps": 12,
        "candidates": [
            {"candidate_id": "cand_sec_critical", "role": "Security Architect", "priority": 5, "urgency": 5, "fit_score": 0.79, "deadline": 1, "available_slots": ["s1", "s2"], "required_specialization": "security"},
            {"candidate_id": "cand_sec_fit", "role": "Security Engineer", "priority": 3, "urgency": 2, "fit_score": 0.97, "deadline": 3, "available_slots": ["s2"], "required_specialization": "security"},
            {"candidate_id": "cand_backend_urgent", "role": "Staff Backend Engineer", "priority": 4, "urgency": 5, "fit_score": 0.88, "deadline": 1, "available_slots": ["s1", "s3"], "required_specialization": "backend"},
            {"candidate_id": "cand_frontend_fit", "role": "Senior Frontend Engineer", "priority": 2, "urgency": 2, "fit_score": 0.95, "deadline": 4, "available_slots": ["s3", "s4"], "required_specialization": "frontend"},
            {"candidate_id": "cand_data_urgent", "role": "Analytics Engineer", "priority": 4, "urgency": 4, "fit_score": 0.84, "deadline": 2, "available_slots": ["s2", "s4"], "required_specialization": "data"},
            {"candidate_id": "cand_backend_low", "role": "Backend Engineer", "priority": 1, "urgency": 1, "fit_score": 0.77, "deadline": 4, "available_slots": ["s1", "s4"], "required_specialization": "backend"},
        ],
        "interviewers": [
            {"interviewer_id": "int_security_only", "specialization": "security", "available_slots": ["s1", "s2"], "max_capacity": 1},
            {"interviewer_id": "int_backend_c", "specialization": "backend", "available_slots": ["s1", "s3", "s4"], "max_capacity": 2},
            {"interviewer_id": "int_frontend_b", "specialization": "frontend", "available_slots": ["s3", "s4"], "max_capacity": 1},
            {"interviewer_id": "int_data_b", "specialization": "data", "available_slots": ["s2", "s4"], "max_capacity": 1},
        ],
    },
    "medium_deadline_pressure": {
        "name": "medium_deadline_pressure",
        "difficulty": "medium",
        "description": "All candidates have tight deadlines; urgency must drive scheduling order.",
        "max_steps": 10,
        "candidates": [
            {"candidate_id": "cand_be_urgent1", "role": "Backend Engineer", "priority": 5, "urgency": 5, "fit_score": 0.88, "deadline": 1, "available_slots": ["s1", "s2"], "required_specialization": "backend"},
            {"candidate_id": "cand_fe_urgent1", "role": "Frontend Engineer", "priority": 4, "urgency": 4, "fit_score": 0.82, "deadline": 1, "available_slots": ["s1", "s3"], "required_specialization": "frontend"},
            {"candidate_id": "cand_ml_urgent1", "role": "ML Engineer", "priority": 3, "urgency": 4, "fit_score": 0.79, "deadline": 2, "available_slots": ["s2", "s3"], "required_specialization": "ml"},
            {"candidate_id": "cand_data_urgent1", "role": "Data Engineer", "priority": 3, "urgency": 3, "fit_score": 0.75, "deadline": 2, "available_slots": ["s1", "s2"], "required_specialization": "data"},
        ],
        "interviewers": [
            {"interviewer_id": "int_backend_d", "specialization": "backend", "available_slots": ["s1", "s2"], "max_capacity": 1},
            {"interviewer_id": "int_frontend_c", "specialization": "frontend", "available_slots": ["s1", "s3"], "max_capacity": 1},
            {"interviewer_id": "int_ml_b", "specialization": "ml", "available_slots": ["s2", "s3"], "max_capacity": 1},
            {"interviewer_id": "int_data_c", "specialization": "data", "available_slots": ["s1", "s2"], "max_capacity": 1},
        ],
    },
    "hard_conflicting_priorities": {
        "name": "hard_conflicting_priorities",
        "difficulty": "hard",
        "description": "Multiple high-priority candidates compete for a single scarce interviewer; trade-offs are unavoidable.",
        "max_steps": 14,
        "candidates": [
            {"candidate_id": "cand_sec_high1", "role": "Security Engineer", "priority": 5, "urgency": 5, "fit_score": 0.91, "deadline": 1, "available_slots": ["s1"], "required_specialization": "security"},
            {"candidate_id": "cand_sec_high2", "role": "Security Architect", "priority": 5, "urgency": 4, "fit_score": 0.87, "deadline": 2, "available_slots": ["s1", "s2"], "required_specialization": "security"},
            {"candidate_id": "cand_sec_mid1", "role": "Security Analyst", "priority": 3, "urgency": 3, "fit_score": 0.95, "deadline": 3, "available_slots": ["s1", "s2"], "required_specialization": "security"},
            {"candidate_id": "cand_be_conf1", "role": "Staff Backend Engineer", "priority": 4, "urgency": 4, "fit_score": 0.84, "deadline": 2, "available_slots": ["s2", "s3"], "required_specialization": "backend"},
            {"candidate_id": "cand_fe_conf1", "role": "Senior Frontend Engineer", "priority": 3, "urgency": 3, "fit_score": 0.78, "deadline": 3, "available_slots": ["s3", "s4"], "required_specialization": "frontend"},
        ],
        "interviewers": [
            {"interviewer_id": "int_security_scarce", "specialization": "security", "available_slots": ["s1", "s2"], "max_capacity": 1},
            {"interviewer_id": "int_backend_e", "specialization": "backend", "available_slots": ["s2", "s3"], "max_capacity": 2},
            {"interviewer_id": "int_frontend_d", "specialization": "frontend", "available_slots": ["s3", "s4"], "max_capacity": 1},
        ],
    },
}


def list_task_names() -> List[str]:
    return list(TASKS.keys())


@dataclass
class GradeContext:
    task_name: str | None
    score: float
    schedule_log: list[dict[str, Any]]


def _normalize_score(value: Any) -> float:
    return round(min(max(float(value), MIN_SCORE), MAX_SCORE), 4)


def _scheduled_ids(context: GradeContext) -> set[str]:
    return {str(item["candidate_id"]) for item in context.schedule_log if item.get("valid") and "candidate_id" in item}


def _valid_ratio(context: GradeContext) -> float:
    if not context.schedule_log:
        return 1.0
    return sum(1 for item in context.schedule_log if item.get("valid")) / len(context.schedule_log)


def _ctx(info: dict[str, Any]) -> GradeContext:
    return GradeContext(
        task_name=str(info.get("task_name")) if info.get("task_name") is not None else None,
        score=_normalize_score(info.get("score", MIN_SCORE)),
        schedule_log=[entry for entry in info.get("schedule_log", []) if isinstance(entry, dict)],
    )


def _apply_adjustments(base_score: float, adjustments: list[float]) -> float:
    return _normalize_score(base_score + sum(adjustments))


def grade_easy_critical_backend(info: dict[str, Any]) -> float:
    context = _ctx(info)
    scheduled = _scheduled_ids(context)
    return _apply_adjustments(context.score, [
        0.03 if "cand_backend_1" in scheduled else -0.08,
        0.015 if "cand_data_1" in scheduled else 0.0,
        0.015 if "cand_frontend_1" in scheduled else 0.0,
        0.02 if len(scheduled) == 3 else -0.02,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ])


def grade_medium_scarce_ml_specialist(info: dict[str, Any]) -> float:
    context = _ctx(info)
    scheduled = _scheduled_ids(context)
    return _apply_adjustments(context.score, [
        0.05 if "cand_ml_urgent" in scheduled else -0.1,
        -0.05 if "cand_ml_nice" in scheduled and "cand_ml_urgent" not in scheduled else 0.0,
        0.025 if "cand_general_backend" in scheduled else -0.015,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ])


def grade_hard_multi_tradeoff(info: dict[str, Any]) -> float:
    context = _ctx(info)
    scheduled = _scheduled_ids(context)
    critical_ids = {"cand_sec_critical", "cand_backend_urgent", "cand_data_urgent"}
    return _apply_adjustments(context.score, [
        0.045 * len(critical_ids & scheduled),
        -0.06 if "cand_sec_critical" not in scheduled else 0.0,
        -0.03 if "cand_sec_fit" in scheduled and "cand_sec_critical" not in scheduled else 0.0,
        0.015 if "cand_frontend_fit" in scheduled else 0.0,
        0.01 if _valid_ratio(context) == 1.0 else -0.04,
    ])


def grade_medium_deadline_pressure(info: dict[str, Any]) -> float:
    context = _ctx(info)
    scheduled = _scheduled_ids(context)
    deadline_one_ids = {"cand_be_urgent1", "cand_fe_urgent1"}
    return _apply_adjustments(context.score, [
        0.035 * len(deadline_one_ids & scheduled),
        -0.05 if not deadline_one_ids.issubset(scheduled) else 0.0,
        0.02 if "cand_ml_urgent1" in scheduled else 0.0,
        -0.015 if "cand_data_urgent1" not in scheduled else 0.0,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ])


def grade_hard_conflicting_priorities(info: dict[str, Any]) -> float:
    context = _ctx(info)
    scheduled = _scheduled_ids(context)
    top_security_ids = {"cand_sec_high1", "cand_sec_high2"}
    return _apply_adjustments(context.score, [
        0.05 if scheduled & top_security_ids else -0.1,
        -0.05 if "cand_sec_mid1" in scheduled and not (scheduled & top_security_ids) else 0.0,
        0.02 if "cand_be_conf1" in scheduled else 0.0,
        0.015 if "cand_fe_conf1" in scheduled else 0.0,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ])


GRADERS = {
    "easy_critical_backend": grade_easy_critical_backend,
    "medium_scarce_ml_specialist": grade_medium_scarce_ml_specialist,
    "hard_multi_tradeoff": grade_hard_multi_tradeoff,
    "medium_deadline_pressure": grade_medium_deadline_pressure,
    "hard_conflicting_priorities": grade_hard_conflicting_priorities,
}


class SchedulerCore:
    benchmark_name = "priority_hire"

    def __init__(self, task_name: str = "easy_critical_backend") -> None:
        self.reset(task_name=task_name)

    def reset(self, task_name: str = "easy_critical_backend") -> dict[str, Any]:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task_name: {task_name}")
        self.task_name = task_name
        self.task_definition = TASKS[task_name]
        self.max_steps = self.task_definition["max_steps"]
        self._candidates = {item["candidate_id"]: {**deepcopy(item), "status": "pending", "scheduled_slot": None, "scheduled_interviewer": None} for item in self.task_definition["candidates"]}
        self._interviewers = {item["interviewer_id"]: {**deepcopy(item), "scheduled_count": 0} for item in self.task_definition["interviewers"]}
        self._used_slots: set[Tuple[str, str]] = set()
        self._schedule_log: List[Dict[str, object]] = []
        self._step_count = 0
        self._done = False
        self._last_action_error: Optional[str] = None
        self._last_action_summary: Optional[str] = "Environment reset"
        return self._build_observation()

    def step(self, action: PriorityHireAction) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._done:
            return self._build_observation(), -0.25, True, self._build_info()
        self._step_count += 1
        reward = 0.0
        self._last_action_error = None
        if action.kind == "schedule":
            reward += self._handle_schedule(action)
        elif action.kind == "defer":
            reward += self._handle_defer(str(action.candidate_id))
        else:
            self._done = True
            self._last_action_summary = "Scheduling batch submitted"
        reward += self._advance_deadlines()
        if self._step_count >= self.max_steps or self._all_candidates_terminal() or self._all_capacity_consumed():
            self._done = True
        return self._build_observation(), round(reward, 4), self._done, self._build_info()

    def compute_score(self) -> float:
        candidates = list(self._candidates.values())
        scheduled = [c for c in candidates if c["status"] == "scheduled"]
        unresolved = [c for c in candidates if c["status"] in {"pending", "missed"}]
        total_priority_mass = sum(self._priority_mass(c) for c in candidates) or 1.0
        scheduled_priority_mass = sum(self._priority_mass(c) for c in scheduled)
        priority_correctness = scheduled_priority_mass / total_priority_mass
        ideal_fit = sum(sorted((c["fit_score"] for c in candidates), reverse=True)[: len(scheduled)]) or 1.0
        achieved_fit = sum(c["fit_score"] for c in scheduled)
        fit_utilization = achieved_fit / ideal_fit if scheduled else 0.0
        specialization_match = 1.0 if not scheduled else sum(1.0 for c in scheduled if self._scheduled_specialization(c["candidate_id"]) == c["required_specialization"]) / len(scheduled)
        slot_validity = 1.0 if not self._schedule_log else sum(1.0 for item in self._schedule_log if item["valid"]) / len(self._schedule_log)
        missed_mass = sum(self._priority_mass(c) for c in unresolved if c["status"] == "missed")
        deadline_handling = max(0.0, 1.0 - (missed_mass / total_priority_mass))
        scarce_slot_preservation = max(0.0, 1.0 - self._scarce_slot_waste_ratio())
        efficiency = min(1.0, len(scheduled) / max(1, self._step_count))
        if self._done and self._all_candidates_terminal():
            efficiency = min(1.0, efficiency + 0.1)
        return _normalize_score(
            0.22 * priority_correctness
            + 0.16 * fit_utilization
            + 0.16 * specialization_match
            + 0.12 * slot_validity
            + 0.16 * deadline_handling
            + 0.10 * scarce_slot_preservation
            + 0.08 * efficiency
        )

    def _handle_schedule(self, action: PriorityHireAction) -> float:
        candidate = self._candidates.get(str(action.candidate_id))
        interviewer = self._interviewers.get(str(action.interviewer_id))
        if candidate is None or interviewer is None:
            self._last_action_error = "unknown_candidate_or_interviewer"
            self._last_action_summary = "Invalid schedule action"
            self._schedule_log.append({"valid": False, "reason": self._last_action_error})
            return -1.2
        if candidate["status"] != "pending":
            self._last_action_error = "candidate_not_pending"
            self._last_action_summary = "Candidate was already processed"
            self._schedule_log.append({"valid": False, "reason": self._last_action_error})
            return -1.0
        valid, error = self._is_valid_schedule(candidate, interviewer, str(action.slot_id))
        if not valid:
            self._last_action_error = error
            self._last_action_summary = "Schedule action rejected"
            self._schedule_log.append({"valid": False, "reason": error})
            return -1.5
        candidate["status"] = "scheduled"
        candidate["scheduled_slot"] = action.slot_id
        candidate["scheduled_interviewer"] = action.interviewer_id
        interviewer["scheduled_count"] = int(interviewer["scheduled_count"]) + 1
        self._used_slots.add((str(action.interviewer_id), str(action.slot_id)))
        self._last_action_summary = f"Scheduled {action.candidate_id} with {action.interviewer_id} at {action.slot_id}"
        self._schedule_log.append({"valid": True, "candidate_id": action.candidate_id, "interviewer_id": action.interviewer_id, "slot_id": action.slot_id})
        return sum([
            0.6,
            0.1 * int(candidate["priority"]),
            0.06 * int(candidate["urgency"]),
            0.5 * float(candidate["fit_score"]),
            0.35,
            0.2,
            self._deadline_reward(candidate),
            self._scarce_slot_penalty(candidate, interviewer, str(action.slot_id)),
            self._ordering_penalty(candidate, interviewer, str(action.slot_id)),
            self._slot_efficiency_bonus(interviewer),
        ])

    def _handle_defer(self, candidate_id: str) -> float:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            self._last_action_error = "unknown_candidate"
            self._last_action_summary = "Unknown candidate could not be deferred"
            return -0.8
        if candidate["status"] != "pending":
            self._last_action_error = "candidate_not_pending"
            self._last_action_summary = "Candidate was already processed"
            return -0.6
        schedulable = self._candidate_has_any_valid_schedule(candidate_id)
        penalty = -0.15
        if schedulable and self._priority_mass(candidate) >= 16:
            penalty = -0.9
        elif schedulable and self._priority_mass(candidate) >= 10:
            penalty = -0.5
        self._last_action_summary = f"Deferred candidate {candidate_id}"
        return penalty

    def _advance_deadlines(self) -> float:
        penalty = 0.0
        for candidate in self._candidates.values():
            if candidate["status"] == "pending":
                candidate["deadline"] = int(candidate["deadline"]) - 1
                if int(candidate["deadline"]) < 0:
                    candidate["status"] = "missed"
                    penalty -= 1.0 + 0.12 * int(candidate["priority"]) + 0.08 * int(candidate["urgency"])
        return penalty

    def _build_observation(self) -> dict[str, Any]:
        pending_candidates = [
            {"candidate_id": item["candidate_id"], "role": item["role"], "priority": int(item["priority"]), "urgency": int(item["urgency"]), "fit_score": float(item["fit_score"]), "deadline": int(item["deadline"]), "available_slots": list(item["available_slots"]), "required_specialization": item["required_specialization"], "status": item["status"]}
            for item in self._candidates.values() if item["status"] == "pending"
        ]
        pending_candidates.sort(key=lambda c: (-(c["priority"] * 10 + c["urgency"] * 3 + c["fit_score"]), c["deadline"], c["candidate_id"]))
        interviewer_pool = [
            {"interviewer_id": item["interviewer_id"], "specialization": item["specialization"], "available_slots": list(item["available_slots"]), "max_capacity": int(item["max_capacity"]), "scheduled_count": int(item["scheduled_count"])}
            for item in self._interviewers.values()
        ]
        interviewer_pool.sort(key=lambda i: i["interviewer_id"])
        return {
            "pending_candidates_queue": pending_candidates,
            "interviewer_pool": interviewer_pool,
            "global_context": {"remaining_slots": self._remaining_capacity(), "remaining_candidates": len(pending_candidates), "remaining_critical_candidates": sum(1 for c in pending_candidates if c["priority"] >= 4 or c["urgency"] >= 4), "step_count": self._step_count, "max_steps": self.max_steps, "task_name": self.task_name, "benchmark": self.benchmark_name},
            "last_action_error": self._last_action_error,
            "last_action_summary": self._last_action_summary,
        }

    def _build_info(self) -> dict[str, Any]:
        return {"task_name": self.task_name, "benchmark": self.benchmark_name, "score": self.compute_score(), "done_reason": self._done_reason(), "schedule_log": deepcopy(self._schedule_log)}

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

    def _is_valid_schedule(self, candidate: Dict[str, object], interviewer: Dict[str, object], slot_id: str) -> Tuple[bool, Optional[str]]:
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
        return any(self._is_valid_schedule(candidate, interviewer, slot_id)[0] for interviewer in self._interviewers.values() for slot_id in candidate["available_slots"])

    def _priority_mass(self, candidate: Dict[str, object]) -> float:
        return float(int(candidate["priority"]) * 2 + int(candidate["urgency"]) * 1.5 + float(candidate["fit_score"]) * 2)

    def _deadline_reward(self, candidate: Dict[str, object]) -> float:
        deadline = int(candidate["deadline"])
        return 0.55 if deadline <= 0 else 0.35 if deadline == 1 else 0.15 if deadline == 2 else 0.05

    def _scarce_slot_penalty(self, candidate: Dict[str, object], interviewer: Dict[str, object], slot_id: str) -> float:
        if not self._is_interviewer_scarce(str(interviewer["interviewer_id"])):
            return 0.0
        higher_priority_waiting = any(other["status"] == "pending" and other["required_specialization"] == interviewer["specialization"] and self._priority_mass(other) > self._priority_mass(candidate) and slot_id in other["available_slots"] for other in self._candidates.values() if other["candidate_id"] != candidate["candidate_id"])
        if higher_priority_waiting and int(candidate["priority"]) <= 2:
            return -0.9
        return -0.4 if higher_priority_waiting else 0.0

    def _ordering_penalty(self, candidate: Dict[str, object], interviewer: Dict[str, object], slot_id: str) -> float:
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
        return 0.08 if int(interviewer["max_capacity"]) - int(interviewer["scheduled_count"]) == 0 else 0.02

    def _scheduled_specialization(self, candidate_id: str) -> Optional[str]:
        interviewer_id = self._candidates[candidate_id]["scheduled_interviewer"]
        interviewer = self._interviewers.get(str(interviewer_id)) if interviewer_id else None
        return None if interviewer is None else str(interviewer["specialization"])

    def _scarce_slot_waste_ratio(self) -> float:
        scarce_ids = [i for i in self._interviewers if self._is_interviewer_scarce(i)]
        if not scarce_ids:
            return 0.0
        waste = 0.0
        total = 0.0
        for interviewer_id in scarce_ids:
            interviewer = self._interviewers[interviewer_id]
            relevant = [candidate for candidate in self._candidates.values() if candidate["required_specialization"] == interviewer["specialization"]]
            if not relevant:
                continue
            total += 1.0
            scheduled = [candidate for candidate in relevant if candidate["scheduled_interviewer"] == interviewer_id]
            if scheduled:
                best_mass = max(self._priority_mass(candidate) for candidate in relevant)
                chosen_mass = max(self._priority_mass(candidate) for candidate in scheduled)
                waste += max(0.0, (best_mass - chosen_mass) / max(best_mass, 1.0))
        return waste / total if total else 0.0

    def _is_interviewer_scarce(self, interviewer_id: str) -> bool:
        interviewer = self._interviewers[interviewer_id]
        same_specialty = [other for other in self._interviewers.values() if other["specialization"] == interviewer["specialization"]]
        total_capacity = sum(int(other["max_capacity"]) for other in same_specialty)
        total_demand = sum(1 for candidate in self._candidates.values() if candidate["required_specialization"] == interviewer["specialization"])
        return total_capacity <= total_demand

    def _remaining_capacity(self) -> int:
        return sum(max(0, int(item["max_capacity"]) - int(item["scheduled_count"])) for item in self._interviewers.values())

    def _all_candidates_terminal(self) -> bool:
        return all(item["status"] in {"scheduled", "missed"} for item in self._candidates.values())

    def _all_capacity_consumed(self) -> bool:
        return self._remaining_capacity() <= 0


def _intersect(left: Iterable[str], right: Iterable[str]) -> List[str]:
    right_set = set(right)
    return sorted(item for item in left if item in right_set)


def _best_valid_schedule(observation: PriorityHireObservation, candidate_id: str) -> Optional[PriorityHireAction]:
    candidate = next((item for item in observation.pending_candidates_queue if item["candidate_id"] == candidate_id), None)
    if candidate is None:
        return None
    same_specialization = [item["candidate_id"] for item in observation.pending_candidates_queue if item["required_specialization"] == candidate["required_specialization"]]
    remaining_capacity = sum(interviewer["max_capacity"] - interviewer["scheduled_count"] for interviewer in observation.interviewer_pool if interviewer["specialization"] == candidate["required_specialization"])
    scarce = remaining_capacity <= len(same_specialization)
    best_score = None
    best_action = None
    for interviewer in observation.interviewer_pool:
        if interviewer["specialization"] != candidate["required_specialization"] or interviewer["scheduled_count"] >= interviewer["max_capacity"]:
            continue
        for slot_id in _intersect(candidate["available_slots"], interviewer["available_slots"]):
            score = candidate["priority"] * 2.3 + candidate["urgency"] * 1.8 + candidate["fit_score"] * 2.5 - candidate["deadline"] * 0.5 + (0.3 if scarce and candidate["priority"] >= 4 else 0.0)
            value = (score, -interviewer["scheduled_count"], slot_id, interviewer["interviewer_id"])
            if best_score is None or value > best_score:
                best_score = value
                best_action = PriorityHireAction(kind="schedule", candidate_id=candidate["candidate_id"], interviewer_id=interviewer["interviewer_id"], slot_id=slot_id)
    return best_action


def priority_fit_policy(observation: PriorityHireObservation) -> PriorityHireAction:
    candidates = sorted(observation.pending_candidates_queue, key=lambda item: (-(item["priority"] * 2.5 + item["urgency"] * 2.0 + item["fit_score"] * 3.0 - item["deadline"] * 0.4), item["deadline"], item["candidate_id"]))
    if not candidates:
        return PriorityHireAction(kind="submit")
    for candidate in candidates:
        action = _best_valid_schedule(observation, candidate["candidate_id"])
        if action:
            return action
    return PriorityHireAction(kind="defer", candidate_id=candidates[0]["candidate_id"])


class PriorityHireEnvironment(Environment[PriorityHireAction, PriorityHireObservation, PriorityHireState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._episode_id: str | None = None
        self._core = SchedulerCore()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> PriorityHireObservation:
        del seed
        task_name = kwargs.get("task_name") or kwargs.get("task_id") or "easy_critical_backend"
        self._episode_id = episode_id or str(uuid.uuid4())
        return self._to_observation(self._core.reset(task_name=task_name), None, False)

    def step(self, action: PriorityHireAction, timeout_s: Optional[float] = None, **kwargs: Any) -> PriorityHireObservation:
        del timeout_s, kwargs
        observation, reward, done, _ = self._core.step(action)
        return self._to_observation(observation, reward, done)

    @property
    def state(self) -> PriorityHireState:
        info = self._core._build_info()
        obs = self._core._build_observation()
        return PriorityHireState(episode_id=self._episode_id, step_count=int(obs["global_context"]["step_count"]), task_name=str(info["task_name"]), max_steps=int(obs["global_context"]["max_steps"]), score=float(info["score"]), done=self._core._done, done_reason=str(info["done_reason"]), schedule_log=list(info["schedule_log"]))

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(name="PriorityHireEnv", description="Dynamic interview scheduling environment for priority-aware hiring batches.", version="0.1.0")

    @staticmethod
    def list_tasks() -> list[dict[str, Any]]:
        return [{"id": task_name, "name": task["name"], "difficulty": task["difficulty"], "description": task["description"], "max_steps": task["max_steps"], "grader": GRADERS[task_name].__name__} for task_name, task in TASKS.items()]

    @staticmethod
    def run_grader(task_name: str, info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if task_name not in GRADERS:
            raise ValueError(f"Unknown task_name: {task_name}")
        final_info = info or run_task_with_policy(task_name)
        return {"task_name": task_name, "grader": GRADERS[task_name].__name__, "score": GRADERS[task_name](final_info)}

    def _to_observation(self, observation: dict[str, Any], reward: Optional[float], done: bool) -> PriorityHireObservation:
        info = self._core._build_info()
        return PriorityHireObservation(done=done, reward=reward, pending_candidates_queue=observation["pending_candidates_queue"], interviewer_pool=observation["interviewer_pool"], global_context=observation["global_context"], last_action_error=observation.get("last_action_error"), last_action_summary=observation.get("last_action_summary"), task_name=str(info["task_name"]), score=float(info["score"]), done_reason=str(info["done_reason"]), schedule_log=list(info["schedule_log"]))


def run_task_with_policy(task_name: str) -> dict[str, Any]:
    core = SchedulerCore(task_name=task_name)
    done = False
    total_reward = 0.0
    info = core._build_info()
    steps = 0
    while not done:
        observation = core._build_observation()
        obs = PriorityHireObservation(done=False, reward=None, pending_candidates_queue=observation["pending_candidates_queue"], interviewer_pool=observation["interviewer_pool"], global_context=observation["global_context"], last_action_error=observation.get("last_action_error"), last_action_summary=observation.get("last_action_summary"), task_name=str(info["task_name"]), score=float(info["score"]), done_reason=str(info["done_reason"]), schedule_log=list(info["schedule_log"]))
        action = priority_fit_policy(obs)
        _, reward, done, info = core.step(action)
        total_reward += reward
        steps += 1
    return {"task_name": task_name, "score": info["score"], "steps": steps, "reward_sum": round(total_reward, 4), "done_reason": info["done_reason"], "schedule_log": list(info["schedule_log"]), "grader_score": GRADERS[task_name](info)}


__all__ = ["GRADERS", "MAX_SCORE", "MIN_SCORE", "PriorityHireEnvironment", "SchedulerCore", "TASKS", "list_task_names", "priority_fit_policy", "run_task_with_policy"]
