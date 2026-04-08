
"""
server/environment.py - Core PriorityHire interview scheduling environment logic.
"""

import copy
import json
import os
import random
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from openenv.core.env_server import Environment
from models import PriorityHireAction, PriorityHireObservation, PriorityHireState

CLAMP_LOW = 0.1001
CLAMP_HIGH = 0.9899
EPSILON = 0.0001


def _clamp_open_interval(value: float) -> float:
    low_open = CLAMP_LOW + EPSILON
    high_open = CLAMP_HIGH - EPSILON
    if value <= CLAMP_LOW:
        return round(low_open, 4)
    if value >= CLAMP_HIGH:
        return round(high_open, 4)
    return round(value, 4)


TASKS: Dict[str, Dict[str, Any]] = {
    "easy_critical_backend": {
        "id": "easy_critical_backend",
        "difficulty": "easy",
        "description": "Prioritize urgent backend candidates while keeping interviewer fit high.",
        "max_attempts": 10,
        "global_context": {
            "company": "PriorityHire",
            "role": "Backend Engineer",
            "hiring_window_days": 5,
            "objective": "Schedule strongest and most urgent backend candidates first.",
        },
        "interviewers": [
            {
                "id": "i_backend_1",
                "name": "Asha",
                "specializations": ["backend", "distributed_systems"],
                "slots": [
                    {"id": "b1_morning", "label": "Tue 10:00", "deadline_pressure": 8},
                    {"id": "b1_evening", "label": "Tue 16:00", "deadline_pressure": 6},
                ],
            },
            {
                "id": "i_general_1",
                "name": "Ravi",
                "specializations": ["generalist", "api_design"],
                "slots": [{"id": "g1_midday", "label": "Wed 12:00", "deadline_pressure": 5}],
            },
        ],
        "candidates": [
            {
                "id": "c_backend_hotfix",
                "name": "Maya",
                "priority": 10,
                "urgency": 9,
                "deadline_pressure": 9,
                "required_specialization": "backend",
                "fit_scores": {"i_backend_1": 0.93, "i_general_1": 0.72},
            },
            {
                "id": "c_backend_mid",
                "name": "Neil",
                "priority": 8,
                "urgency": 7,
                "deadline_pressure": 7,
                "required_specialization": "backend",
                "fit_scores": {"i_backend_1": 0.86, "i_general_1": 0.74},
            },
            {
                "id": "c_platform",
                "name": "Ira",
                "priority": 6,
                "urgency": 5,
                "deadline_pressure": 6,
                "required_specialization": "distributed_systems",
                "fit_scores": {"i_backend_1": 0.81, "i_general_1": 0.78},
            },
        ],
        "oracle_plan": [
            {"action_type": "schedule", "candidate_id": "c_backend_hotfix", "interviewer_id": "i_backend_1", "slot_id": "b1_morning"},
            {"action_type": "schedule", "candidate_id": "c_backend_mid", "interviewer_id": "i_backend_1", "slot_id": "b1_evening"},
            {"action_type": "schedule", "candidate_id": "c_platform", "interviewer_id": "i_general_1", "slot_id": "g1_midday"},
            {"action_type": "submit"},
        ],
    },
    "medium_scarce_ml_specialist": {
        "id": "medium_scarce_ml_specialist",
        "difficulty": "medium",
        "description": "Allocate scarce ML specialist slots to maximize match quality under pressure.",
        "max_attempts": 12,
        "global_context": {
            "company": "PriorityHire",
            "role": "ML Engineer",
            "hiring_window_days": 4,
            "objective": "Use specialist capacity wisely; avoid wasting ML-only slots.",
        },
        "interviewers": [
            {
                "id": "i_ml_1",
                "name": "Sara",
                "specializations": ["ml", "nlp"],
                "slots": [
                    {"id": "ml1_prime", "label": "Mon 09:00", "deadline_pressure": 10},
                    {"id": "ml1_backup", "label": "Mon 14:00", "deadline_pressure": 8},
                ],
            },
            {
                "id": "i_data_1",
                "name": "Om",
                "specializations": ["data", "analytics"],
                "slots": [
                    {"id": "d1_morning", "label": "Tue 11:00", "deadline_pressure": 6},
                    {"id": "d1_evening", "label": "Tue 17:00", "deadline_pressure": 5},
                ],
            },
        ],
        "candidates": [
            {
                "id": "c_ml_research",
                "name": "Kiara",
                "priority": 9,
                "urgency": 8,
                "deadline_pressure": 9,
                "required_specialization": "ml",
                "fit_scores": {"i_ml_1": 0.95, "i_data_1": 0.62},
            },
            {
                "id": "c_mle_prod",
                "name": "Dev",
                "priority": 8,
                "urgency": 8,
                "deadline_pressure": 8,
                "required_specialization": "ml",
                "fit_scores": {"i_ml_1": 0.91, "i_data_1": 0.68},
            },
            {
                "id": "c_data_eng",
                "name": "Arun",
                "priority": 6,
                "urgency": 6,
                "deadline_pressure": 6,
                "required_specialization": "data",
                "fit_scores": {"i_ml_1": 0.64, "i_data_1": 0.87},
            },
            {
                "id": "c_analytics",
                "name": "Rhea",
                "priority": 5,
                "urgency": 4,
                "deadline_pressure": 5,
                "required_specialization": "analytics",
                "fit_scores": {"i_ml_1": 0.58, "i_data_1": 0.83},
            },
        ],
        "oracle_plan": [
            {"action_type": "schedule", "candidate_id": "c_ml_research", "interviewer_id": "i_ml_1", "slot_id": "ml1_prime"},
            {"action_type": "schedule", "candidate_id": "c_mle_prod", "interviewer_id": "i_ml_1", "slot_id": "ml1_backup"},
            {"action_type": "schedule", "candidate_id": "c_data_eng", "interviewer_id": "i_data_1", "slot_id": "d1_morning"},
            {"action_type": "defer", "candidate_id": "c_analytics"},
            {"action_type": "submit"},
        ],
    },
    "hard_multi_tradeoff": {
        "id": "hard_multi_tradeoff",
        "difficulty": "hard",
        "description": "Balance competing constraints across leadership, platform, and product priorities.",
        "max_attempts": 12,
        "global_context": {
            "company": "PriorityHire",
            "role": "Senior IC + Staff mix",
            "hiring_window_days": 3,
            "objective": "Optimize priority and fit while preserving specialist bandwidth.",
        },
        "interviewers": [
            {
                "id": "i_arch_1",
                "name": "Tara",
                "specializations": ["architecture", "backend"],
                "slots": [
                    {"id": "a1_early", "label": "Wed 09:30", "deadline_pressure": 9},
                    {"id": "a1_late", "label": "Wed 15:30", "deadline_pressure": 6},
                ],
            },
            {
                "id": "i_ml_2",
                "name": "Nikhil",
                "specializations": ["ml", "systems"],
                "slots": [{"id": "ml2_prime", "label": "Thu 10:00", "deadline_pressure": 9}],
            },
            {
                "id": "i_pm_1",
                "name": "Leena",
                "specializations": ["product", "execution"],
                "slots": [{"id": "pm1_mid", "label": "Thu 13:00", "deadline_pressure": 7}],
            },
        ],
        "candidates": [
            {
                "id": "c_staff_backend",
                "name": "Jon",
                "priority": 10,
                "urgency": 8,
                "deadline_pressure": 9,
                "required_specialization": "architecture",
                "fit_scores": {"i_arch_1": 0.94, "i_ml_2": 0.72, "i_pm_1": 0.66},
            },
            {
                "id": "c_mlsys",
                "name": "Zara",
                "priority": 9,
                "urgency": 9,
                "deadline_pressure": 10,
                "required_specialization": "ml",
                "fit_scores": {"i_arch_1": 0.69, "i_ml_2": 0.96, "i_pm_1": 0.64},
            },
            {
                "id": "c_prod_lead",
                "name": "Rohan",
                "priority": 8,
                "urgency": 7,
                "deadline_pressure": 8,
                "required_specialization": "product",
                "fit_scores": {"i_arch_1": 0.63, "i_ml_2": 0.61, "i_pm_1": 0.91},
            },
            {
                "id": "c_generalist",
                "name": "Nora",
                "priority": 6,
                "urgency": 6,
                "deadline_pressure": 7,
                "required_specialization": "systems",
                "fit_scores": {"i_arch_1": 0.78, "i_ml_2": 0.84, "i_pm_1": 0.72},
            },
        ],
        "oracle_plan": [
            {"action_type": "schedule", "candidate_id": "c_mlsys", "interviewer_id": "i_ml_2", "slot_id": "ml2_prime"},
            {"action_type": "schedule", "candidate_id": "c_staff_backend", "interviewer_id": "i_arch_1", "slot_id": "a1_early"},
            {"action_type": "schedule", "candidate_id": "c_prod_lead", "interviewer_id": "i_pm_1", "slot_id": "pm1_mid"},
            {"action_type": "schedule", "candidate_id": "c_generalist", "interviewer_id": "i_arch_1", "slot_id": "a1_late"},
            {"action_type": "submit"},
        ],
    },
    "medium_deadline_pressure": {
        "id": "medium_deadline_pressure",
        "difficulty": "medium",
        "description": "Handle near-term deadlines; late scheduling heavily impacts score.",
        "max_attempts": 10,
        "global_context": {
            "company": "PriorityHire",
            "role": "Frontend + UX",
            "hiring_window_days": 2,
            "objective": "Place highest urgency candidates into earliest viable slots.",
        },
        "interviewers": [
            {
                "id": "i_frontend_1",
                "name": "Pooja",
                "specializations": ["frontend", "ui"],
                "slots": [
                    {"id": "f1_soon", "label": "Today 16:00", "deadline_pressure": 10},
                    {"id": "f1_tomorrow", "label": "Tomorrow 11:00", "deadline_pressure": 7},
                ],
            },
            {
                "id": "i_design_1",
                "name": "Kabir",
                "specializations": ["ux", "ui"],
                "slots": [{"id": "dsg1_today", "label": "Today 18:00", "deadline_pressure": 9}],
            },
        ],
        "candidates": [
            {
                "id": "c_ui_hot",
                "name": "Aditi",
                "priority": 9,
                "urgency": 10,
                "deadline_pressure": 10,
                "required_specialization": "ui",
                "fit_scores": {"i_frontend_1": 0.92, "i_design_1": 0.90},
            },
            {
                "id": "c_frontend_sr",
                "name": "Vik",
                "priority": 8,
                "urgency": 8,
                "deadline_pressure": 9,
                "required_specialization": "frontend",
                "fit_scores": {"i_frontend_1": 0.90, "i_design_1": 0.70},
            },
            {
                "id": "c_ux_mid",
                "name": "Mithra",
                "priority": 6,
                "urgency": 6,
                "deadline_pressure": 7,
                "required_specialization": "ux",
                "fit_scores": {"i_frontend_1": 0.71, "i_design_1": 0.88},
            },
        ],
        "oracle_plan": [
            {"action_type": "schedule", "candidate_id": "c_ui_hot", "interviewer_id": "i_design_1", "slot_id": "dsg1_today"},
            {"action_type": "schedule", "candidate_id": "c_frontend_sr", "interviewer_id": "i_frontend_1", "slot_id": "f1_soon"},
            {"action_type": "schedule", "candidate_id": "c_ux_mid", "interviewer_id": "i_frontend_1", "slot_id": "f1_tomorrow"},
            {"action_type": "submit"},
        ],
    },
    "hard_conflicting_priorities": {
        "id": "hard_conflicting_priorities",
        "difficulty": "hard",
        "description": "Resolve conflicts where top-priority candidates compete for overlapping specialist slots.",
        "max_attempts": 14,
        "global_context": {
            "company": "PriorityHire",
            "role": "Cross-functional founding team",
            "hiring_window_days": 3,
            "objective": "Trade off priority, urgency, fit, specialization, and deadline pressure.",
        },
        "interviewers": [
            {
                "id": "i_founder_backend",
                "name": "Meera",
                "specializations": ["backend", "architecture"],
                "slots": [{"id": "fb_critical", "label": "Fri 09:00", "deadline_pressure": 10}],
            },
            {
                "id": "i_founder_ml",
                "name": "Ishan",
                "specializations": ["ml", "systems"],
                "slots": [
                    {"id": "fm_critical", "label": "Fri 10:00", "deadline_pressure": 10},
                    {"id": "fm_late", "label": "Fri 17:00", "deadline_pressure": 5},
                ],
            },
            {
                "id": "i_founder_product",
                "name": "Sonal",
                "specializations": ["product", "go_to_market"],
                "slots": [{"id": "fp_critical", "label": "Fri 11:00", "deadline_pressure": 9}],
            },
        ],
        "candidates": [
            {
                "id": "c_backend_star",
                "name": "Ritvik",
                "priority": 10,
                "urgency": 9,
                "deadline_pressure": 9,
                "required_specialization": "backend",
                "fit_scores": {"i_founder_backend": 0.97, "i_founder_ml": 0.74, "i_founder_product": 0.61},
            },
            {
                "id": "c_ml_star",
                "name": "Sia",
                "priority": 10,
                "urgency": 10,
                "deadline_pressure": 10,
                "required_specialization": "ml",
                "fit_scores": {"i_founder_backend": 0.66, "i_founder_ml": 0.98, "i_founder_product": 0.59},
            },
            {
                "id": "c_product_star",
                "name": "Arya",
                "priority": 9,
                "urgency": 8,
                "deadline_pressure": 9,
                "required_specialization": "product",
                "fit_scores": {"i_founder_backend": 0.60, "i_founder_ml": 0.63, "i_founder_product": 0.95},
            },
            {
                "id": "c_systems_high",
                "name": "Neer",
                "priority": 8,
                "urgency": 8,
                "deadline_pressure": 8,
                "required_specialization": "systems",
                "fit_scores": {"i_founder_backend": 0.75, "i_founder_ml": 0.90, "i_founder_product": 0.64},
            },
            {
                "id": "c_gtm_mid",
                "name": "Diya",
                "priority": 6,
                "urgency": 5,
                "deadline_pressure": 6,
                "required_specialization": "go_to_market",
                "fit_scores": {"i_founder_backend": 0.58, "i_founder_ml": 0.60, "i_founder_product": 0.88},
            },
        ],
        "oracle_plan": [
            {"action_type": "schedule", "candidate_id": "c_ml_star", "interviewer_id": "i_founder_ml", "slot_id": "fm_critical"},
            {"action_type": "schedule", "candidate_id": "c_backend_star", "interviewer_id": "i_founder_backend", "slot_id": "fb_critical"},
            {"action_type": "schedule", "candidate_id": "c_product_star", "interviewer_id": "i_founder_product", "slot_id": "fp_critical"},
            {"action_type": "schedule", "candidate_id": "c_systems_high", "interviewer_id": "i_founder_ml", "slot_id": "fm_late"},
            {"action_type": "defer", "candidate_id": "c_gtm_mid"},
            {"action_type": "submit"},
        ],
    },
}


def _find_interviewer(task: Dict[str, Any], interviewer_id: str) -> Optional[Dict[str, Any]]:
    for interviewer in task["interviewers"]:
        if interviewer["id"] == interviewer_id:
            return interviewer
    return None


def _find_slot(interviewer: Dict[str, Any], slot_id: str) -> Optional[Dict[str, Any]]:
    for slot in interviewer["slots"]:
        if slot["id"] == slot_id:
            return slot
    return None


def _find_candidate(candidates: List[Dict[str, Any]], candidate_id: str) -> Optional[Dict[str, Any]]:
    for candidate in candidates:
        if candidate["id"] == candidate_id:
            return candidate
    return None


def _specialization_match(candidate: Dict[str, Any], interviewer: Dict[str, Any]) -> float:
    return 1.0 if candidate["required_specialization"] in interviewer["specializations"] else 0.35


def _score_assignment(candidate: Dict[str, Any], interviewer: Dict[str, Any], slot: Dict[str, Any]) -> float:
    priority = candidate["priority"] / 10.0
    urgency = candidate["urgency"] / 10.0
    deadline = candidate["deadline_pressure"] / 10.0
    fit = float(candidate["fit_scores"].get(interviewer["id"], 0.2))
    specialization = _specialization_match(candidate, interviewer)
    slot_weight = max(0.6, min(1.0, slot["deadline_pressure"] / 10.0))

    composite = (
        (0.30 * priority)
        + (0.23 * urgency)
        + (0.20 * fit)
        + (0.15 * specialization)
        + (0.12 * deadline)
    )
    return composite * (0.85 + (0.15 * slot_weight))

def compute_score(
    task: Dict[str, Any],
    assignments: List[Dict[str, Any]],
    deferred: List[Dict[str, Any]],
    pending: List[Dict[str, Any]],
) -> float:
    """
    Core score from scheduling quality.
    Always returns a float strictly between 0.1001 and 0.9899.
    """
    total_candidates = max(1, len(task["candidates"]))
    scheduled_count = len(assignments)

    if assignments:
        assignment_avg = sum(a["assignment_score"] for a in assignments) / len(assignments)
    else:
        assignment_avg = 0.0

    coverage = scheduled_count / total_candidates

    defer_penalty = 0.0
    if deferred:
        defer_penalty = sum(
            ((c["urgency"] / 10.0) * 0.05) + ((c["deadline_pressure"] / 10.0) * 0.05)
            for c in deferred
        )

    pending_penalty = 0.0
    if pending:
        pending_penalty = sum(
            ((c["priority"] / 10.0) * 0.06) + ((c["urgency"] / 10.0) * 0.04)
            for c in pending
        )

    raw = 0.23 + (0.57 * assignment_avg) + (0.30 * coverage) - defer_penalty - pending_penalty
    return _clamp_open_interval(raw)


def _grade_easy_critical_backend(
    task: Dict[str, Any], assignments: List[Dict[str, Any]], deferred: List[Dict[str, Any]], pending: List[Dict[str, Any]]
) -> Tuple[float, str]:
    base = compute_score(task, assignments, deferred, pending)
    bonus = 0.0
    hotfix = next((a for a in assignments if a["candidate"]["id"] == "c_backend_hotfix"), None)
    if hotfix and hotfix["step"] <= 2:
        bonus += 0.04
    if any(c["id"] == "c_backend_hotfix" for c in deferred):
        bonus -= 0.12
    score = _clamp_open_interval(base + bonus)
    return score, "Backend urgency emphasized; prioritize c_backend_hotfix early."


def _grade_medium_scarce_ml_specialist(
    task: Dict[str, Any], assignments: List[Dict[str, Any]], deferred: List[Dict[str, Any]], pending: List[Dict[str, Any]]
) -> Tuple[float, str]:
    base = compute_score(task, assignments, deferred, pending)
    ml_waste = 0
    for a in assignments:
        if a["interviewer"]["id"] == "i_ml_1" and a["candidate"]["required_specialization"] != "ml":
            ml_waste += 1
    score = _clamp_open_interval(base - (0.06 * ml_waste))
    return score, "Scarce ML specialist slots should mostly serve ML-required candidates."


def _grade_hard_multi_tradeoff(
    task: Dict[str, Any], assignments: List[Dict[str, Any]], deferred: List[Dict[str, Any]], pending: List[Dict[str, Any]]
) -> Tuple[float, str]:
    base = compute_score(task, assignments, deferred, pending)
    specialization_hits = sum(1 for a in assignments if a["specialization_match"] >= 1.0)
    modifier = 0.02 if specialization_hits >= 3 else -0.03
    score = _clamp_open_interval(base + modifier)
    return score, "Tradeoff task rewards broad high-quality specialization alignment."


def _grade_medium_deadline_pressure(
    task: Dict[str, Any], assignments: List[Dict[str, Any]], deferred: List[Dict[str, Any]], pending: List[Dict[str, Any]]
) -> Tuple[float, str]:
    base = compute_score(task, assignments, deferred, pending)
    urgency_alignment = 0.0
    for a in assignments:
        cand_deadline = a["candidate"]["deadline_pressure"] / 10.0
        slot_deadline = a["slot"]["deadline_pressure"] / 10.0
        urgency_alignment += (cand_deadline * slot_deadline)
    urgency_alignment = urgency_alignment / len(assignments) if assignments else 0.0
    score = _clamp_open_interval(base + ((urgency_alignment - 0.5) * 0.08))
    return score, "Deadline-pressure task rewards assigning urgent candidates to urgent slots."


def _grade_hard_conflicting_priorities(
    task: Dict[str, Any], assignments: List[Dict[str, Any]], deferred: List[Dict[str, Any]], pending: List[Dict[str, Any]]
) -> Tuple[float, str]:
    base = compute_score(task, assignments, deferred, pending)
    top_ids = {"c_backend_star", "c_ml_star", "c_product_star"}
    scheduled_top = sum(1 for a in assignments if a["candidate"]["id"] in top_ids)
    deferred_top = sum(1 for c in deferred if c["id"] in top_ids)
    modifier = (0.03 * scheduled_top) - (0.08 * deferred_top)
    score = _clamp_open_interval(base + modifier)
    return score, "Conflicting priorities task strongly favors placing top-priority candidates."


TASK_GRADERS = {
    "easy_critical_backend": _grade_easy_critical_backend,
    "medium_scarce_ml_specialist": _grade_medium_scarce_ml_specialist,
    "hard_multi_tradeoff": _grade_hard_multi_tradeoff,
    "medium_deadline_pressure": _grade_medium_deadline_pressure,
    "hard_conflicting_priorities": _grade_hard_conflicting_priorities,
}


class PriorityHireEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_ATTEMPTS = 12

    def __init__(self):
        self._state = PriorityHireState()
        self._task: Optional[Dict[str, Any]] = None
        self._attempt = 0
        self._last_score = 0.0
        self._pending_candidates: List[Dict[str, Any]] = []
        self._deferred_candidates: List[Dict[str, Any]] = []
        self._assignments: List[Dict[str, Any]] = []
        self._occupied_slots: set = set()

    def _make_observation(self, done: bool, reward: float, feedback: str) -> PriorityHireObservation:
        interviewer_pool = []
        for interviewer in self._task["interviewers"]:
            open_slots = [s for s in interviewer["slots"] if s["id"] not in self._occupied_slots]
            interviewer_pool.append(
                {
                    "id": interviewer["id"],
                    "name": interviewer["name"],
                    "specializations": interviewer["specializations"],
                    "available_slots": open_slots,
                }
            )

        return PriorityHireObservation(
            done=done,
            reward=reward,
            pending_candidates_queue=copy.deepcopy(self._pending_candidates),
            interviewer_pool=interviewer_pool,
            global_context=copy.deepcopy(self._task["global_context"]),
            task_description=self._task["description"],
            task_id=self._task["id"],
            difficulty=self._task["difficulty"],
            attempt_number=self._attempt,
            max_attempts=self._task.get("max_attempts", self.MAX_ATTEMPTS),
            feedback=feedback,
        )

    def reset(
        self,
        seed=None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> PriorityHireObservation:
        if seed is not None:
            random.seed(seed)

        if task_id and task_id in TASKS:
            self._task = copy.deepcopy(TASKS[task_id])
        else:
            self._task = copy.deepcopy(random.choice(list(TASKS.values())))

        self._attempt = 0
        self._last_score = 0.0
        self._pending_candidates = copy.deepcopy(self._task["candidates"])
        self._deferred_candidates = []
        self._assignments = []
        self._occupied_slots = set()

        self._state = PriorityHireState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self._task["id"],
            difficulty=self._task["difficulty"],
            max_attempts=self._task.get("max_attempts", self.MAX_ATTEMPTS),
            last_score=0.0,
            completed=False,
            scheduled_candidates=[],
            deferred_candidates=[],
        )

        return self._make_observation(
            done=False,
            reward=0.0,
            feedback="Episode started. Use schedule(candidate_id, interviewer_id, slot_id), defer(candidate_id), then submit().",
        )

    def step(
        self,
        action: PriorityHireAction,
        timeout_s=None,
        **kwargs,
    ) -> PriorityHireObservation:
        self._attempt += 1
        self._state.step_count += 1

        feedback_parts: List[str] = []
        done = False

        action_type = (action.action_type or "").strip().lower()
        if action_type not in {"schedule", "defer", "submit"}:
            feedback_parts.append("Invalid action_type. Allowed: schedule, defer, submit.")

        elif action_type == "schedule":
            candidate = _find_candidate(self._pending_candidates, action.candidate_id)
            interviewer = _find_interviewer(self._task, action.interviewer_id)
            if not candidate:
                feedback_parts.append(f"Candidate '{action.candidate_id}' is not in pending queue.")
            elif not interviewer:
                feedback_parts.append(f"Interviewer '{action.interviewer_id}' not found.")
            else:
                slot = _find_slot(interviewer, action.slot_id)
                if not slot:
                    feedback_parts.append(f"Slot '{action.slot_id}' not found for interviewer '{interviewer['id']}'.")
                elif slot["id"] in self._occupied_slots:
                    feedback_parts.append(f"Slot '{action.slot_id}' is already occupied.")
                else:
                    spec_match = _specialization_match(candidate, interviewer)
                    assignment_score = _score_assignment(candidate, interviewer, slot)
                    self._assignments.append(
                        {
                            "step": self._attempt,
                            "candidate": copy.deepcopy(candidate),
                            "interviewer": copy.deepcopy(interviewer),
                            "slot": copy.deepcopy(slot),
                            "specialization_match": spec_match,
                            "assignment_score": assignment_score,
                        }
                    )
                    self._occupied_slots.add(slot["id"])
                    self._pending_candidates = [c for c in self._pending_candidates if c["id"] != candidate["id"]]
                    feedback_parts.append(
                        f"Scheduled {candidate['id']} with {interviewer['id']} at {slot['id']} (fit={candidate['fit_scores'].get(interviewer['id'], 0.0):.2f})."
                    )

        elif action_type == "defer":
            candidate = _find_candidate(self._pending_candidates, action.candidate_id)
            if not candidate:
                feedback_parts.append(f"Candidate '{action.candidate_id}' is not in pending queue.")
            else:
                self._deferred_candidates.append(candidate)
                self._pending_candidates = [c for c in self._pending_candidates if c["id"] != candidate["id"]]
                feedback_parts.append(f"Deferred candidate {candidate['id']}.")

        elif action_type == "submit":
            grader = TASK_GRADERS[self._task["id"]]
            final_score, grader_feedback = grader(
                self._task,
                self._assignments,
                self._deferred_candidates,
                self._pending_candidates,
            )
            self._last_score = final_score
            self._state.last_score = final_score
            self._state.completed = final_score >= 0.85
            done = True
            feedback_parts.append(grader_feedback)
            feedback_parts.append(f"Final score: {final_score:.4f}")

        if not done:
            provisional = compute_score(
                self._task,
                self._assignments,
                self._deferred_candidates,
                self._pending_candidates,
            )
            self._last_score = provisional
            self._state.last_score = provisional

            if self._attempt >= self._task.get("max_attempts", self.MAX_ATTEMPTS):
                done = True
                feedback_parts.append("Max attempts reached; auto-submitting current schedule.")
                grader = TASK_GRADERS[self._task["id"]]
                final_score, grader_feedback = grader(
                    self._task,
                    self._assignments,
                    self._deferred_candidates,
                    self._pending_candidates,
                )
                self._last_score = final_score
                self._state.last_score = final_score
                self._state.completed = final_score >= 0.85
                feedback_parts.append(grader_feedback)
                feedback_parts.append(f"Final score: {final_score:.4f}")

        self._state.scheduled_candidates = [a["candidate"]["id"] for a in self._assignments]
        self._state.deferred_candidates = [c["id"] for c in self._deferred_candidates]

        return self._make_observation(
            done=done,
            reward=self._last_score,
            feedback=" | ".join(feedback_parts) if feedback_parts else "No-op action.",
        )

    @property
    def state(self) -> PriorityHireState:
        return self._state

    def get_last_score(self) -> float:
        return self._last_score

    def get_current_task(self) -> Dict[str, Any]:
        return self._task or {}

    @staticmethod
    def list_tasks() -> List[Dict[str, Any]]:
        return [
            {
                "task_id": t["id"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "action_schema": {
                    "action_type": "string - one of: schedule | defer | submit",
                    "candidate_id": "string - required for schedule/defer",
                    "interviewer_id": "string - required for schedule",
                    "slot_id": "string - required for schedule",
                    "explanation": "string (optional) - strategy trace",
                },
            }
            for t in TASKS.values()
        ]

    @staticmethod
    def run_grader(task_id: str, plan_json: str) -> Dict[str, Any]:
        if task_id not in TASKS:
            return {"error": f"Unknown task_id: {task_id}."}

        try:
            raw_plan = json.loads(plan_json) if plan_json else []
        except Exception as exc:
            return {"error": f"Invalid plan_json: {exc}"}

        if not isinstance(raw_plan, list):
            return {"error": "plan_json must decode to a list of action objects."}

        env = PriorityHireEnv()
        env.reset(task_id=task_id)

        history: List[Dict[str, Any]] = []
        last_obs = None

        for item in raw_plan:
            action = PriorityHireAction(
                action_type=str(item.get("action_type", "submit")),
                candidate_id=str(item.get("candidate_id", "")),
                interviewer_id=str(item.get("interviewer_id", "")),
                slot_id=str(item.get("slot_id", "")),
                explanation=str(item.get("explanation", "")),
            )
            last_obs = env.step(action)
            history.append({
                "action": item,
                "reward": last_obs.reward,
                "done": last_obs.done,
                "feedback": last_obs.feedback,
            })
            if last_obs.done:
                break

        if not last_obs or not last_obs.done:
            last_obs = env.step(PriorityHireAction.submit("auto-submit"))
            history.append({
                "action": {"action_type": "submit"},
                "reward": last_obs.reward,
                "done": last_obs.done,
                "feedback": last_obs.feedback,
            })

        return {
            "task_id": task_id,
            "score": float(last_obs.reward or 0.0),
            "feedback": last_obs.feedback,
            "passed": bool((last_obs.reward or 0.0) >= 0.85),
            "steps": history,
        }


PriorityHireEnvironment = PriorityHireEnv
