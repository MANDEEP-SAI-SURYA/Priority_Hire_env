from __future__ import annotations

from typing import Iterable, List, Optional

from .models import DeferAction, Observation, ScheduleAction, SubmitAction


def fifo_policy(observation: Observation) -> object:
    candidates = observation.pending_candidates_queue
    if not candidates:
        return SubmitAction()
    candidate = candidates[0]
    return _best_valid_schedule(observation, candidate.candidate_id) or DeferAction(candidate_id=candidate.candidate_id)


def priority_only_policy(observation: Observation) -> object:
    candidates = sorted(
        observation.pending_candidates_queue,
        key=lambda item: (-item.priority, -item.urgency, item.deadline, item.candidate_id),
    )
    if not candidates:
        return SubmitAction()
    for candidate in candidates:
        action = _best_valid_schedule(observation, candidate.candidate_id)
        if action:
            return action
    return DeferAction(candidate_id=candidates[0].candidate_id)


def fit_only_policy(observation: Observation) -> object:
    candidates = sorted(
        observation.pending_candidates_queue,
        key=lambda item: (-item.fit_score, -item.priority, item.deadline, item.candidate_id),
    )
    if not candidates:
        return SubmitAction()
    for candidate in candidates:
        action = _best_valid_schedule(observation, candidate.candidate_id)
        if action:
            return action
    return DeferAction(candidate_id=candidates[0].candidate_id)


def priority_fit_policy(observation: Observation) -> object:
    candidates = sorted(
        observation.pending_candidates_queue,
        key=lambda item: (
            -(item.priority * 2.5 + item.urgency * 2.0 + item.fit_score * 3.0 - item.deadline * 0.4),
            item.deadline,
            item.candidate_id,
        ),
    )
    if not candidates:
        return SubmitAction()
    for candidate in candidates:
        action = _best_valid_schedule(observation, candidate.candidate_id)
        if action:
            return action
    return DeferAction(candidate_id=candidates[0].candidate_id)


def _best_valid_schedule(observation: Observation, candidate_id: str) -> Optional[ScheduleAction]:
    candidate = next((item for item in observation.pending_candidates_queue if item.candidate_id == candidate_id), None)
    if candidate is None:
        return None

    same_specialization = [
        item.candidate_id
        for item in observation.pending_candidates_queue
        if item.required_specialization == candidate.required_specialization
    ]
    remaining_specialist_capacity = sum(
        interviewer.max_capacity - interviewer.scheduled_count
        for interviewer in observation.interviewer_pool
        if interviewer.specialization == candidate.required_specialization
    )
    scarce = remaining_specialist_capacity <= len(same_specialization)

    best_score = None
    best_action = None
    for interviewer in observation.interviewer_pool:
        if interviewer.specialization != candidate.required_specialization:
            continue
        if interviewer.scheduled_count >= interviewer.max_capacity:
            continue
        for slot_id in _intersect(candidate.available_slots, interviewer.available_slots):
            score = (
                candidate.priority * 2.3
                + candidate.urgency * 1.8
                + candidate.fit_score * 2.5
                - candidate.deadline * 0.5
                + (0.3 if scarce and candidate.priority >= 4 else 0.0)
            )
            value = (score, -interviewer.scheduled_count, slot_id, interviewer.interviewer_id)
            if best_score is None or value > best_score:
                best_score = value
                best_action = ScheduleAction(
                    candidate_id=candidate.candidate_id,
                    interviewer_id=interviewer.interviewer_id,
                    slot_id=slot_id,
                )
    return best_action


def _intersect(left: Iterable[str], right: Iterable[str]) -> List[str]:
    right_set = set(right)
    return sorted(item for item in left if item in right_set)
