from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .env import PriorityHireEnv


MIN_SCORE = 0.1001
MAX_SCORE = 0.9899


@dataclass
class GradeContext:
    task_name: str | None
    score: float
    schedule_log: list[dict[str, Any]]


def _normalize_score(value: Any) -> float:
    score = float(value)
    return round(min(max(score, MIN_SCORE), MAX_SCORE), 4)


def _extract_task_name(item: Any) -> str | None:
    if isinstance(item, PriorityHireEnv):
        return item.task_name

    task_name = getattr(item, "task_name", None)
    if isinstance(task_name, str):
        return task_name

    global_context = getattr(item, "global_context", None)
    if global_context is not None:
        nested_task_name = getattr(global_context, "task_name", None)
        if isinstance(nested_task_name, str):
            return nested_task_name

    if isinstance(item, dict):
        task_name = item.get("task_name")
        if isinstance(task_name, str):
            return task_name

        observation = item.get("observation")
        if observation is not None:
            nested_task_name = _extract_task_name(observation)
            if nested_task_name is not None:
                return nested_task_name

        info = item.get("info")
        if isinstance(info, dict):
            nested_task_name = info.get("task_name")
            if isinstance(nested_task_name, str):
                return nested_task_name

    return None


def _extract_info(item: Any) -> dict[str, Any] | None:
    if isinstance(item, PriorityHireEnv):
        return item.state().info

    info = getattr(item, "info", None)
    if isinstance(info, dict):
        return info

    if isinstance(item, dict):
        nested_info = item.get("info")
        if isinstance(nested_info, dict):
            return nested_info
        if "score" in item:
            return item

    return None


def _extract_score(*args: Any, **kwargs: Any) -> float:
    candidates = list(args) + list(kwargs.values())

    for item in candidates:
        if isinstance(item, PriorityHireEnv):
            return _normalize_score(item.compute_score())

    for item in candidates:
        normalized_score = getattr(item, "normalized_score", None)
        if normalized_score is not None:
            return _normalize_score(normalized_score)

        reward = getattr(item, "reward", None)
        if reward is not None:
            nested_score = getattr(reward, "normalized_score", None)
            if nested_score is not None:
                return _normalize_score(nested_score)

        info = _extract_info(item)
        if isinstance(info, dict) and "score" in info:
            return _normalize_score(info["score"])

    score = kwargs.get("score")
    if score is not None:
        return _normalize_score(score)

    raise ValueError("Unable to extract score for grader invocation")


def _build_context(*args: Any, **kwargs: Any) -> GradeContext:
    candidates = list(args) + list(kwargs.values())
    task_name = None
    schedule_log: list[dict[str, Any]] = []

    for item in candidates:
        task_name = task_name or _extract_task_name(item)
        info = _extract_info(item)
        if isinstance(info, dict):
            task_name = task_name or info.get("task_name")
            raw_log = info.get("schedule_log", [])
            if isinstance(raw_log, list):
                schedule_log = [entry for entry in raw_log if isinstance(entry, dict)]
                break

    return GradeContext(
        task_name=task_name,
        score=_extract_score(*args, **kwargs),
        schedule_log=schedule_log,
    )


def _assert_task(expected_task_name: str, context: GradeContext) -> None:
    if context.task_name is not None and context.task_name != expected_task_name:
        raise ValueError(
            f"Grader for {expected_task_name} received payload for {context.task_name}"
        )


def _scheduled_ids(context: GradeContext) -> set[str]:
    return {
        str(entry["candidate_id"])
        for entry in context.schedule_log
        if entry.get("valid") and "candidate_id" in entry
    }


def _valid_ratio(context: GradeContext) -> float:
    if not context.schedule_log:
        return 1.0
    valid = sum(1 for entry in context.schedule_log if entry.get("valid"))
    return valid / len(context.schedule_log)


def _apply_adjustments(base_score: float, adjustments: list[float]) -> float:
    return _normalize_score(base_score + sum(adjustments))


def grade_easy_critical_backend(*args: Any, **kwargs: Any) -> float:
    context = _build_context(*args, **kwargs)
    _assert_task("easy_critical_backend", context)
    scheduled = _scheduled_ids(context)
    adjustments = [
        0.03 if "cand_backend_1" in scheduled else -0.08,
        0.015 if "cand_data_1" in scheduled else 0.0,
        0.015 if "cand_frontend_1" in scheduled else 0.0,
        0.02 if len(scheduled) == 3 else -0.02,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ]
    return _apply_adjustments(context.score, adjustments)


def grade_medium_scarce_ml_specialist(*args: Any, **kwargs: Any) -> float:
    context = _build_context(*args, **kwargs)
    _assert_task("medium_scarce_ml_specialist", context)
    scheduled = _scheduled_ids(context)
    adjustments = [
        0.05 if "cand_ml_urgent" in scheduled else -0.1,
        -0.05 if "cand_ml_nice" in scheduled and "cand_ml_urgent" not in scheduled else 0.0,
        0.025 if "cand_general_backend" in scheduled else -0.015,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ]
    return _apply_adjustments(context.score, adjustments)


def grade_hard_multi_tradeoff(*args: Any, **kwargs: Any) -> float:
    context = _build_context(*args, **kwargs)
    _assert_task("hard_multi_tradeoff", context)
    scheduled = _scheduled_ids(context)
    critical_ids = {"cand_sec_critical", "cand_backend_urgent", "cand_data_urgent"}
    adjustments = [
        0.045 * len(critical_ids & scheduled),
        -0.06 if "cand_sec_critical" not in scheduled else 0.0,
        -0.03 if "cand_sec_fit" in scheduled and "cand_sec_critical" not in scheduled else 0.0,
        0.015 if "cand_frontend_fit" in scheduled else 0.0,
        0.01 if _valid_ratio(context) == 1.0 else -0.04,
    ]
    return _apply_adjustments(context.score, adjustments)


def grade_medium_deadline_pressure(*args: Any, **kwargs: Any) -> float:
    context = _build_context(*args, **kwargs)
    _assert_task("medium_deadline_pressure", context)
    scheduled = _scheduled_ids(context)
    deadline_one_ids = {"cand_be_urgent1", "cand_fe_urgent1"}
    adjustments = [
        0.035 * len(deadline_one_ids & scheduled),
        -0.05 if not deadline_one_ids.issubset(scheduled) else 0.0,
        0.02 if "cand_ml_urgent1" in scheduled else 0.0,
        -0.015 if "cand_data_urgent1" not in scheduled else 0.0,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ]
    return _apply_adjustments(context.score, adjustments)


def grade_hard_conflicting_priorities(*args: Any, **kwargs: Any) -> float:
    context = _build_context(*args, **kwargs)
    _assert_task("hard_conflicting_priorities", context)
    scheduled = _scheduled_ids(context)
    top_security_ids = {"cand_sec_high1", "cand_sec_high2"}
    adjustments = [
        0.05 if scheduled & top_security_ids else -0.1,
        -0.05 if "cand_sec_mid1" in scheduled and not (scheduled & top_security_ids) else 0.0,
        0.02 if "cand_be_conf1" in scheduled else 0.0,
        0.015 if "cand_fe_conf1" in scheduled else 0.0,
        0.01 if _valid_ratio(context) == 1.0 else -0.03,
    ]
    return _apply_adjustments(context.score, adjustments)
