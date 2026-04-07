from __future__ import annotations

from typing import Any

from .env import PriorityHireEnv


def _extract_score(*args: Any, **kwargs: Any) -> float:
    candidates = list(args) + list(kwargs.values())

    for item in candidates:
        if isinstance(item, PriorityHireEnv):
            return float(item.compute_score())

    for item in candidates:
        if isinstance(item, dict):
            info = item.get("info")
            if isinstance(info, dict) and "score" in info:
                return float(info["score"])
            if "score" in item:
                return float(item["score"])

    score = kwargs.get("score")
    if score is not None:
        return float(score)

    raise ValueError("Unable to extract score for grader invocation")


def _grade_task(expected_task_name: str, *args: Any, **kwargs: Any) -> float:
    candidates = list(args) + list(kwargs.values())

    for item in candidates:
        if isinstance(item, PriorityHireEnv) and item.task_name != expected_task_name:
            raise ValueError(
                f"Grader for {expected_task_name} received env for {item.task_name}"
            )

    return _extract_score(*args, **kwargs)


def grade_easy_critical_backend(*args: Any, **kwargs: Any) -> float:
    return _grade_task("easy_critical_backend", *args, **kwargs)


def grade_medium_scarce_ml_specialist(*args: Any, **kwargs: Any) -> float:
    return _grade_task("medium_scarce_ml_specialist", *args, **kwargs)


def grade_hard_multi_tradeoff(*args: Any, **kwargs: Any) -> float:
    return _grade_task("hard_multi_tradeoff", *args, **kwargs)


def grade_medium_deadline_pressure(*args: Any, **kwargs: Any) -> float:
    return _grade_task("medium_deadline_pressure", *args, **kwargs)


def grade_hard_conflicting_priorities(*args: Any, **kwargs: Any) -> float:
    return _grade_task("hard_conflicting_priorities", *args, **kwargs)
