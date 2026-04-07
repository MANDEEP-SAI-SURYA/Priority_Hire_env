from __future__ import annotations

from typing import Any

from .env import PriorityHireEnv


MIN_SCORE = 0.1001
MAX_SCORE = 0.9899


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

        info = getattr(item, "info", None)
        if isinstance(info, dict) and "score" in info:
            return _normalize_score(info["score"])

    for item in candidates:
        if isinstance(item, dict):
            info = item.get("info")
            if isinstance(info, dict) and "score" in info:
                return _normalize_score(info["score"])
            if "score" in item:
                return _normalize_score(item["score"])
            reward = item.get("reward")
            if reward is not None:
                nested_score = getattr(reward, "normalized_score", None)
                if nested_score is not None:
                    return _normalize_score(nested_score)
                if isinstance(reward, dict) and reward.get("normalized_score") is not None:
                    return _normalize_score(reward["normalized_score"])

    score = kwargs.get("score")
    if score is not None:
        return _normalize_score(score)

    raise ValueError("Unable to extract score for grader invocation")


def _grade_task(expected_task_name: str, *args: Any, **kwargs: Any) -> float:
    candidates = list(args) + list(kwargs.values())

    for item in candidates:
        task_name = _extract_task_name(item)
        if task_name is not None and task_name != expected_task_name:
            raise ValueError(
                f"Grader for {expected_task_name} received payload for {task_name}"
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
