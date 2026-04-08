from __future__ import annotations

from server.environment import MAX_SCORE, MIN_SCORE, list_task_names, run_task_with_policy


def main() -> None:
    for task_name in list_task_names():
        result = run_task_with_policy(task_name)
        assert MIN_SCORE <= float(result["score"]) <= MAX_SCORE
        assert MIN_SCORE <= float(result["grader_score"]) <= MAX_SCORE
        print(task_name, result["score"], result["done_reason"])


if __name__ == "__main__":
    main()
