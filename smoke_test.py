from __future__ import annotations

from priority_hire_env.baselines import priority_fit_policy
from priority_hire_env.env import PriorityHireEnv
from priority_hire_env.graders import (
    grade_easy_critical_backend,
    grade_hard_conflicting_priorities,
    grade_hard_multi_tradeoff,
    grade_medium_deadline_pressure,
    grade_medium_scarce_ml_specialist,
)
from priority_hire_env.tasks import list_task_names


GRADERS = {
    "easy_critical_backend": grade_easy_critical_backend,
    "medium_scarce_ml_specialist": grade_medium_scarce_ml_specialist,
    "hard_multi_tradeoff": grade_hard_multi_tradeoff,
    "medium_deadline_pressure": grade_medium_deadline_pressure,
    "hard_conflicting_priorities": grade_hard_conflicting_priorities,
}
MIN_SCORE = 0.1001
MAX_SCORE = 0.9899


def main() -> None:
    for task_name in list_task_names():
        env = PriorityHireEnv(task_name=task_name)
        observation = env.reset(task_name=task_name)
        done = False
        while not done:
            action = priority_fit_policy(observation)
            observation, reward, done, info = env.step(action)
            assert MIN_SCORE <= float(info["score"]) <= MAX_SCORE
            assert reward.normalized_score is None or MIN_SCORE <= reward.normalized_score <= MAX_SCORE
        grader_score = GRADERS[task_name](info)
        assert MIN_SCORE <= grader_score <= MAX_SCORE
        print(task_name, info["score"], info["done_reason"])


if __name__ == "__main__":
    main()
