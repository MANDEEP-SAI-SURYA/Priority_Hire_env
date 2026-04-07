from __future__ import annotations

from priority_hire_env.baselines import priority_fit_policy
from priority_hire_env.env import PriorityHireEnv
from priority_hire_env.tasks import list_task_names


def main() -> None:
    for task_name in list_task_names():
        env = PriorityHireEnv(task_name=task_name)
        observation = env.reset(task_name=task_name)
        done = False
        while not done:
            action = priority_fit_policy(observation)
            observation, reward, done, info = env.step(action)
            assert 0.0 <= float(info["score"]) <= 1.0
            assert reward.normalized_score is None or 0.0 <= reward.normalized_score <= 1.0
        print(task_name, info["score"], info["done_reason"])


if __name__ == "__main__":
    main()
