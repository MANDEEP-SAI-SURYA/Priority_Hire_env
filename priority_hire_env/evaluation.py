from __future__ import annotations

from typing import Dict, List

from .baselines import fifo_policy, fit_only_policy, priority_fit_policy, priority_only_policy
from .env import PriorityHireEnv
from .tasks import list_task_names


POLICIES = {
    "fifo": fifo_policy,
    "priority_only": priority_only_policy,
    "fit_only": fit_only_policy,
    "priority_fit": priority_fit_policy,
}


def run_policy(task_name: str, policy_name: str) -> Dict[str, object]:
    env = PriorityHireEnv(task_name=task_name)
    observation = env.reset(task_name=task_name)
    policy = POLICIES[policy_name]
    rewards: List[float] = []
    done = False

    while not done:
        action = policy(observation)
        observation, reward, done, info = env.step(action)
        rewards.append(reward.reward)

    return {
        "task": task_name,
        "policy": policy_name,
        "score": info["score"],
        "steps": len(rewards),
        "reward_sum": round(sum(rewards), 4),
        "done_reason": info["done_reason"],
    }


def run_all_baselines() -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for task_name in list_task_names():
        for policy_name in POLICIES:
            results.append(run_policy(task_name=task_name, policy_name=policy_name))
    return results
