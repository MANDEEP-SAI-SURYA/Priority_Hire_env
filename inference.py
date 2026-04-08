import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import PriorityHireAction, PriorityHireEnv
from server.environment import priority_fit_policy

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("PRIORITY_HIRE_TASK", "easy_critical_backend")
BENCHMARK = os.getenv("PRIORITY_HIRE_BENCHMARK", "priority_hire")
MAX_STEPS = 16
TEMPERATURE = 0.2
MAX_TOKENS = 220
SUCCESS_SCORE_THRESHOLD = 0.65

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are scheduling interviews in PriorityHireEnv.
    Return exactly one JSON action object.

    Valid actions:
    {"kind":"schedule","candidate_id":"...","interviewer_id":"...","slot_id":"..."}
    {"kind":"defer","candidate_id":"..."}
    {"kind":"submit"}

    Prioritize urgent critical roles, specialization correctness, scarce specialist preservation,
    deadline risk reduction, and fit quality.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(state: dict) -> str:
    return textwrap.dedent(
        f"""
        Current state:
        {json.dumps(state, indent=2)}

        Return one valid JSON action only.
        """
    ).strip()


def choose_action_with_model(client: OpenAI, observation_dict: dict) -> Optional[dict]:
    if not API_KEY:
        return None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation_dict)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = (completion.choices[0].message.content or "").strip()
        return json.loads(content)
    except Exception:
        return None


def action_to_log_string(action) -> str:
    if getattr(action, "kind", None) == "schedule":
        return f"schedule({action.candidate_id},{action.interviewer_id},{action.slot_id})"
    if getattr(action, "kind", None) == "defer":
        return f"defer({action.candidate_id})"
    return "submit()"


def coerce_action(model_action: Optional[dict], fallback_action):
    if not model_action or not isinstance(model_action, dict):
        return fallback_action
    kind = model_action.get("kind")
    if kind == "schedule" and {"candidate_id", "interviewer_id", "slot_id"} <= set(model_action):
        return PriorityHireAction(
            kind="schedule",
            candidate_id=model_action.get("candidate_id"),
            interviewer_id=model_action.get("interviewer_id"),
            slot_id=model_action.get("slot_id"),
        )
    if kind == "defer" and "candidate_id" in model_action:
        return PriorityHireAction(kind="defer", candidate_id=model_action.get("candidate_id"))
    if kind == "submit":
        return PriorityHireAction(kind="submit")
    return fallback_action


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-token")
    if not IMAGE_NAME:
        raise RuntimeError("Missing LOCAL_IMAGE_NAME or IMAGE_NAME for from_docker_image()")

    env = await PriorityHireEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        transition = await env.reset(task_name=TASK_NAME)
        observation = transition.observation
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            fallback_action = priority_fit_policy(observation)
            model_action = choose_action_with_model(client, observation.model_dump())
            action = coerce_action(model_action, fallback_action)

            transition = await env.step(action)
            observation = transition.observation
            reward = float(transition.reward or 0.0)
            done = transition.done
            error = observation.last_action_error

            rewards.append(reward)
            steps_taken = step
            score = float(getattr(observation, "score", 0.0))

            log_step(step=step, action=action_to_log_string(action), reward=reward, done=done, error=error)

            if done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
