from __future__ import annotations

import json

from priority_hire_env.evaluation import run_all_baselines


if __name__ == "__main__":
    print(json.dumps(run_all_baselines(), indent=2))
