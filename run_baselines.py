from __future__ import annotations

import json

from server.environment import list_task_names, run_task_with_policy


if __name__ == "__main__":
    print(json.dumps([run_task_with_policy(task_name) for task_name in list_task_names()], indent=2))
