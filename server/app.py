from __future__ import annotations

import os

import uvicorn

from priority_hire_env.server import app


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
