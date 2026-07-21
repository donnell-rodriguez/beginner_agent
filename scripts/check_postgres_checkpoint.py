from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from beginner_agent.checkpointing import build_checkpointer, checkpoint_backend_name  # noqa: E402


def main() -> None:
    """验证 LangGraph Postgres checkpoint 是否可初始化。"""

    os.environ.setdefault("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    os.environ.setdefault(
        "DATABASE_URL",
        "postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent",
    )
    checkpointer = build_checkpointer()
    backend = checkpoint_backend_name()
    print("Postgres checkpoint check passed.")
    print(f"backend={backend}")
    print(f"checkpointer={type(checkpointer).__name__}")


if __name__ == "__main__":
    main()
