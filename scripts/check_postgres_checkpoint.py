from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from beginner_agent.checkpointing import (  # noqa: E402
    build_checkpointer,
    checkpoint_backend_config,
    checkpoint_backend_name,
)


def main() -> None:
    """验证 LangGraph Postgres checkpoint 是否可初始化。"""

    os.environ.setdefault("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    checkpointer = build_checkpointer()
    config = checkpoint_backend_config()
    backend = checkpoint_backend_name()
    print("Postgres checkpoint check passed.")
    print(f"backend={backend}")
    print(f"setup_mode={config.setup_mode}")
    print(f"checkpointer={type(checkpointer).__name__}")


if __name__ == "__main__":
    main()
