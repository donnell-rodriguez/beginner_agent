from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from beginner_agent.checkpointing import (  # noqa: E402
    check_checkpoint_health,
    checkpoint_backend_config,
    setup_checkpoint_schema,
)


def main(argv: list[str] | None = None) -> int:
    """Checkpoint schema 运维入口。

    中文注释：
    大厂生产环境通常不会让应用启动自动改数据库表。
    这个脚本就是把 setup/migration 变成显式动作：

    - status：只查看当前 schema 状态，不改数据库。
    - setup：执行 LangGraph PostgresSaver.setup()。

    本地仍然可以使用自动 setup；
    生产可以设置 BEGINNER_AGENT_CHECKPOINT_AUTO_SETUP=false，
    然后由 CI/CD 或运维任务调用这个脚本。
    """

    parser = argparse.ArgumentParser(description="Manage beginner_agent checkpoint schema.")
    parser.add_argument("command", choices=["status", "setup"])
    parser.add_argument(
        "--database-url",
        default=None,
        help="临时覆盖 BEGINNER_AGENT_CHECKPOINT_DATABASE_URL。",
    )
    args = parser.parse_args(argv)

    if args.database_url:
        os.environ["BEGINNER_AGENT_CHECKPOINT_DATABASE_URL"] = args.database_url
    os.environ.setdefault("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")

    if args.command == "status":
        config = checkpoint_backend_config()
        health = check_checkpoint_health()
        payload = {
            "backend": config.effective_backend,
            "requested_backend": config.requested_backend,
            "setup_mode": config.setup_mode,
            "auto_setup_enabled": config.auto_setup_enabled,
            "health": health.model_dump(mode="json"),
        }
    else:
        payload = setup_checkpoint_schema().model_dump(mode="json")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
