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

from beginner_agent.config import load_project_env  # noqa: E402
from beginner_agent.memory_migrations import (  # noqa: E402
    backfill_memory_governance_fields,
    current_memory_schema_version,
    pending_memory_migrations,
    rollback_memory_migration,
    run_memory_migrations,
)


def _database_url() -> str:
    load_project_env()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL，无法管理 Postgres memory migrations。")
    return database_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage beginner_agent memory migrations.")
    parser.add_argument(
        "command",
        choices=["status", "upgrade", "rollback", "backfill"],
        help="status 查看版本；upgrade 执行迁移；rollback 回滚；backfill 回填历史数据。",
    )
    parser.add_argument("--target-version", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    database_url = _database_url()
    if args.command == "status":
        result = {
            "current_version": current_memory_schema_version(database_url),
            "pending": pending_memory_migrations(database_url),
        }
    elif args.command == "upgrade":
        result = run_memory_migrations(database_url)
    elif args.command == "rollback":
        result = rollback_memory_migration(
            database_url,
            target_version=args.target_version,
        )
    else:
        result = backfill_memory_governance_fields(database_url, limit=args.limit)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
