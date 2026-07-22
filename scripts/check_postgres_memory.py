from __future__ import annotations

import os
import sys
from pathlib import Path


# 中文注释：
# 这个脚本可能从 beginner_agent 仓库根目录运行。
# 为了能 import beginner_agent.memory，需要把仓库父目录加入 sys.path。
PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from beginner_agent.memory.models import MemoryRecord  # noqa: E402
from beginner_agent.memory.postgres_store import PostgresMemoryStore  # noqa: E402


def main() -> None:
    """验证 Docker Postgres memory 后端是否真的可写、可读。"""

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent",
    )
    store = PostgresMemoryStore(database_url)
    record = MemoryRecord(
        id="postgres-smoke-test",
        kind="eval",
        task_id="smoke-test",
        title="Postgres memory smoke test",
        summary="Verify beginner_agent_memory upsert/list works against local Docker Postgres.",
        status="done",
        tool_name="check_postgres_memory",
        tool_result_status="success",
        paths=["memory.py"],
        tags=["postgres", "memory", "smoke-test"],
        confidence=0.99,
        metadata={"database_url_host": "127.0.0.1:55432"},
    )
    store.upsert_record(record)
    records = store.list_records(20)
    matched = [item for item in records if item.get("id") == record.id]
    if not matched:
        raise SystemExit("Postgres memory check failed: inserted record was not returned.")
    print("Postgres memory check passed.")
    print(f"backend=postgres records_checked={len(records)} matched_id={matched[0]['id']}")


if __name__ == "__main__":
    main()
