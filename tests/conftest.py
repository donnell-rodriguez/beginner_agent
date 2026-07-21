from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


@pytest.fixture(autouse=True)
def stable_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """给测试提供稳定的本地默认环境。

    中文注释：
    测试不要依赖用户 shell 里偶然存在的 env。
    这里把危险/外部依赖默认关掉，需要 Postgres 的测试单独打开。
    """

    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_BACKEND", "jsonl")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_HISTORY_BACKEND", "jsonl")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_RUN_COMPACTION", "false")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_REBUILD_EMBEDDINGS", "false")
    monkeypatch.setenv("BEGINNER_AGENT_PRIVACY_HASH_SALT", "test-salt")


@pytest.fixture
def postgres_database_url() -> str:
    """Postgres fixture。

    中文注释：
    默认 pytest 不跑真实 Postgres。
    只有显式设置 BEGINNER_AGENT_RUN_POSTGRES_TESTS=true 时才跑。
    这样普通 unit test 不会因为本地 Docker、网络沙箱或数据库权限失败。
    """

    if os.getenv("BEGINNER_AGENT_RUN_POSTGRES_TESTS", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        pytest.skip("Set BEGINNER_AGENT_RUN_POSTGRES_TESTS=true to run Postgres tests.")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("DATABASE_URL is not set; skipping Postgres integration test.")
    return database_url


class FakeMemoryStore:
    """测试用内存版 MemoryStore。

    中文注释：
    它只模拟 memory store 协议，不碰真实数据库/文件。
    用来测试 compaction、retrieval 这类业务逻辑。
    """

    backend_name = "fake"

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records or []
        self.upserts: list[Any] = []
        self.audit_events: list[Any] = []
        self.status_updates: list[dict[str, Any]] = []

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        return self.records[:limit]

    def search_similar_records(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        return self.records[:limit]

    def upsert_record(self, record: Any) -> None:
        self.upserts.append(record)

    def mark_records_status(
        self,
        memory_ids: list[str],
        status: str,
        *,
        superseded_by: str | None = None,
    ) -> None:
        self.status_updates.append(
            {
                "memory_ids": memory_ids,
                "status": status,
                "superseded_by": superseded_by,
            }
        )

    def cleanup_expired_records(self) -> int:
        return 0

    def rebuild_embeddings(self, limit: int) -> int:
        return 0

    def upsert_audit_event(self, event: Any) -> None:
        self.audit_events.append(event)


@pytest.fixture
def fake_memory_store() -> FakeMemoryStore:
    return FakeMemoryStore()
