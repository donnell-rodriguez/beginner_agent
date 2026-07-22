from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from beginner_agent.memory.models import MemoryRecord  # noqa: E402
from beginner_agent.memory.postgres_store import PostgresMemoryStore  # noqa: E402


def main() -> None:
    """验证 Postgres + pgvector memory 是否可写、可向量检索。"""

    os.environ.setdefault("BEGINNER_AGENT_EMBEDDING_PROVIDER", "omlx")
    os.environ.setdefault("BEGINNER_AGENT_EMBEDDING_DIM", "1024")
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent",
    )
    store = PostgresMemoryStore(database_url)
    record = MemoryRecord(
        id="pgvector-smoke-test",
        kind="project",
        task_id="vector-smoke-test",
        title="理解 memory.py 的 pgvector 检索",
        summary="Verify beginner_agent_memory_embeddings can store and retrieve vector records.",
        status="done",
        tool_name="check_vector_memory",
        tool_result_status="success",
        paths=["memory.py", "embeddings.py"],
        tags=["postgres", "pgvector", "memory", "embedding"],
        confidence=0.99,
        metadata={
            "requested_embedding_provider": os.getenv(
                "BEGINNER_AGENT_EMBEDDING_PROVIDER", "omlx"
            ),
            "requested_embedding_dim": os.getenv("BEGINNER_AGENT_EMBEDDING_DIM", "1024"),
        },
    )
    store.upsert_record(record)
    matches = store.search_similar_records("memory.py pgvector embedding retrieval", 5)
    matched = [item for item in matches if item.get("id") == record.id]
    if not matched:
        raise SystemExit("Vector memory check failed: inserted vector record was not returned.")
    print("Vector memory check passed.")
    print(
        "backend=postgres+pgvector "
        f"records_checked={len(matches)} matched_id={matched[0]['id']} "
        f"distance={matched[0].get('vector_distance')} "
        f"actual_provider={matched[0].get('embedding_provider')} "
        f"actual_model={matched[0].get('embedding_model')}"
    )


if __name__ == "__main__":
    main()
