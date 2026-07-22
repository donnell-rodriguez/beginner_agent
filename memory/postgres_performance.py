from __future__ import annotations

from typing import Any

from .migrations import current_memory_schema_version, pending_memory_migrations


# 中文注释：
# postgres_performance.py 负责 Postgres 查询性能和迁移治理体检。
#
# 大厂生产系统不会只关心“SQL 能不能跑”。
# 还会持续检查：
# - schema version 是否落后。
# - 关键索引是否存在。
# - 表规模是否异常。
# - embedding 表是否存在。
# - 是否有 pending migration。


REQUIRED_MEMORY_INDEXES = {
    "idx_beginner_agent_memory_kind",
    "idx_beginner_agent_memory_task_id",
    "idx_beginner_agent_memory_tool_status",
    "idx_beginner_agent_memory_created_at",
    "idx_beginner_agent_memory_governance",
    "idx_beginner_agent_memory_acl",
    "idx_beginner_agent_memory_tags",
    "idx_beginner_agent_memory_audit_memory_id",
    "idx_beginner_agent_memory_audit_created_at",
}


def memory_postgres_governance_report(database_url: str) -> dict[str, Any]:
    """生成 Postgres memory 治理报告。"""

    import psycopg

    with psycopg.connect(database_url) as conn:
        index_rows = conn.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND (
                tablename = 'beginner_agent_memory'
                OR tablename = 'beginner_agent_memory_audit'
              )
            """
        ).fetchall()
        indexes = {str(row[0]) for row in index_rows}
        embedding_rows = conn.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE 'beginner_agent_memory_embeddings_%'
            ORDER BY tablename
            """
        ).fetchall()
        counts = _table_counts(conn)

    missing_indexes = sorted(REQUIRED_MEMORY_INDEXES - indexes)
    pending = pending_memory_migrations(database_url)
    return {
        "current_version": current_memory_schema_version(database_url),
        "pending_migrations": pending,
        "pending_migration_count": len(pending),
        "index_count": len(indexes),
        "missing_required_indexes": missing_indexes,
        "embedding_tables": [str(row[0]) for row in embedding_rows],
        "table_counts": counts,
        "healthy": not pending and not missing_indexes,
        "recommendations": _recommendations(pending, missing_indexes, counts),
    }


def _table_counts(conn: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in ("beginner_agent_memory", "beginner_agent_memory_audit"):
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        except Exception:
            counts[table] = -1
            continue
        counts[table] = int(row[0]) if row else 0
    return counts


def _recommendations(
    pending: list[dict[str, Any]],
    missing_indexes: list[str],
    counts: dict[str, int],
) -> list[str]:
    recommendations: list[str] = []
    if pending:
        recommendations.append("先运行 scripts/manage_memory_migrations.py upgrade。")
    if missing_indexes:
        recommendations.append("补齐缺失索引，否则 retrieval / API 查询会变慢。")
    if counts.get("beginner_agent_memory", 0) > 100000:
        recommendations.append("memory 表已较大，建议增加分区、归档或专门向量库。")
    if counts.get("beginner_agent_memory_audit", 0) > 500000:
        recommendations.append("audit 表已较大，建议迁移到日志/数据仓库并做冷热分层。")
    return recommendations
