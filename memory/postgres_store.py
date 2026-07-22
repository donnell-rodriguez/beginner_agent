from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from ..embeddings import safe_embedding, vector_to_sql
from .migrations import pending_memory_migrations, run_memory_migrations
from .models import MemoryAuditEvent, MemoryRecord, ValidityStatus
from .settings import MAX_INDEXED_VECTOR_DIMENSION, MAX_MEMORY_TEXT_CHARS

def _validate_embedding_dimension(dimension: int) -> None:
    """校验当前 pgvector 索引使用的向量维度。

    中文注释：
    Qwen3-Embedding-8B 可以输出很高维度的向量。
    但是对 Postgres + pgvector 来说，本地工程更推荐使用 1024 维。

    原因：
    - 1024 维已经足够做 memory / code 检索入门。
    - 向量更短，写入和查询更快。
    - pgvector 常规索引更适合 2000 维以内的 vector。

    如果你真的要用 4096 维，后续应该单独设计 halfvec / binary quantization /
    专用向量数据库，而不是直接塞进当前教学项目的索引表。
    """

    if dimension <= 0:
        raise ValueError("embedding 维度必须大于 0。")
    if dimension > MAX_INDEXED_VECTOR_DIMENSION:
        raise ValueError(
            "当前 pgvector 索引表最多支持 "
            f"{MAX_INDEXED_VECTOR_DIMENSION} 维。"
            "Qwen3-Embedding-8B 请先设置 BEGINNER_AGENT_EMBEDDING_DIM=1024。"
        )


def _embedding_table_name(dimension: int) -> str:
    """根据维度生成安全的 embedding 表名。"""

    _validate_embedding_dimension(dimension)
    return f"beginner_agent_memory_embeddings_{dimension}"

def _embedding_text_for_record(record: MemoryRecord) -> str:
    """构造用于 embedding 的短文本。

    中文注释：
    不要把完整 tool output 或大文件内容塞进 embedding。
    memory embedding 应该索引“经验摘要”，而不是索引所有原始数据。
    """

    failure_profile = record.metadata.get("failure_memory")
    if not isinstance(failure_profile, dict):
        failure_profile = {}
    preference = record.metadata.get("preference_memory")
    if not isinstance(preference, dict):
        preference = {}
    return "\n".join(
        [
            f"kind: {record.kind}",
            f"title: {record.title}",
            f"summary: {record.summary}",
            f"status: {record.status}",
            f"tool: {record.tool_name}",
            f"tool_result_status: {record.tool_result_status}",
            f"scope: {record.scope}",
            f"visibility: {record.visibility}",
            f"sensitivity_level: {record.sensitivity_level}",
            f"tenant_id: {record.tenant_id}",
            f"workspace_id: {record.workspace_id}",
            f"project_id: {record.project_id}",
            f"retention_policy: {record.retention_policy}",
            f"validity_status: {record.validity_status}",
            f"importance: {record.importance}",
            f"quality_score: {record.quality_score}",
            f"trust_score: {record.trust_score}",
            f"decay_score: {record.decay_score}",
            f"failure_category: {failure_profile.get('category', '')}",
            f"failure_owner: {failure_profile.get('owner', '')}",
            f"failure_retry_class: {failure_profile.get('retry_class', '')}",
            f"failure_pattern_id: {failure_profile.get('pattern_id', '')}",
            f"preference_key: {preference.get('key', '')}",
            f"preference_value: {preference.get('value', '')}",
            f"preference_scope: {preference.get('scope', '')}",
            f"preference_category: {preference.get('category', '')}",
            f"paths: {', '.join(record.paths)}",
            f"tags: {', '.join(record.tags)}",
        ]
    )[:2000]

class PostgresMemoryStore:
    """Postgres memory store。

    中文注释：
    启用方式：

        BEGINNER_AGENT_MEMORY_BACKEND=postgres
        DATABASE_URL=postgresql://...

    这里使用延迟 import psycopg。
    如果你的环境没有安装 psycopg，默认 JSONL 路径不受影响。

    小白理解：
    这个类负责把 agent 的记忆写进 Postgres。
    它会维护两张表：

    1. beginner_agent_memory
       保存普通结构化字段。
       例如 kind、task_id、title、summary、tool_name、status。

    2. beginner_agent_memory_embeddings_<dimension>
       保存向量字段。
       例如 beginner_agent_memory_embeddings_1024 保存 vector(1024)。

    为什么要两张表？
    因为普通字段适合做精确过滤：

        找 failure 类型
        找 read_file 工具
        找 memory.py 相关记录

    向量字段适合做语义相似搜索：

        当前任务和过去哪个经验意思相近？
        这次报错和历史哪次失败相似？

    真实 agent 通常会把两者结合起来，也就是 hybrid retrieval。
    """

    backend_name = "postgres"

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url)

    def _ensure_table(self) -> None:
        """确保 Postgres memory schema 已经迁移到最新版本。

        中文注释：
        这里不再手写 CREATE TABLE / ALTER TABLE。
        schema 变更统一放到 memory_migrations.py：
        - 有版本号。
        - 有 up/down。
        - 有 checksum。
        - 有 backfill job。

        这样 PostgresMemoryStore 只负责读写数据，不负责设计数据库历史。

        生产级提醒：
        - 本地开发可以自动 upgrade，体验更顺。
        - 线上服务通常会关闭自动 upgrade，改由发布流水线执行 migration。
        - 如果关闭后还有 pending migration，这里会直接报错，避免业务进程偷偷改表。
        """

        auto_upgrade = (
            os.getenv("BEGINNER_AGENT_MEMORY_MIGRATION_AUTO_UPGRADE", "true")
            .strip()
            .lower()
        )
        if auto_upgrade in {"1", "true", "yes", "on"}:
            run_memory_migrations(self.database_url)
            return

        pending = pending_memory_migrations(self.database_url)
        if pending:
            versions = ", ".join(str(item["version"]) for item in pending)
            raise RuntimeError(
                "Postgres memory schema has pending migrations. "
                f"versions={versions}. "
                "请先运行 scripts/manage_memory_migrations.py upgrade，"
                "或在本地开发时设置 BEGINNER_AGENT_MEMORY_MIGRATION_AUTO_UPGRADE=true。"
            )

    def _ensure_embedding_table(self, dimension: int) -> str:
        """确保当前 embedding 维度对应的 pgvector 表存在。

        中文注释：
        pgvector 的 vector 列通常要固定维度，例如 vector(384)、vector(1024)。
        之前本项目只有单一固定维度表。

        现在你要接 Qwen3-Embedding-8B。
        推荐让它输出 1024 维向量，因此这里改成“按维度分表”：

            beginner_agent_memory_embeddings_384
            beginner_agent_memory_embeddings_1024

        这样做的好处：
        - 旧的 384 维测试数据不会影响新的 1024 维 Qwen 数据。
        - 每张表的 pgvector 列维度固定，索引更清楚。
        - 以后换 1536 / 2000 维，也可以并存。

        注意：
        pgvector 的常规 vector 索引更适合 2000 维以内。
        Qwen3-Embedding-8B 原生可以到 4096 维，但本项目建议请求 1024 维。
        """

        _validate_embedding_dimension(dimension)
        table_name = _embedding_table_name(dimension)
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL REFERENCES beginner_agent_memory(id) ON DELETE CASCADE,
                    embedding_model TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding VECTOR({dimension}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_memory_id
                ON {table_name} (memory_id)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_model
                ON {table_name} (embedding_provider, embedding_model)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_vector
                ON {table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 20)
                """
            )
        return table_name

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        self._ensure_table()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, task_id, title, summary, status, tool_name,
                       tool_result_status, paths, tags, confidence, source,
                       created_at::text, metadata, importance, quality_score,
                       trust_score, decay_score, scope, visibility,
                       sensitivity_level, tenant_id, workspace_id, project_id,
                       user_id, retention_policy, validity_status, pinned,
                       expires_at::text, supersedes, contradiction_key
                FROM beginner_agent_memory
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "id": row[0],
                    "kind": row[1],
                    "task_id": row[2],
                    "title": row[3],
                    "summary": row[4],
                    "status": row[5],
                    "tool_name": row[6],
                    "tool_result_status": row[7],
                    "paths": row[8],
                    "tags": row[9],
                    "confidence": row[10],
                    "source": row[11],
                    "created_at": row[12],
                    "metadata": row[13],
                    "importance": row[14],
                    "quality_score": row[15],
                    "trust_score": row[16],
                    "decay_score": row[17],
                    "scope": row[18],
                    "visibility": row[19],
                    "sensitivity_level": row[20],
                    "tenant_id": row[21],
                    "workspace_id": row[22],
                    "project_id": row[23],
                    "user_id": row[24],
                    "retention_policy": row[25],
                    "validity_status": row[26],
                    "pinned": row[27],
                    "expires_at": row[28],
                    "supersedes": row[29],
                    "contradiction_key": row[30],
                }
            )
        return records

    def upsert_audit_event(self, event: MemoryAuditEvent) -> None:
        """写入 memory audit event。"""

        self._ensure_table()
        data = event.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO beginner_agent_memory_audit (
                    id, action, memory_id, reason, backend, created_at, metadata
                )
                VALUES (
                    %(id)s, %(action)s, %(memory_id)s, %(reason)s,
                    %(backend)s, %(created_at)s::timestamptz, %(metadata)s::jsonb
                )
                ON CONFLICT (id) DO UPDATE SET
                    action = EXCLUDED.action,
                    memory_id = EXCLUDED.memory_id,
                    reason = EXCLUDED.reason,
                    backend = EXCLUDED.backend,
                    created_at = EXCLUDED.created_at,
                    metadata = EXCLUDED.metadata
                """,
                {
                    **data,
                    "metadata": json.dumps(data["metadata"], ensure_ascii=False),
                },
            )

    def upsert_record(self, record: MemoryRecord) -> None:
        self._ensure_table()
        data = record.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO beginner_agent_memory (
                    id, kind, task_id, title, summary, status, tool_name,
                    tool_result_status, paths, tags, confidence, importance,
                    quality_score, trust_score, decay_score, scope, visibility,
                    sensitivity_level, tenant_id, workspace_id, project_id, user_id,
                    retention_policy, validity_status, pinned, expires_at, supersedes,
                    contradiction_key, source, created_at, metadata
                )
                VALUES (
                    %(id)s, %(kind)s, %(task_id)s, %(title)s, %(summary)s,
                    %(status)s, %(tool_name)s, %(tool_result_status)s,
                    %(paths)s::jsonb, %(tags)s::jsonb, %(confidence)s,
                    %(importance)s, %(quality_score)s, %(trust_score)s,
                    %(decay_score)s, %(scope)s, %(visibility)s, %(sensitivity_level)s,
                    %(tenant_id)s, %(workspace_id)s, %(project_id)s, %(user_id)s,
                    %(retention_policy)s, %(validity_status)s, %(pinned)s,
                    %(expires_at)s::timestamptz, %(supersedes)s, %(contradiction_key)s,
                    %(source)s, %(created_at)s::timestamptz, %(metadata)s::jsonb
                )
                ON CONFLICT (id) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    task_id = EXCLUDED.task_id,
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    status = EXCLUDED.status,
                    tool_name = EXCLUDED.tool_name,
                    tool_result_status = EXCLUDED.tool_result_status,
                    paths = EXCLUDED.paths,
                    tags = EXCLUDED.tags,
                    confidence = EXCLUDED.confidence,
                    importance = EXCLUDED.importance,
                    quality_score = EXCLUDED.quality_score,
                    trust_score = EXCLUDED.trust_score,
                    decay_score = EXCLUDED.decay_score,
                    scope = EXCLUDED.scope,
                    visibility = EXCLUDED.visibility,
                    sensitivity_level = EXCLUDED.sensitivity_level,
                    tenant_id = EXCLUDED.tenant_id,
                    workspace_id = EXCLUDED.workspace_id,
                    project_id = EXCLUDED.project_id,
                    user_id = EXCLUDED.user_id,
                    retention_policy = EXCLUDED.retention_policy,
                    validity_status = EXCLUDED.validity_status,
                    pinned = EXCLUDED.pinned,
                    expires_at = EXCLUDED.expires_at,
                    supersedes = EXCLUDED.supersedes,
                    contradiction_key = EXCLUDED.contradiction_key,
                    source = EXCLUDED.source,
                    created_at = EXCLUDED.created_at,
                    metadata = EXCLUDED.metadata
                """,
                {
                    **data,
                    "paths": json.dumps(data["paths"], ensure_ascii=False),
                    "tags": json.dumps(data["tags"], ensure_ascii=False),
                    "metadata": json.dumps(data["metadata"], ensure_ascii=False),
                },
            )
            if record.supersedes:
                conn.execute(
                    """
                    UPDATE beginner_agent_memory
                    SET validity_status = 'superseded'
                    WHERE id = %s AND id <> %s
                    """,
                    (record.supersedes, record.id),
                )
            if record.contradiction_key:
                conn.execute(
                    """
                    UPDATE beginner_agent_memory
                    SET validity_status = 'superseded'
                    WHERE contradiction_key = %s
                      AND id <> %s
                      AND validity_status = 'active'
                    """,
                    (record.contradiction_key, record.id),
                )
        self.upsert_embedding(record)

    def mark_records_status(
        self,
        memory_ids: list[str],
        status: ValidityStatus,
        *,
        superseded_by: str | None = None,
    ) -> None:
        """批量更新 Postgres 里的 memory validity_status。"""

        if not memory_ids:
            return
        self._ensure_table()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE beginner_agent_memory
                SET validity_status = %(status)s,
                    metadata = jsonb_set(
                        metadata,
                        '{compaction}',
                        COALESCE(metadata->'compaction', '{}'::jsonb)
                        || %(compaction)s::jsonb,
                        true
                    )
                WHERE id = ANY(%(memory_ids)s)
                """,
                {
                    "status": status,
                    "memory_ids": memory_ids,
                    "compaction": json.dumps(
                        {
                            "superseded_by": superseded_by,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                        ensure_ascii=False,
                    ),
                },
            )

    def cleanup_expired_records(self) -> int:
        """清理 Postgres 里的过期非 pinned 记忆。"""

        self._ensure_table()
        with self._connect() as conn:
            result = conn.execute(
                """
                DELETE FROM beginner_agent_memory
                WHERE expires_at IS NOT NULL
                  AND expires_at <= NOW()
                  AND pinned = FALSE
                """
            )
            return int(result.rowcount or 0)

    def rebuild_embeddings(self, limit: int) -> int:
        """为最近 active memory 重建 embedding。

        中文注释：
        embedding 模型升级、维度变化、索引损坏后，
        生命周期任务可以定期重建向量，而不是等检索时才发现问题。
        """

        records = self.list_records(limit)
        rebuilt = 0
        for record in records:
            if str(record.get("validity_status", "active")) != "active":
                continue
            try:
                self.upsert_embedding(MemoryRecord(**record))
            except Exception:
                continue
            rebuilt += 1
        return rebuilt

    def upsert_embedding(self, record: MemoryRecord) -> None:
        """为 MemoryRecord 写入 pgvector embedding。"""

        embedding_text = _embedding_text_for_record(record)
        vector, provider_name, model_name, dimension = safe_embedding(embedding_text)
        table_name = self._ensure_embedding_table(dimension)
        embedding_id = f"{record.id}:{provider_name}:{model_name}:0"
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {table_name} (
                    id, memory_id, embedding_model, embedding_provider,
                    embedding_dim, text, embedding, created_at
                )
                VALUES (
                    %(id)s, %(memory_id)s, %(embedding_model)s,
                    %(embedding_provider)s, %(embedding_dim)s, %(text)s,
                    %(embedding)s::vector, %(created_at)s::timestamptz
                )
                ON CONFLICT (id) DO UPDATE SET
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_provider = EXCLUDED.embedding_provider,
                    embedding_dim = EXCLUDED.embedding_dim,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at
                """,
                {
                    "id": embedding_id,
                    "memory_id": record.id,
                    "embedding_model": model_name,
                    "embedding_provider": provider_name,
                    "embedding_dim": dimension,
                    "text": embedding_text,
                    "embedding": vector_to_sql(vector),
                    "created_at": record.created_at,
                },
            )

    def search_similar_records(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        """用 pgvector 查询语义相近的 memory records。"""

        self._ensure_table()
        vector, provider_name, model_name, dimension = safe_embedding(query_text)
        table_name = self._ensure_embedding_table(dimension)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.id, m.kind, m.task_id, m.title, m.summary, m.status,
                       m.tool_name, m.tool_result_status, m.paths, m.tags,
                       m.confidence, m.source, m.created_at::text, m.metadata,
                       m.importance, m.quality_score, m.trust_score,
                       m.decay_score, m.scope, m.visibility,
                       m.sensitivity_level, m.tenant_id, m.workspace_id,
                       m.project_id, m.user_id, m.retention_policy,
                       m.validity_status, m.pinned, m.expires_at::text,
                       m.supersedes, m.contradiction_key,
                       e.embedding <=> %(query_embedding)s::vector AS distance,
                       e.embedding_provider, e.embedding_model
                FROM {table_name} e
                JOIN beginner_agent_memory m ON m.id = e.memory_id
                WHERE e.embedding_provider = %(provider)s
                  AND e.embedding_model = %(model)s
                  AND m.validity_status = 'active'
                  AND (
                    m.expires_at IS NULL
                    OR m.expires_at > NOW()
                    OR m.pinned = TRUE
                  )
                ORDER BY e.embedding <=> %(query_embedding)s::vector
                LIMIT %(limit)s
                """,
                {
                    "query_embedding": vector_to_sql(vector),
                    "provider": provider_name,
                    "model": model_name,
                    "limit": limit,
                },
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "id": row[0],
                    "kind": row[1],
                    "task_id": row[2],
                    "title": row[3],
                    "summary": row[4],
                    "status": row[5],
                    "tool_name": row[6],
                    "tool_result_status": row[7],
                    "paths": row[8],
                    "tags": row[9],
                    "confidence": row[10],
                    "source": row[11],
                    "created_at": row[12],
                    "metadata": row[13],
                    "importance": row[14],
                    "quality_score": row[15],
                    "trust_score": row[16],
                    "decay_score": row[17],
                    "scope": row[18],
                    "visibility": row[19],
                    "sensitivity_level": row[20],
                    "tenant_id": row[21],
                    "workspace_id": row[22],
                    "project_id": row[23],
                    "user_id": row[24],
                    "retention_policy": row[25],
                    "validity_status": row[26],
                    "pinned": row[27],
                    "expires_at": row[28],
                    "supersedes": row[29],
                    "contradiction_key": row[30],
                    "vector_distance": float(row[31]),
                    "embedding_provider": row[32],
                    "embedding_model": row[33],
                }
            )
        return records
