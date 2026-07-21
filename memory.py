from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from .embeddings import safe_embedding, vector_to_sql
from .node_utils import goal_progress_snapshot
from .state import State
from .tooling.core import STATE_DIR, ensure_state_dirs
from pydantic import BaseModel, ConfigDict, Field, field_validator


MemoryKind = Literal["task", "failure", "patch", "project", "user", "tool", "eval"]
MemoryScope = Literal["global", "user", "project", "thread", "task", "tool", "file"]
RetentionPolicy = Literal["none", "session", "ttl", "long_term", "pinned"]
ValidityStatus = Literal["active", "superseded", "deprecated", "disputed", "rejected"]
MemoryPolicyAction = Literal["store", "discard"]
MemoryAuditAction = Literal[
    "store",
    "discard",
    "supersede",
    "promote",
    "fallback",
    "retrieve",
]
MemoryWriterRoute = Literal["schedule", "finish"]

MEMORY_DIR = STATE_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "memory.jsonl"
MEMORY_AUDIT_FILE = MEMORY_DIR / "memory_audit.jsonl"
MAX_MEMORY_RECORDS = 500
MAX_MEMORY_AUDIT_EVENTS = 1000
MAX_RETRIEVED_RECORDS = 8
MAX_INDEXED_VECTOR_DIMENSION = 2000
MAX_MEMORY_TEXT_CHARS = 2000
DEFAULT_MEMORY_TTL_DAYS = 30
DEFAULT_MEMORY_BACKEND = "postgres"
MEMORY_PROMOTION_SUCCESS_THRESHOLD = 3
SENSITIVE_FIELD_NAMES = {
    "api_key",
    "authorization",
    "cookie",
    "database_url",
    "password",
    "secret",
    "token",
}


# 中文注释：
# 先把这几个概念分清楚，否则很容易混在一起：
#
# 1. MemoryRecord
#    这是“记忆本体”，也就是 agent 真的想保存的经验。
#    例如：
#      - 哪个任务完成了
#      - 哪个工具执行失败了
#      - 哪个文件和这次任务有关
#      - 这次失败原因是什么
#
# 2. Postgres
#    这是数据库，负责把 MemoryRecord 长期保存下来。
#    没有 Postgres 的时候，本项目会退回到 JSONL 文件。
#
# 3. pgvector
#    这是 Postgres 的向量扩展。
#    它让 Postgres 不仅能存普通字段，还能存 embedding 向量，
#    并且可以做“相似度搜索”。
#
# 4. EmbeddingProvider
#    这是“把文本变成向量”的组件。
#    向量数据库本身不会理解文本。
#    需要 embedding 模型先把文本转成数字向量。
#
# 5. OmlxEmbeddingProvider
#    当前项目直接使用真正的 embedding 模型。
#    如果你的本地 OMLX 提供 /v1/embeddings 接口，
#    并且加载的是 embedding 模型，就可以通过它生成语义向量。
#
# 重要：
#   当前本地向量数据库是 Postgres + pgvector。
#   当前默认向量生成器是 OmlxEmbeddingProvider。
#
# 最终运行链路大致是：
#
#   MemoryRecord 文本经验
#      -> EmbeddingProvider 生成固定维度向量
#      -> Postgres + pgvector 保存向量
#      -> Memory Retriever 根据当前任务做相似度搜索
#      -> 找回和当前任务最相关的历史经验


# Production Memory Governance
#
# 中文注释：
# 当前 memory.py 已经实现：
# - Pydantic MemoryRecord。
# - JSONL fallback。
# - Postgres 结构化表。
# - pgvector embedding 表。
# - 规则分数 + 向量检索的 hybrid retrieval 雏形。
# - scope / retention_policy / validity_status / importance / pinned。
# - expires_at 过期过滤。
# - supersedes 修正关系字段。
# - MemoryPolicy 写入前决定 store / discard。
# - 敏感字段和长文本裁剪，避免把密钥或大段日志写进长期记忆。
# - 自动把 supersedes 指向的旧记录标记为 superseded。
# - 同 contradiction_key 只返回最新 active 记忆。
# - 增加 MemoryPromotion，把多次成功/人工确认的记忆晋升为 pinned。
# - 增加 MemoryAudit，记录哪条记忆影响了哪个 Planner/Evaluator 决策。
# - 清理 expires_at 已过期的非 pinned 记忆。
#
# 后续 TODO：
# - 增加 MemoryReranker，用 reranker 或 LLM judge 做最终排序。
# - 把 MemoryAudit 接到独立查询 API / dashboard。


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


class MemoryRecord(BaseModel):
    """结构化记忆记录。

    中文注释：
    生产级 agent 的 memory 不应该只是随便塞一个 dict。
    至少要知道：
    - 这条记忆是什么类型。
    - 来自哪个 task/tool。
    - 是否成功。
    - 跟哪些路径相关。
    - 什么时候产生。
    - 是否可信。
    - 适用范围是什么。
    - 保留多久。
    - 当前是否仍然有效。

    现在使用 Pydantic，而不是普通 dict / dataclass。
    好处是：
    - 写入前做运行时校验。
    - 字段类型更明确。
    - 后续可以直接导出 JSON Schema。
    - 更接近生产级 agent 的 memory record 设计。
    """
    # 不允许出现未定义字段。
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: MemoryKind
    task_id: str
    title: str
    summary: str
    status: str
    tool_name: str
    tool_result_status: str
    paths: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    importance: float = 0.5
    scope: MemoryScope = "project"
    retention_policy: RetentionPolicy = "ttl"
    validity_status: ValidityStatus = "active"
    pinned: bool = False
    expires_at: str | None = None
    supersedes: str | None = None
    contradiction_key: str | None = None
    source: str = "memory_writer_node"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", "importance")
    @classmethod
    def _confidence_between_zero_and_one(cls, value: float) -> float:
        """限制 confidence / importance 在 0 到 1 之间。"""

        if value < 0 or value > 1:
            raise ValueError("confidence / importance 必须在 0 到 1 之间。")
        return value


class MemoryAuditEvent(BaseModel):
    """记忆治理审计事件。

    中文注释：
    生产级 memory 系统不能只保存“最后结果”，
    还要保存“为什么这么做”。
    例如：
    - 为什么这条记忆被保存？
    - 为什么这条记忆被丢弃？
    - 哪条旧记忆被 superseded？
    - 哪条记忆因为多次成功被 promotion？
    - 检索时哪些记忆进入了上下文？

    这类信息不直接参与 agent 推理，但对排查问题、复盘策略、
    调整 memory policy 很重要。
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    action: MemoryAuditAction
    memory_id: str
    reason: str
    backend: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# frozen=True = 这个对象创建后就冻结，不允许再改字段
# 这种结果最好是“确定后不再被随手改掉”。否则后面某个函数
# 不小心把 action 从 "discard" 改成 "store"，memory 系统就会出现难查的 bug。
@dataclass(frozen=True)
class MemoryPolicyDecision:
    """MemoryPolicy 的判断结果。

    中文注释：
    这层专门回答一个问题：

        pending_memory 到底要不要写入长期记忆？

    它让 Memory Writer 不再“看到 pending_memory 就无脑写入”。
    """

    action: MemoryPolicyAction
    reason: str
    scope: MemoryScope = "project"
    retention_policy: RetentionPolicy = "ttl"
    importance: float = 0.5
    pinned: bool = False
    expires_at: str | None = None
    validity_status: ValidityStatus = "active"


# Python Protocol ≈ Rust trait
# 不要求显式继承 MemoryStore
# 只要方法长得一样，就算符合
class MemoryStore(Protocol):
    """Memory 存储适配器协议。

    中文注释：
    大厂工程里通常不会让业务节点直接读写具体数据库。
    节点只依赖 MemoryStore 协议：
    - JSONL 是本地开发实现。
    - Postgres 是生产化实现。
    - 以后也可以替换成 Redis、向量数据库或云存储。
    """

    backend_name: str

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        """读取最近的记忆记录。"""

    def upsert_record(self, record: MemoryRecord) -> None:
        """插入或更新一条记忆记录。"""

    def search_similar_records(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        """语义检索相似记忆。"""

    def upsert_audit_event(self, event: MemoryAuditEvent) -> None:
        """插入或更新一条记忆审计事件。"""


class JsonlMemoryStore:
    """本地 JSONL fallback memory store。

    中文注释：
    当前项目的主记忆存储是 Postgres + pgvector。
    JSONL 只作为 fallback：
    - Postgres 没启动。
    - DATABASE_URL 没配置。
    - 本地开发临时离线。

    这样 agent 不会因为数据库暂时不可用而完全中断，
    但正常情况下不要把 JSONL 当成主记忆库。
    """

    backend_name = "jsonl"

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        return _read_jsonl_memory_records(limit)

    def upsert_record(self, record: MemoryRecord) -> None:
        _upsert_jsonl_memory_record(record.model_dump(mode="json"))

    def search_similar_records(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        return []

    def upsert_audit_event(self, event: MemoryAuditEvent) -> None:
        _upsert_jsonl_audit_event(event.model_dump(mode="json"))


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
        # 中文注释：
        # 这里会自动初始化数据库结构。
        #
        # CREATE EXTENSION vector：
        #   开启 pgvector 扩展，让 Postgres 支持 vector 类型。
        #
        # beginner_agent_memory：
        #   保存结构化记忆。
        #
        # beginner_agent_memory_embeddings_<dimension>：
        #   保存 embedding 向量。
        #   这类表由 _ensure_embedding_table(...) 按需创建。
        #
        # CREATE INDEX：
        #   给常见查询加索引，让检索更快。
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS beginner_agent_memory (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_result_status TEXT NOT NULL,
                    paths JSONB NOT NULL,
                    tags JSONB NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                    scope TEXT NOT NULL DEFAULT 'project',
                    retention_policy TEXT NOT NULL DEFAULT 'ttl',
                    validity_status TEXT NOT NULL DEFAULT 'active',
                    pinned BOOLEAN NOT NULL DEFAULT FALSE,
                    expires_at TIMESTAMPTZ,
                    supersedes TEXT,
                    contradiction_key TEXT,
                    source TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS beginner_agent_memory_audit (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL
                )
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS importance DOUBLE PRECISION NOT NULL DEFAULT 0.5
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS scope TEXT NOT NULL DEFAULT 'project'
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS retention_policy TEXT NOT NULL DEFAULT 'ttl'
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS validity_status TEXT NOT NULL DEFAULT 'active'
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS supersedes TEXT
                """
            )
            conn.execute(
                """
                ALTER TABLE beginner_agent_memory
                ADD COLUMN IF NOT EXISTS contradiction_key TEXT
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_kind
                ON beginner_agent_memory (kind)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_task_id
                ON beginner_agent_memory (task_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_tool_status
                ON beginner_agent_memory (tool_name, tool_result_status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_created_at
                ON beginner_agent_memory (created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_governance
                ON beginner_agent_memory (
                    validity_status, retention_policy, scope, pinned, expires_at
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_tags
                ON beginner_agent_memory USING GIN (tags)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_audit_memory_id
                ON beginner_agent_memory_audit (memory_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_audit_created_at
                ON beginner_agent_memory_audit (created_at DESC)
                """
            )
            conn.execute(
                """
                DELETE FROM beginner_agent_memory
                WHERE expires_at IS NOT NULL
                  AND expires_at <= NOW()
                  AND pinned = FALSE
                """
            )
            conn.execute(
                "DROP INDEX IF EXISTS idx_beginner_agent_memory_embeddings_vector"
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
                       created_at::text, metadata, importance, scope,
                       retention_policy, validity_status, pinned,
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
                    "scope": row[15],
                    "retention_policy": row[16],
                    "validity_status": row[17],
                    "pinned": row[18],
                    "expires_at": row[19],
                    "supersedes": row[20],
                    "contradiction_key": row[21],
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
                    scope, retention_policy, validity_status, pinned, expires_at,
                    supersedes, contradiction_key, source, created_at, metadata
                )
                VALUES (
                    %(id)s, %(kind)s, %(task_id)s, %(title)s, %(summary)s,
                    %(status)s, %(tool_name)s, %(tool_result_status)s,
                    %(paths)s::jsonb, %(tags)s::jsonb, %(confidence)s,
                    %(importance)s, %(scope)s, %(retention_policy)s,
                    %(validity_status)s, %(pinned)s, %(expires_at)s::timestamptz,
                    %(supersedes)s, %(contradiction_key)s, %(source)s,
                    %(created_at)s::timestamptz, %(metadata)s::jsonb
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
                    scope = EXCLUDED.scope,
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
                       m.importance, m.scope, m.retention_policy,
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
                    "scope": row[15],
                    "retention_policy": row[16],
                    "validity_status": row[17],
                    "pinned": row[18],
                    "expires_at": row[19],
                    "supersedes": row[20],
                    "contradiction_key": row[21],
                    "vector_distance": float(row[22]),
                    "embedding_provider": row[23],
                    "embedding_model": row[24],
                }
            )
        return records


def memory_record_json_schema() -> dict[str, Any]:
    """导出 MemoryRecord 的 JSON Schema。"""
    # model_json_schema() 是从 Pydantic 的 BaseModel 继承来的方法
    return MemoryRecord.model_json_schema()


def _embedding_text_for_record(record: MemoryRecord) -> str:
    """构造用于 embedding 的短文本。

    中文注释：
    不要把完整 tool output 或大文件内容塞进 embedding。
    memory embedding 应该索引“经验摘要”，而不是索引所有原始数据。
    """

    return "\n".join(
        [
            f"kind: {record.kind}",
            f"title: {record.title}",
            f"summary: {record.summary}",
            f"status: {record.status}",
            f"tool: {record.tool_name}",
            f"tool_result_status: {record.tool_result_status}",
            f"scope: {record.scope}",
            f"retention_policy: {record.retention_policy}",
            f"validity_status: {record.validity_status}",
            f"importance: {record.importance}",
            f"paths: {', '.join(record.paths)}",
            f"tags: {', '.join(record.tags)}",
        ]
    )[:2000]


def _query_text_for_state(state: State) -> str:
    """构造 Memory Retriever 的向量查询文本。"""

    current_task = state["task_tree"].get(state["current_task_id"], {})
    return "\n".join(
        [
            f"user_goal: {state['user_input']}",
            f"current_task: {current_task.get('title', '')}",
            f"tool_name: {state.get('tool_name', 'none')}",
            f"tool_result_status: {state.get('tool_result_status', 'none')}",
        ]
    )


def _memory_ttl_days() -> int:
    """读取 TTL 记忆默认保留天数。"""

    raw = os.getenv("BEGINNER_AGENT_MEMORY_TTL_DAYS", str(DEFAULT_MEMORY_TTL_DAYS))
    try:
        return max(1, int(raw.strip()))
    except ValueError:
        return DEFAULT_MEMORY_TTL_DAYS


def _expires_at_for_policy(retention_policy: RetentionPolicy) -> str | None:
    """根据 retention_policy 计算 expires_at。"""

    if retention_policy in {"none", "session", "long_term", "pinned"}:
        return None
    return (datetime.now(timezone.utc) + timedelta(days=_memory_ttl_days())).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    """安全解析 ISO datetime。"""

    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _record_is_active(record: dict[str, Any]) -> bool:
    """判断记忆是否仍可被默认检索使用。"""

    if str(record.get("validity_status", "active")) != "active":
        return False
    if bool(record.get("pinned", False)):
        return True
    expires_at = _parse_datetime(record.get("expires_at"))
    return expires_at is None or expires_at > datetime.now(timezone.utc)


def _record_should_be_deleted(record: dict[str, Any]) -> bool:
    """判断过期记忆是否应该被物理清理。"""

    if bool(record.get("pinned", False)):
        return False
    expires_at = _parse_datetime(record.get("expires_at"))
    return expires_at is not None and expires_at <= datetime.now(timezone.utc)


def _record_created_at(record: dict[str, Any]) -> datetime:
    """读取记录创建时间，解析失败时使用很早的时间。"""

    return _parse_datetime(record.get("created_at")) or datetime.min.replace(
        tzinfo=timezone.utc
    )


def _dedupe_contradiction_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同 contradiction_key 只保留最新 active 记忆。

    中文注释：
    contradiction_key 表示“这些记忆在同一个问题上可能互相修正”。
    例如：
    - 旧记忆：embedding 默认模型是 A。
    - 新记忆：embedding 默认模型已经改成 B。

    这时不应该把 A 和 B 同时喂给 Planner。
    当前策略是：同 key 只返回最新 active 记录。
    """

    without_key: list[dict[str, Any]] = []
    latest_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("contradiction_key") or "").strip()
        if not key:
            without_key.append(record)
            continue
        current = latest_by_key.get(key)
        if current is None or _record_created_at(record) > _record_created_at(current):
            latest_by_key[key] = record
    return [*without_key, *latest_by_key.values()]


def _scope_matches_state(record: dict[str, Any], state: State) -> bool:
    """判断记忆 scope 是否适合当前任务。"""

    # 后续 TODO：Memory Scope Intelligence
    #
    # 中文注释：
    # 当前函数只做“硬规则过滤”，它适合守住安全边界：
    # - global / user / project 级记忆可以进入候选池。
    # - task 级记忆只能匹配当前 task_id。
    # - tool 级记忆只能匹配当前工具。
    # - file 级记忆只能匹配当前文件路径。
    #
    # 更接近大厂风格的下一步，不是把这里直接替换成 LLM 判断，
    # 而是在硬过滤之后增加单独的智能层：
    #
    # 1. MemoryReranker
    #    对已经通过 scope 硬过滤的候选记忆重新排序。
    #
    # 2. MemoryRelevanceJudge
    #    用 LLM / reranker / cross-encoder 判断记忆是否真的有帮助。
    #
    # 3. MemoryUsageAudit
    #    记录“哪条记忆影响了哪个 Planner / Evaluator 决策”。
    #
    # 4. MemoryAccessPolicy
    #    对敏感 scope 做更严格权限控制，例如 user / file / thread。
    #
    # 原则：
    #   这个函数继续保持简单、稳定、可解释；
    #   智能判断放到后面的 reranker / judge / audit 层。
    scope = str(record.get("scope", "project"))
    if scope in {"global", "user", "project"}:
        return True
    if scope == "task":
        return str(record.get("task_id", "")) == state["current_task_id"]
    if scope == "tool":
        return str(record.get("tool_name", "")) in {state.get("tool_name", ""), "none"}
    if scope == "file":
        current_task = state["task_tree"].get(state["current_task_id"], {})
        args = current_task.get("args", {}) if isinstance(current_task, dict) else {}
        current_path = str(args.get("path", "")) if isinstance(args, dict) else ""
        return bool(current_path and current_path in record.get("paths", []))
    if scope == "thread":
        return True
    return False


def _redact_sensitive_text(text: str) -> str:
    """对常见敏感片段做轻量脱敏和截断。

    中文注释：
    记忆系统会跨轮次保存内容，所以不能把 api key、token、password
    这类值原样写进去。这里不追求替代专业 DLP 系统，但至少把常见
    key=value / key: value 形式的秘密值替换成 [REDACTED]。
    """

    redacted = text
    for marker in ("api_key", "token", "password", "secret", "authorization"):
        redacted = re.sub(
            rf"(?i)({marker})\s*[:=]\s*([^\s,;|]+)",
            r"\1=[REDACTED]",
            redacted,
        )
    if len(redacted) > MAX_MEMORY_TEXT_CHARS:
        return redacted[:MAX_MEMORY_TEXT_CHARS] + "...[TRUNCATED]"
    return redacted


def _safe_memory_value(value: Any, *, key: str = "") -> Any:
    """把要写入 memory metadata 的值裁剪成安全版本。"""

    normalized_key = key.lower()
    if any(name in normalized_key for name in SENSITIVE_FIELD_NAMES):
        return "[REDACTED]"
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    if isinstance(value, dict):
        return {
            str(item_key): _safe_memory_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_safe_memory_value(item) for item in value[:50]]
    return value


def _memory_has_success_evidence(pending_memory: dict[str, Any]) -> bool:
    """判断 pending_memory 是否带有可验证成功证据。

    中文注释：
    生产级 memory 不应该只因为模型说“成功了”就长期保存。
    更可信的证据通常来自：
    - tests_passed
    - build_passed
    - verification_passed
    - lint_passed / typecheck_passed
    - human_confirmed / approval
    """

    data = pending_memory.get("tool_result_data")
    if not isinstance(data, dict):
        data = {}
    evidence_keys = {
        "tests_passed",
        "build_passed",
        "verification_passed",
        "lint_passed",
        "typecheck_passed",
        "human_confirmed",
        "approval",
    }
    if any(bool(data.get(key)) for key in evidence_keys):
        return True
    metadata = pending_memory.get("metadata")
    if isinstance(metadata, dict) and any(bool(metadata.get(key)) for key in evidence_keys):
        return True
    return any(bool(pending_memory.get(key)) for key in evidence_keys)


def _memory_policy_for_pending(
    state: State,
    pending_memory: dict[str, Any],
    *,
    tool_name: str,
    tool_result_status: str,
) -> MemoryPolicyDecision:
    """决定 pending_memory 是否应该写入长期记忆系统。

    中文注释：
    这是当前版本的 MemoryPolicy：
    - 明显没有信息量的内容丢弃。
    - 成功的代码修改/项目结构经验保留更久。
    - 失败经验默认 ttl，因为它有价值但可能会过期。
    - secret_scan 相关内容不存原始结果，只允许存摘要。
    """

    # Production Memory Policy
    #
    # 中文注释：
    # 当前 MemoryPolicy 已经不只是简单 if/else。
    # 它会和 _govern_memory_record_before_write(...) 配合，形成：
    # - 规则判断。
    # - 证据保留。
    # - 多次成功 promotion。
    # - supersedes / contradiction_key 治理。
    # - store / discard / retrieve 审计。
    #
    # 已实现：
    #
    # 1. Evidence-based Retention 雏形
    #    - 有测试通过 / build 通过 / 人工确认 -> long_term 或 pinned。
    #    - 只是一次临时失败日志 -> ttl。
    #    - 没有可验证证据 -> 降低 importance 或 discard。
    #
    # 2. Frequency-based Promotion
    #    同类成功经验多次出现后，自动提高 importance。
    #    例如同一个修复模式连续 3 次成功，可以晋升为 pinned 候选。
    #
    # 3. Failure Memory Governance 雏形
    #    失败经验有价值，但也容易过期。
    #    当前失败默认 TTL，避免长期污染记忆库。
    #
    # 4. Human-confirmed Memory
    #    governance 层会读取 approval / reviewer / human_confirmed 信号，
    #    并提升 confidence、importance、pinned。
    #
    # 5. Sensitive Memory Policy
    #    安全相关工具不应该保存原始密钥、token、隐私内容。
    #    当前会做轻量脱敏，并把拒绝保存原因写入 audit log。
    #
    # 后续 TODO：
    #
    # 6. LLM Memory Judge
    #    不要让 LLM 直接替代这层硬规则。
    #    更合理的是：规则先过滤明显情况，再让 LLM / reranker
    #    判断“这条记忆是否值得长期保存、是否会误导后续任务”。
    #
    # 7. DLP / Secret Scanner
    #    当前是轻量正则脱敏。
    #    后续可以接入更强 secret scanner / DLP 策略。
    title = str(pending_memory.get("title", "")).strip()
    reason = str(pending_memory.get("reason", "")).strip()
    if not title and not reason:
        return MemoryPolicyDecision("discard", "pending_memory 缺少 title/reason。")

    if tool_name == "none" and tool_result_status == "none":
        return MemoryPolicyDecision("discard", "没有工具结果，不写入长期记忆。")

    if tool_name == "secret_scan":
        return MemoryPolicyDecision(
            "store",
            "安全扫描结果只保存摘要。",
            retention_policy="ttl",
            importance=0.7,
        )

    if tool_result_status in {"failed", "blocked", "empty"}:
        return MemoryPolicyDecision(
            "store",
            "失败经验可帮助后续恢复，但默认 TTL。",
            retention_policy="ttl",
            importance=0.75,
        )

    if tool_result_status == "success" and _memory_has_success_evidence(pending_memory):
        return MemoryPolicyDecision(
            "store",
            "有测试/build/人工确认等成功证据，长期保留。",
            retention_policy="long_term",
            importance=0.9,
        )

    if tool_name in {"apply_patch", "apply_patch_plan", "rollback", "revert_file_patch"}:
        return MemoryPolicyDecision(
            "store",
            "代码修改经验需要长期保留。",
            retention_policy="long_term",
            importance=0.85,
        )

    if tool_name in {"read_file", "summarize_file", "inspect_symbol", "inspect_import_graph"}:
        return MemoryPolicyDecision(
            "store",
            "项目理解类记忆按项目 scope 保存。",
            scope="project",
            retention_policy="ttl",
            importance=0.6,
        )

    if state.get("done") and tool_result_status == "success":
        return MemoryPolicyDecision(
            "store",
            "完成目标的成功经验可长期保留。",
            retention_policy="long_term",
            importance=0.8,
        )

    return MemoryPolicyDecision("store", "默认写入 TTL 记忆。")


def _configured_store() -> MemoryStore:
    """根据环境变量选择主 memory store。

    中文注释：
    生产风格的默认选择是 Postgres + pgvector。
    因为它同时支持：
    - 结构化字段查询。
    - 持久化审计。
    - pgvector 语义检索。

    JSONL 仍然保留，但定位是 fallback。
    如果你显式设置：

        BEGINNER_AGENT_MEMORY_BACKEND=jsonl

    才会直接使用 JSONL。
    """

    backend = os.getenv("BEGINNER_AGENT_MEMORY_BACKEND", DEFAULT_MEMORY_BACKEND)
    backend = backend.strip().lower()
    if backend == "jsonl":
        return JsonlMemoryStore()
    if backend in {"postgres", "pgvector", "vector"}:
        database_url = os.getenv("DATABASE_URL", "").strip()
        if database_url:
            return PostgresMemoryStore(database_url)
        raise RuntimeError(
            "BEGINNER_AGENT_MEMORY_BACKEND=postgres 需要配置 DATABASE_URL。"
        )
    raise ValueError(
        "不支持的 BEGINNER_AGENT_MEMORY_BACKEND："
        f"{backend}。当前支持 postgres / pgvector / jsonl。"
    )


def _ensure_memory_file() -> None:
    """确保 memory 存储文件存在。"""

    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("", encoding="utf-8")


def _stable_memory_id(record: dict[str, Any]) -> str:
    """基于关键字段生成稳定 ID，用于去重。

    中文注释：
    如果同一个 task/tool/status 重复写入，我们希望覆盖旧记录，
    而不是把 memory.jsonl 写成很多重复噪声。
    """

    raw = json.dumps(
        {
            "kind": record.get("kind", "task"),
            "task_id": record.get("task_id", ""),
            "title": record.get("title", ""),
            "tool_name": record.get("tool_name", "none"),
            "status": record.get("status", ""),
            "paths": record.get("paths", []),
            "scope": record.get("scope", "project"),
            "contradiction_key": record.get("contradiction_key"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _stable_audit_id(event: dict[str, Any]) -> str:
    """为 audit event 生成稳定 ID。"""

    raw = json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _build_audit_event(
    *,
    action: MemoryAuditAction,
    memory_id: str,
    reason: str,
    backend: str,
    metadata: dict[str, Any] | None = None,
) -> MemoryAuditEvent:
    """构造标准化 memory audit event。"""

    safe_metadata = _safe_memory_value(metadata or {})
    raw_event = {
        "action": action,
        "memory_id": memory_id,
        "reason": reason,
        "backend": backend,
        "metadata": safe_metadata,
    }
    return MemoryAuditEvent(
        id=_stable_audit_id(raw_event),
        action=action,
        memory_id=memory_id,
        reason=reason,
        backend=backend,
        metadata=safe_metadata,
    )


def _read_jsonl_memory_records(limit: int) -> list[dict[str, Any]]:
    """从 JSONL fallback 文件读取历史记忆。

    中文注释：
    JSONL 是一行一个 JSON。
    这个格式简单、可追加、方便肉眼查看。

    但在当前架构里，它不是主记忆库。
    主记忆库是 Postgres + pgvector。
    """

    _ensure_memory_file()
    records: list[dict[str, Any]] = []
    for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    kept_records = [record for record in records if not _record_should_be_deleted(record)]
    if len(kept_records) != len(records):
        _write_jsonl_memory_records(kept_records)
    records = kept_records
    return records[-limit:]


def _write_jsonl_memory_records(records: list[dict[str, Any]]) -> None:
    """把记忆记录写回 JSONL 文件。"""

    _ensure_memory_file()
    trimmed = records[-MAX_MEMORY_RECORDS:]
    MEMORY_FILE.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            for record in trimmed
        ),
        encoding="utf-8",
    )


def _read_jsonl_audit_events(limit: int) -> list[dict[str, Any]]:
    """读取 JSONL fallback 审计事件。"""

    _ensure_memory_file()
    if not MEMORY_AUDIT_FILE.exists():
        MEMORY_AUDIT_FILE.write_text("", encoding="utf-8")
    events: list[dict[str, Any]] = []
    for line in MEMORY_AUDIT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events[-limit:]


def _write_jsonl_audit_events(events: list[dict[str, Any]]) -> None:
    """把审计事件写回 JSONL fallback 文件。"""

    _ensure_memory_file()
    trimmed = events[-MAX_MEMORY_AUDIT_EVENTS:]
    MEMORY_AUDIT_FILE.write_text(
        "".join(
            f"{json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            for event in trimmed
        ),
        encoding="utf-8",
    )


def _upsert_jsonl_audit_event(event: dict[str, Any]) -> None:
    """插入或更新一条 JSONL audit event。"""

    events = _read_jsonl_audit_events(MAX_MEMORY_AUDIT_EVENTS)
    event_id = str(event["id"])
    kept = [item for item in events if str(item.get("id", "")) != event_id]
    kept.append(event)
    _write_jsonl_audit_events(kept)


def _upsert_jsonl_memory_record(record: dict[str, Any]) -> None:
    """插入或更新一条记忆。"""

    records = _read_jsonl_memory_records(MAX_MEMORY_RECORDS)
    record_id = str(record["id"])
    supersedes = str(record.get("supersedes") or "").strip()
    contradiction_key = str(record.get("contradiction_key") or "").strip()
    kept = [item for item in records if str(item.get("id", "")) != record_id]
    for item in kept:
        same_superseded_id = supersedes and str(item.get("id", "")) == supersedes
        same_contradiction_key = (
            contradiction_key
            and str(item.get("contradiction_key") or "") == contradiction_key
            and str(item.get("validity_status", "active")) == "active"
        )
        if same_superseded_id or same_contradiction_key:
            item["validity_status"] = "superseded"
    kept.append(record)
    _write_jsonl_memory_records(kept)


def _list_memory_records() -> tuple[list[dict[str, Any]], str, str]:
    """读取 memory records，并返回 backend 信息。

    中文注释：
    如果配置了 Postgres 但连接失败，不让整个 agent 崩掉。
    它会回退到 JSONL，并把错误原因写进 memory_context。
    """

    try:
        store = _configured_store()
        records = [
            record
            for record in store.list_records(MAX_MEMORY_RECORDS)
            if _record_is_active(record)
        ]
        records = _dedupe_contradiction_records(records)
        return records, store.backend_name, ""
    except Exception as exc:
        fallback = JsonlMemoryStore()
        records = [
            record
            for record in fallback.list_records(MAX_MEMORY_RECORDS)
            if _record_is_active(record)
        ]
        records = _dedupe_contradiction_records(records)
        return records, fallback.backend_name, str(exc)


def _search_vector_records(query_text: str) -> tuple[list[dict[str, Any]], str, str]:
    """执行向量检索，并返回 backend 信息。"""

    try:
        store = _configured_store()
        records = store.search_similar_records(query_text, MAX_RETRIEVED_RECORDS)
        return records, store.backend_name, ""
    except Exception as exc:
        return [], "jsonl-fallback", str(exc)


def _records_share_memory_pattern(record: MemoryRecord, existing: dict[str, Any]) -> bool:
    """判断两条记忆是否属于同一类可复用经验。"""

    if str(existing.get("tool_result_status", "")) != "success":
        return False
    if str(existing.get("kind", "")) != record.kind:
        return False
    if str(existing.get("tool_name", "")) != record.tool_name:
        return False
    if (
        record.contradiction_key
        and str(existing.get("contradiction_key") or "") == record.contradiction_key
    ):
        return True
    existing_paths = {str(path) for path in existing.get("paths", [])}
    return bool(existing_paths.intersection(record.paths))


def _govern_memory_record_before_write(
    record: MemoryRecord,
    existing_records: list[dict[str, Any]],
    *,
    backend: str,
) -> tuple[MemoryRecord, list[MemoryAuditEvent]]:
    """写入前执行记忆治理。

    中文注释：
    这里把原来 TODO 里的几件事落成代码：
    - supersedes：新记录明确修正旧记录时，写审计事件。
    - contradiction_key：同一问题的新记录会让旧记录失效。
    - promotion：同类成功经验多次出现时，提高 importance / pinned。

    真正更新旧记录状态由 store 层完成：
    - PostgresMemoryStore.upsert_record(...)
    - _upsert_jsonl_memory_record(...)
    """

    events: list[MemoryAuditEvent] = []
    updates: dict[str, Any] = {}
    metadata_updates = dict(record.metadata)

    if record.supersedes:
        events.append(
            _build_audit_event(
                action="supersede",
                memory_id=record.id,
                reason="新记忆声明 supersedes，旧记忆将被标记为 superseded。",
                backend=backend,
                metadata={
                    "supersedes": record.supersedes,
                    "new_memory_id": record.id,
                },
            )
        )

    if record.contradiction_key:
        shadowed_ids = [
            str(item.get("id"))
            for item in existing_records
            if str(item.get("id", "")) != record.id
            and str(item.get("contradiction_key") or "") == record.contradiction_key
            and str(item.get("validity_status", "active")) == "active"
        ]
        if shadowed_ids:
            events.append(
                _build_audit_event(
                    action="supersede",
                    memory_id=record.id,
                    reason="同 contradiction_key 的旧 active 记忆将被新记忆覆盖。",
                    backend=backend,
                    metadata={
                        "contradiction_key": record.contradiction_key,
                        "shadowed_memory_ids": shadowed_ids[:20],
                    },
                )
            )

    matching_successes = [
        item for item in existing_records if _records_share_memory_pattern(record, item)
    ]
    human_confirmed = bool(
        metadata_updates.get("approval")
        or metadata_updates.get("human_confirmed")
        or metadata_updates.get("reviewer")
    )
    should_promote = (
        record.tool_result_status == "success"
        and (
            len(matching_successes) + 1 >= MEMORY_PROMOTION_SUCCESS_THRESHOLD
            or human_confirmed
        )
    )
    if should_promote:
        updates["importance"] = max(record.importance, 0.95)
        updates["pinned"] = True
        updates["retention_policy"] = "pinned"
        updates["expires_at"] = None
        metadata_updates["memory_promotion"] = {
            "reason": "同类成功经验多次出现或经过人工确认。",
            "matching_success_count": len(matching_successes) + 1,
            "human_confirmed": human_confirmed,
        }
        updates["metadata"] = metadata_updates
        events.append(
            _build_audit_event(
                action="promote",
                memory_id=record.id,
                reason="记忆被提升为 pinned 长期经验。",
                backend=backend,
                metadata=metadata_updates["memory_promotion"],
            )
        )

    if not updates:
        return record, events
    return record.model_copy(update=updates), events


def _upsert_memory_record(record: MemoryRecord) -> tuple[str, str]:
    """写入 memory record，并返回 backend 信息。"""

    try:
        store = _configured_store()
        existing_records = store.list_records(MAX_MEMORY_RECORDS)
        governed_record, governance_events = _govern_memory_record_before_write(
            record,
            existing_records,
            backend=store.backend_name,
        )
        store.upsert_record(governed_record)
        for event in governance_events:
            store.upsert_audit_event(event)
        store.upsert_audit_event(
            _build_audit_event(
                action="store",
                memory_id=governed_record.id,
                reason="Memory Writer 写入结构化记忆。",
                backend=store.backend_name,
                metadata={
                    "kind": governed_record.kind,
                    "tool_name": governed_record.tool_name,
                    "retention_policy": governed_record.retention_policy,
                    "importance": governed_record.importance,
                    "pinned": governed_record.pinned,
                },
            )
        )
        return store.backend_name, ""
    except Exception as exc:
        fallback = JsonlMemoryStore()
        existing_records = fallback.list_records(MAX_MEMORY_RECORDS)
        governed_record, governance_events = _govern_memory_record_before_write(
            record,
            existing_records,
            backend=fallback.backend_name,
        )
        fallback.upsert_record(governed_record)
        for event in governance_events:
            fallback.upsert_audit_event(event)
        fallback.upsert_audit_event(
            _build_audit_event(
                action="fallback",
                memory_id=governed_record.id,
                reason="主 memory store 不可用，写入 JSONL fallback。",
                backend=fallback.backend_name,
                metadata={
                    "error": str(exc),
                    "kind": governed_record.kind,
                    "tool_name": governed_record.tool_name,
                },
            )
        )
        return fallback.backend_name, str(exc)


def _upsert_memory_audit_event(event: MemoryAuditEvent) -> tuple[str, str]:
    """写入 memory audit event，并在失败时回退 JSONL。"""

    try:
        store = _configured_store()
        store.upsert_audit_event(event)
        return store.backend_name, ""
    except Exception as exc:
        fallback = JsonlMemoryStore()
        fallback.upsert_audit_event(
            event.model_copy(
                update={
                    "backend": fallback.backend_name,
                    "metadata": {
                        **event.metadata,
                        "fallback_error": str(exc),
                    },
                }
            )
        )
        return fallback.backend_name, str(exc)


def _extract_paths(memory: dict[str, Any]) -> list[str]:
    """从 pending_memory 里提取相关文件路径。"""

    paths: list[str] = []
    task_args = memory.get("args")
    if isinstance(task_args, dict) and task_args.get("path"):
        paths.append(str(task_args["path"]))
    metadata = memory.get("metadata")
    if isinstance(metadata, dict):
        path = metadata.get("path")
        if path:
            paths.append(str(path))
    artifact_paths = memory.get("artifact_paths")
    if isinstance(artifact_paths, list):
        paths.extend(str(path) for path in artifact_paths)
    return sorted(set(paths))


def _classify_memory_kind(memory: dict[str, Any]) -> MemoryKind:
    """根据任务结果粗略分类记忆类型。"""

    decision = str(memory.get("decision", ""))
    tool_result_status = str(memory.get("tool_result_status", "none"))
    tool_name = str(memory.get("tool_name", "none"))
    if decision == "fail" or tool_result_status in ("failed", "blocked", "empty"):
        return "failure"
    if tool_name in ("apply_patch", "apply_patch_plan", "rollback", "revert_file_patch"):
        return "patch"
    if tool_name.startswith("run_") or tool_name in ("static_check", "lint_typecheck", "run_tests"):
        return "eval"
    if tool_name in ("inspect_symbol", "inspect_import_graph", "summarize_file", "read_file"):
        return "project"
    return "task"


def _memory_tags(memory: dict[str, Any]) -> list[str]:
    """生成便于检索的标签。"""

    tags = {
        str(memory.get("decision", "")),
        str(memory.get("tool_result_status", "")),
        str(memory.get("tool_name", "")),
    }
    tags.update(
        Path(path).suffix.lstrip(".")
        for path in _extract_paths(memory)
        if Path(path).suffix
    )
    return sorted(tag for tag in tags if tag and tag != "none")


def _build_memory_record(state: State, pending_memory: dict[str, Any]) -> MemoryRecord:
    """把 Task Committer 产出的 pending_memory 标准化成 MemoryRecord。"""

    task_id = str(pending_memory.get("task_id", state["current_task_id"]))
    task = dict(state["task_tree"].get(task_id, {}))
    tool_result_data = pending_memory.get("tool_result_data")
    if not isinstance(tool_result_data, dict):
        tool_result_data = {}
    tool_name = str(task.get("tool") or state["tool_name"] or "none")
    changed_files = [
        str(path)
        for path in tool_result_data.get("changed_files", [])
        if isinstance(tool_result_data.get("changed_files", []), list)
    ]
    paths = sorted(
        set(_extract_paths({**pending_memory, "args": task.get("args", {})}) + changed_files)
    )
    status = str(task.get("status") or pending_memory.get("decision") or "unknown")
    tool_result_status = str(
        pending_memory.get("tool_result_status")
        or task.get("tool_result_status")
        or state["tool_result_status"]
        or "none"
    )
    policy = _memory_policy_for_pending(
        state,
        pending_memory,
        tool_name=tool_name,
        tool_result_status=tool_result_status,
    )
    summary = _redact_sensitive_text(
        f"{pending_memory.get('title', task.get('title', ''))} | "
        f"decision={pending_memory.get('decision', 'none')} | "
        f"reason={pending_memory.get('reason', '')}"
    )[:800]
    raw_record = {
        "kind": _classify_memory_kind({**pending_memory, "tool_name": tool_name}),
        "task_id": task_id,
        "title": str(pending_memory.get("title") or task.get("title", "")),
        "tool_name": tool_name,
        "status": status,
        "paths": paths,
        "scope": policy.scope,
        "contradiction_key": pending_memory.get("contradiction_key"),
    }
    record_id = _stable_memory_id(raw_record)
    confidence = 0.9 if tool_result_status == "success" else 0.65
    if policy.pinned:
        confidence = max(confidence, 0.95)
    return MemoryRecord(
        id=record_id,
        kind=raw_record["kind"],
        task_id=task_id,
        title=_redact_sensitive_text(raw_record["title"])[:200],
        summary=summary,
        status=status,
        tool_name=tool_name,
        tool_result_status=tool_result_status,
        paths=paths,
        tags=_memory_tags({**pending_memory, "tool_name": tool_name}),
        confidence=confidence,
        importance=policy.importance,
        scope=policy.scope,
        retention_policy=policy.retention_policy,
        validity_status=policy.validity_status,
        pinned=policy.pinned or policy.retention_policy == "pinned",
        expires_at=policy.expires_at or _expires_at_for_policy(policy.retention_policy),
        supersedes=str(pending_memory.get("supersedes") or "") or None,
        contradiction_key=(
            str(pending_memory.get("contradiction_key") or "") or None
        ),
        metadata={
            "memory_policy": {
                "action": policy.action,
                "reason": policy.reason,
            },
            "parent_evaluation": _safe_memory_value(
                pending_memory.get("parent_evaluation", {})
            ),
            "goal_progress": _safe_memory_value(pending_memory.get("goal_progress", {})),
            "tool_result_data": _safe_memory_value(tool_result_data),
            "source_memory": _safe_memory_value(pending_memory),
        },
    )

def _score_record(record: dict[str, Any], state: State) -> int:
    """给一条历史记忆打分，分数越高越相关。

    中文注释：
    这是 hybrid retrieval 里的“规则打分”部分。
    它和 pgvector 语义检索一起工作：
    - 规则分数适合处理路径、工具名、状态、关键词。
    - 向量检索适合处理“意思相近但词不一样”的经验。
    - 后续 TODO 里的 reranker 可以在两者之后做最终排序。
    """

    query = state["user_input"].lower()
    current_task = state["task_tree"].get(state["current_task_id"], {})
    task_text = str(current_task.get("title", "")).lower()
    score = 0
    haystack = " ".join(
        [
            str(record.get("title", "")),
            str(record.get("summary", "")),
            str(record.get("tool_name", "")),
            " ".join(str(tag) for tag in record.get("tags", [])),
            " ".join(str(path) for path in record.get("paths", [])),
        ]
    ).lower()
    for token in set(query.replace("/", " ").replace("_", " ").split()):
        if len(token) >= 2 and token in haystack:
            score += 2
    for token in set(task_text.replace("/", " ").replace("_", " ").split()):
        if len(token) >= 2 and token in haystack:
            score += 3
    if record.get("kind") == "failure":
        score += 1
    if record.get("tool_result_status") == "success":
        score += 1
    if record.get("pinned"):
        score += 5
    score += int(float(record.get("importance", 0.5)) * 4)
    score += int(float(record.get("confidence", 0.7)) * 2)
    return score


def _retrieve_relevant_records(state: State) -> tuple[list[dict[str, Any]], str, str]:
    """检索和当前目标最相关的历史记忆。"""

    records, backend, backend_error = _list_memory_records()
    query_text = _query_text_for_state(state)
    vector_records, vector_backend, vector_error = _search_vector_records(query_text)
    records = [record for record in records if _scope_matches_state(record, state)]
    vector_records = [
        record
        for record in vector_records
        if _record_is_active(record) and _scope_matches_state(record, state)
    ]
    vector_records = _dedupe_contradiction_records(vector_records)
    scored = [(record, _score_record(record, state)) for record in records]
    relevant = [item for item in scored if item[1] > 0]
    relevant.sort(key=lambda item: item[1], reverse=True)
    merged: dict[str, dict[str, Any]] = {}
    for record in vector_records:
        merged[str(record.get("id", ""))] = {
            **record,
            "retrieval_source": "vector",
            "retrieval_reason": "pgvector 相似度召回。",
            "retrieval_score": (
                1.0
                - float(record.get("vector_distance", 1.0))
                + float(record.get("importance", 0.5))
            ),
        }
    for record, score in relevant:
        record_id = str(record.get("id", ""))
        if record_id in merged:
            merged[record_id]["retrieval_source"] = "hybrid"
            merged[record_id]["rule_score"] = score
            merged[record_id]["retrieval_reason"] = "规则关键词 + pgvector 混合召回。"
            previous_score = float(merged[record_id].get("retrieval_score", 0))
            merged[record_id]["retrieval_score"] = previous_score + (score / 10)
        else:
            merged[record_id] = {
                **record,
                "retrieval_source": "rule",
                "retrieval_reason": "规则关键词召回。",
                "rule_score": score,
                "retrieval_score": score / 10,
            }
    results = sorted(
        merged.values(),
        key=lambda record: float(record.get("retrieval_score", 0)),
        reverse=True,
    )[:MAX_RETRIEVED_RECORDS]
    errors = "; ".join(error for error in (backend_error, vector_error) if error)
    if vector_records:
        backend = f"{backend}+vector"
    elif vector_error:
        backend = f"{backend}; vector_backend={vector_backend}"
    return results, backend, errors


def memory_retriever_node(state: State) -> dict[str, object]:
    """Memory Retriever：在复杂 agent loop 开始前读取相关记忆。

    中文注释：
    升级后这里不只是读 State.memory_notes。
    它会同时读取并治理：
    - 当前 State 里的短期记忆。
    - JSONL / Postgres 里的持久化记忆。
    - 默认过滤 superseded / rejected / expired 记忆。
    - 根据 scope、importance、confidence、pinned 做排序。

    这样 agent 下一次运行时，也能看到之前任务沉淀下来的经验。
    """

    state_notes = list(state["memory_notes"])[-5:]
    persisted_records, backend, backend_error = _retrieve_relevant_records(state)
    retrieved_ids = [str(record.get("id", "")) for record in persisted_records]
    audit_backend, audit_error = _upsert_memory_audit_event(
        _build_audit_event(
            action="retrieve",
            memory_id=state.get("current_task_id", "") or "memory_retriever",
            reason="Memory Retriever 将相关记忆写入 memory_context。",
            backend=backend,
            metadata={
                "retrieved_memory_ids": retrieved_ids,
                "retrieved_count": len(retrieved_ids),
                "backend_error": backend_error,
            },
        )
    )
    memory_context = {
        "source": "state.memory_notes + postgres/pgvector memory with jsonl fallback",
        "backend": backend,
        "backend_error": backend_error,
        "audit_backend": audit_backend,
        "audit_error": audit_error,
        "record_schema": memory_record_json_schema(),
        "governance": {
            "filters": ["active", "not_expired", "scope_matched"],
            "ranking": [
                "vector_distance",
                "rule_score",
                "importance",
                "confidence",
                "pinned",
            ],
        },
        "state_note_count": len(state_notes),
        "persisted_match_count": len(persisted_records),
        "recent_notes": state_notes,
        "relevant_records": persisted_records,
    }
    return {
        "memory_context": memory_context,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Memory Retriever：读取到 "
                    f"{len(state_notes)} 条短期记忆，"
                    f"{len(persisted_records)} 条相关持久记忆，"
                    f"backend={backend}。"
                ),
            }
        ],
    }


def memory_writer_node(state: State) -> dict[str, object]:
    """Memory Writer：把本轮任务经验写入轻量记忆和持久化记忆库。

    中文注释：
    当前是本地生产化的记忆写入链路：
    - State.memory_notes：方便你在运行结果里直接看到。
    - Postgres + pgvector：主记忆库，支持结构化查询和向量检索。
    - memory.jsonl：fallback，只有主记忆库不可用时兜底。
    - MemoryRecord：让记忆有类型、状态、来源、路径和可信度。

    当前已经引入记忆治理：
    - MemoryPolicy 先判断 store / discard。
    - MemoryRecord 带 scope、retention_policy、validity_status、importance。
    - 敏感字段和长文本会被裁剪。
    - Postgres 后端会写 pgvector embedding。
    """

    pending_memory = dict(state["pending_memory"])
    task_tree = dict(state["task_tree"])
    goal_progress = goal_progress_snapshot(state, task_tree)
    update: dict[str, object] = {
        "pending_memory": {},
        "goal_progress": goal_progress,
        "next_action": "finish" if state["done"] else "schedule",
        "messages": [
            {
                "role": "assistant",
                "content": "Memory Writer：没有新的 pending_memory，回到 Scheduler。",
            }
        ],
    }
    if not pending_memory:
        return update

    task_id = str(pending_memory.get("task_id", state["current_task_id"]))
    task = dict(task_tree.get(task_id, {}))
    tool_name = str(task.get("tool") or state["tool_name"] or "none")
    tool_result_status = str(
        pending_memory.get("tool_result_status")
        or task.get("tool_result_status")
        or state["tool_result_status"]
        or "none"
    )
    policy = _memory_policy_for_pending(
        state,
        pending_memory,
        tool_name=tool_name,
        tool_result_status=tool_result_status,
    )
    if policy.action == "discard":
        _upsert_memory_audit_event(
            _build_audit_event(
                action="discard",
                memory_id=task_id,
                reason=policy.reason,
                backend="memory_policy",
                metadata={
                    "tool_name": tool_name,
                    "tool_result_status": tool_result_status,
                    "pending_memory": _safe_memory_value(pending_memory),
                },
            )
        )
        update["messages"] = [
            {
                "role": "assistant",
                "content": f"Memory Writer：跳过写入记忆，原因：{policy.reason}",
            }
        ]
        return update

    record = _build_memory_record(state, pending_memory)
    backend, backend_error = _upsert_memory_record(record)
    record_dict = record.model_dump(mode="json")
    if backend_error:
        record_dict["backend_warning"] = backend_error
    record_dict["backend"] = backend
    update["memory_notes"] = [record_dict]
    update["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Memory Writer：写入结构化记忆 "
                f"{record.id}，类型={record.kind}，"
                f"工具={record.tool_name}，backend={backend}。"
            ),
        }
    ]
    return update


def route_after_memory_writer(state: State) -> MemoryWriterRoute:
    """Memory Writer 后的路由。"""

    if state["done"] or state["next_action"] == "finish":
        return "finish"
    return "schedule"
