from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from .embeddings import safe_embedding, vector_to_sql
from .node_utils import goal_progress_snapshot
from .state import State
from .tooling.core import STATE_DIR, ensure_state_dirs
from pydantic import BaseModel, ConfigDict, Field, field_validator


MemoryKind = Literal["task", "failure", "patch", "project", "user", "tool", "eval"]
MemoryWriterRoute = Literal["schedule", "finish"]

MEMORY_DIR = STATE_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "memory.jsonl"
MAX_MEMORY_RECORDS = 500
MAX_RETRIEVED_RECORDS = 8
MAX_INDEXED_VECTOR_DIMENSION = 2000


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
#    向量数据库本身不会理解文本，需要 embedding 模型先把文本转成数字向量。
#
# 5. HashEmbeddingProvider
#    这是本项目默认的测试 embedding。
#    它不是智能语义模型，但能稳定生成 384 维向量，
#    用来验证 pgvector 的写入和查询链路。
#
# 6. OmlxEmbeddingProvider
#    如果你的本地 OMLX 后续提供真正的 /v1/embeddings 接口，
#    并且加载的是 embedding 模型，就可以切换到它。
#
# 重要：
#   Qwen3-ASR-1.7B-bf16 是语音识别模型，不是向量数据库，也不是 embedding 模型。
#   当前本地向量数据库是 Postgres + pgvector。
#   当前默认向量生成器是 HashEmbeddingProvider。
#
# 最终运行链路大致是：
#
#   MemoryRecord 文本经验
#      -> EmbeddingProvider 生成固定维度向量
#      -> Postgres + pgvector 保存向量
#      -> Memory Retriever 根据当前任务做相似度搜索
#      -> 找回和当前任务最相关的历史经验


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

    现在使用 Pydantic，而不是普通 dict / dataclass。
    好处是：
    - 写入前做运行时校验。
    - 字段类型更明确。
    - 后续可以直接导出 JSON Schema。
    - 更接近生产级 agent 的 memory record 设计。
    """

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
    source: str = "memory_writer_node"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _confidence_between_zero_and_one(cls, value: float) -> float:
        """限制 confidence 在 0 到 1 之间。"""

        if value < 0 or value > 1:
            raise ValueError("confidence 必须在 0 到 1 之间。")
        return value


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


class JsonlMemoryStore:
    """本地 JSONL memory store。"""

    backend_name = "jsonl"

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        return _read_jsonl_memory_records(limit)

    def upsert_record(self, record: MemoryRecord) -> None:
        _upsert_jsonl_memory_record(record.model_dump(mode="json"))

    def search_similar_records(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        return []


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
                    source TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL
                )
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
                CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_tags
                ON beginner_agent_memory USING GIN (tags)
                """
            )
            conn.execute(
                "DROP INDEX IF EXISTS idx_beginner_agent_memory_embeddings_vector"
            )

    def _ensure_embedding_table(self, dimension: int) -> str:
        """确保当前 embedding 维度对应的 pgvector 表存在。

        中文注释：
        pgvector 的 vector 列通常要固定维度，例如 vector(384)、vector(1024)。
        之前本项目只有 384 维 hash embedding，所以只有一张固定表。

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
                       created_at::text, metadata
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
                }
            )
        return records

    def upsert_record(self, record: MemoryRecord) -> None:
        self._ensure_table()
        data = record.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO beginner_agent_memory (
                    id, kind, task_id, title, summary, status, tool_name,
                    tool_result_status, paths, tags, confidence, source,
                    created_at, metadata
                )
                VALUES (
                    %(id)s, %(kind)s, %(task_id)s, %(title)s, %(summary)s,
                    %(status)s, %(tool_name)s, %(tool_result_status)s,
                    %(paths)s::jsonb, %(tags)s::jsonb, %(confidence)s,
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
                       e.embedding <=> %(query_embedding)s::vector AS distance,
                       e.embedding_provider, e.embedding_model
                FROM {table_name} e
                JOIN beginner_agent_memory m ON m.id = e.memory_id
                WHERE e.embedding_provider = %(provider)s
                  AND e.embedding_model = %(model)s
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
                    "vector_distance": float(row[14]),
                    "embedding_provider": row[15],
                    "embedding_model": row[16],
                }
            )
        return records


def memory_record_json_schema() -> dict[str, Any]:
    """导出 MemoryRecord 的 JSON Schema。"""

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


def _configured_store() -> MemoryStore:
    """根据环境变量选择 memory store。"""

    backend = os.getenv("BEGINNER_AGENT_MEMORY_BACKEND", "jsonl").strip().lower()
    if backend == "postgres":
        database_url = os.getenv("DATABASE_URL", "").strip()
        if database_url:
            return PostgresMemoryStore(database_url)
    return JsonlMemoryStore()


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
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_jsonl_memory_records(limit: int) -> list[dict[str, Any]]:
    """从 JSONL 文件读取历史记忆。

    中文注释：
    JSONL 是一行一个 JSON。
    这个格式简单、可追加、方便肉眼查看，适合当前本地版 agent。
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


def _upsert_jsonl_memory_record(record: dict[str, Any]) -> None:
    """插入或更新一条记忆。"""

    records = _read_jsonl_memory_records(MAX_MEMORY_RECORDS)
    record_id = str(record["id"])
    kept = [item for item in records if str(item.get("id", "")) != record_id]
    kept.append(record)
    _write_jsonl_memory_records(kept)


def _list_memory_records() -> tuple[list[dict[str, Any]], str, str]:
    """读取 memory records，并返回 backend 信息。

    中文注释：
    如果配置了 Postgres 但连接失败，不让整个 agent 崩掉。
    它会回退到 JSONL，并把错误原因写进 memory_context。
    """

    store = _configured_store()
    try:
        return store.list_records(MAX_MEMORY_RECORDS), store.backend_name, ""
    except Exception as exc:
        fallback = JsonlMemoryStore()
        return fallback.list_records(MAX_MEMORY_RECORDS), fallback.backend_name, str(exc)


def _search_vector_records(query_text: str) -> tuple[list[dict[str, Any]], str, str]:
    """执行向量检索，并返回 backend 信息。"""

    store = _configured_store()
    try:
        return store.search_similar_records(query_text, MAX_RETRIEVED_RECORDS), store.backend_name, ""
    except Exception as exc:
        return [], store.backend_name, str(exc)


def _upsert_memory_record(record: MemoryRecord) -> tuple[str, str]:
    """写入 memory record，并返回 backend 信息。"""

    store = _configured_store()
    try:
        store.upsert_record(record)
        return store.backend_name, ""
    except Exception as exc:
        fallback = JsonlMemoryStore()
        fallback.upsert_record(record)
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
    tags.update(Path(path).suffix.lstrip(".") for path in _extract_paths(memory) if Path(path).suffix)
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
    paths = sorted(set(_extract_paths({**pending_memory, "args": task.get("args", {})}) + changed_files))
    status = str(task.get("status") or pending_memory.get("decision") or "unknown")
    tool_result_status = str(
        pending_memory.get("tool_result_status")
        or task.get("tool_result_status")
        or state["tool_result_status"]
        or "none"
    )
    summary = (
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
    }
    record_id = _stable_memory_id(raw_record)
    confidence = 0.9 if tool_result_status == "success" else 0.65
    return MemoryRecord(
        id=record_id,
        kind=raw_record["kind"],
        task_id=task_id,
        title=raw_record["title"],
        summary=summary,
        status=status,
        tool_name=tool_name,
        tool_result_status=tool_result_status,
        paths=paths,
        tags=_memory_tags({**pending_memory, "tool_name": tool_name}),
        confidence=confidence,
        metadata={
            "parent_evaluation": pending_memory.get("parent_evaluation", {}),
            "goal_progress": pending_memory.get("goal_progress", {}),
            "tool_result_data": tool_result_data,
            "source_memory": pending_memory,
        },
    )


def _score_record(record: dict[str, Any], state: State) -> int:
    """给一条历史记忆打分，分数越高越相关。

    中文注释：
    这里先用确定性规则，不用 embedding。
    生产级可以把这个函数替换成向量检索 + rerank。
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
    return score


def _retrieve_relevant_records(state: State) -> tuple[list[dict[str, Any]], str, str]:
    """检索和当前目标最相关的历史记忆。"""

    records, backend, backend_error = _list_memory_records()
    query_text = _query_text_for_state(state)
    vector_records, vector_backend, vector_error = _search_vector_records(query_text)
    scored = [(record, _score_record(record, state)) for record in records]
    relevant = [item for item in scored if item[1] > 0]
    relevant.sort(key=lambda item: item[1], reverse=True)
    merged: dict[str, dict[str, Any]] = {}
    for record in vector_records:
        merged[str(record.get("id", ""))] = {
            **record,
            "retrieval_source": "vector",
            "retrieval_score": 1.0 - float(record.get("vector_distance", 1.0)),
        }
    for record, score in relevant:
        record_id = str(record.get("id", ""))
        if record_id in merged:
            merged[record_id]["retrieval_source"] = "hybrid"
            merged[record_id]["rule_score"] = score
            merged[record_id]["retrieval_score"] = float(merged[record_id].get("retrieval_score", 0)) + (
                score / 10
            )
        else:
            merged[record_id] = {
                **record,
                "retrieval_source": "rule",
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
    它会同时读取：
    - 当前 State 里的短期记忆。
    - .agent_state/memory/memory.jsonl 里的持久化记忆。

    这样 agent 下一次运行时，也能看到之前任务沉淀下来的经验。
    """

    state_notes = list(state["memory_notes"])[-5:]
    persisted_records, backend, backend_error = _retrieve_relevant_records(state)
    memory_context = {
        "source": "state.memory_notes + .agent_state/memory/memory.jsonl",
        "backend": backend,
        "backend_error": backend_error,
        "record_schema": memory_record_json_schema(),
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
    """Memory Writer：把本轮任务经验写入轻量记忆和持久化 JSONL。

    中文注释：
    当前是本地生产化的第一步：
    - State.memory_notes：方便你在运行结果里直接看到。
    - memory.jsonl：方便跨运行保留经验。
    - MemoryRecord：让记忆有类型、状态、来源、路径和可信度。

    还没有引入向量库，但已经比“只存一个 list”更接近真实 agent memory。
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
                f"{record.id}，类型={record.kind}，工具={record.tool_name}，backend={backend}。"
            ),
        }
    ]
    return update


def route_after_memory_writer(state: State) -> MemoryWriterRoute:
    """Memory Writer 后的路由。"""

    if state["done"] or state["next_action"] == "finish":
        return "finish"
    return "schedule"
