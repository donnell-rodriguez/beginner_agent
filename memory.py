from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

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


class JsonlMemoryStore:
    """本地 JSONL memory store。"""

    backend_name = "jsonl"

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        return _read_jsonl_memory_records(limit)

    def upsert_record(self, record: MemoryRecord) -> None:
        _upsert_jsonl_memory_record(record.model_dump(mode="json"))


class PostgresMemoryStore:
    """Postgres memory store。

    中文注释：
    启用方式：

        BEGINNER_AGENT_MEMORY_BACKEND=postgres
        DATABASE_URL=postgresql://...

    这里使用延迟 import psycopg。
    如果你的环境没有安装 psycopg，默认 JSONL 路径不受影响。
    """

    backend_name = "postgres"

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
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


def memory_record_json_schema() -> dict[str, Any]:
    """导出 MemoryRecord 的 JSON Schema。"""

    return MemoryRecord.model_json_schema()


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
    scored = [(record, _score_record(record, state)) for record in records]
    relevant = [item for item in scored if item[1] > 0]
    relevant.sort(key=lambda item: item[1], reverse=True)
    return [record for record, _score in relevant[:MAX_RETRIEVED_RECORDS]], backend, backend_error


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
