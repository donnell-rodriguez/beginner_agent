from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from beginner_agent.memory.audit import _build_audit_event
from beginner_agent.memory.eval_cases import read_memory_eval_cases
from beginner_agent.memory.effectiveness import read_memory_usage, summarize_memory_usage
from beginner_agent.memory.feedback import feedback_summary_for_memory, read_memory_feedback
from beginner_agent.memory.jsonl_store import JsonlMemoryStore, _read_jsonl_audit_events
from beginner_agent.memory.observability_sinks import read_memory_observability_events
from beginner_agent.memory.online_eval import read_online_eval_events, summarize_online_eval
from beginner_agent.memory.postgres_performance import memory_postgres_governance_report
from beginner_agent.memory.postgres_store import PostgresMemoryStore
from beginner_agent.memory.rerank_observability import (
    read_rerank_telemetry,
    summarize_rerank_telemetry,
)
from beginner_agent.memory.settings import MAX_MEMORY_AUDIT_EVENTS, MAX_MEMORY_RECORDS
from beginner_agent.memory.store import _configured_store, _upsert_memory_audit_event
from beginner_agent.run_lineage import lineage_for_run_id

from .models import AuditQuery, MemoryQuery, PageInfo
from .security import ApiRequestContext, context_metadata


SENSITIVE_REDACTION = {
    "redacted": True,
    "reason": "include_sensitive=false，敏感 metadata 默认不通过 API 展示。",
}


@dataclass(frozen=True)
class RepositoryResult:
    """Repository 统一返回结构。

    中文注释：
    以前仓库方法返回 tuple，字段多了以后很容易搞错顺序。
    现在统一返回 data/backend/error/page，更接近生产代码里的 QueryResult。
    """

    data: Any
    backend: str
    error: str = ""
    page: PageInfo | None = None


def _decode_cursor(cursor: str | None) -> int:
    """把 cursor 解码成 offset。"""

    if not cursor:
        return 0
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8")
        if not raw.startswith("offset:"):
            return 0
        return max(0, int(raw.split(":", 1)[1]))
    except Exception:
        return 0


def _encode_cursor(offset: int) -> str:
    raw = f"offset:{offset}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _paginate(
    items: list[dict[str, Any]],
    *,
    limit: int,
    cursor: str | None,
) -> tuple[list[dict[str, Any]], PageInfo]:
    """对内存结果做 cursor 分页。

    中文注释：
    当前仓库仍会从 store 取一批记录再过滤。
    所以这里用 offset cursor。
    后续如果全部迁到 Postgres 查询，可以升级成 created_at + id cursor。
    """

    offset = _decode_cursor(cursor)
    page_items = items[offset : offset + limit]
    next_offset = offset + len(page_items)
    next_cursor = _encode_cursor(next_offset) if next_offset < len(items) else ""
    return page_items, PageInfo(limit=limit, cursor=cursor or "", next_cursor=next_cursor)


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _failure_memory(record: dict[str, Any]) -> dict[str, Any]:
    failure = _metadata(record).get("failure_memory")
    return failure if isinstance(failure, dict) else {}


def _safe_record(record: dict[str, Any], *, include_sensitive: bool) -> dict[str, Any]:
    """根据敏感级别裁剪 API 返回。

    中文注释：
    Memory Query API 是本地 Admin 查询入口。
    但默认仍不直接展示 confidential / secret metadata。
    真正需要排障时，调用方必须显式 include_sensitive=true。
    """

    sensitivity = str(record.get("sensitivity_level", "internal"))
    if include_sensitive or sensitivity in {"public", "internal"}:
        return record
    return {**record, "metadata": SENSITIVE_REDACTION}


def _safe_event(event: dict[str, Any], *, include_sensitive: bool) -> dict[str, Any]:
    if include_sensitive:
        return event
    metadata = event.get("metadata")
    if isinstance(metadata, dict) and any(
        key in metadata for key in ("tool_result_data", "source_memory", "pending_memory")
    ):
        return {**event, "metadata": SENSITIVE_REDACTION}
    return event


def _record_visible_to_api_context(record: dict[str, Any], context: ApiRequestContext) -> bool:
    """判断当前 API principal 是否可以看到这条 memory。

    中文注释：
    这就是 tenant isolation middleware 之后真正落到数据层的隔离规则。
    middleware/依赖负责识别“你是谁”，仓库层负责判断“这条记录你能不能看”。
    """

    visibility = str(record.get("visibility", "project"))
    principal = context.principal
    if visibility == "public":
        return True
    if visibility == "tenant":
        return str(record.get("tenant_id", "")) == principal.tenant_id
    if visibility == "workspace":
        return (
            str(record.get("tenant_id", "")) == principal.tenant_id
            and str(record.get("workspace_id", "")) == principal.workspace_id
        )
    if visibility in {"project", "retrieval_only"}:
        return (
            str(record.get("tenant_id", "")) == principal.tenant_id
            and str(record.get("workspace_id", "")) == principal.workspace_id
            and str(record.get("project_id", "")) == principal.project_id
        )
    if visibility == "private":
        return (
            str(record.get("tenant_id", "")) == principal.tenant_id
            and str(record.get("workspace_id", "")) == principal.workspace_id
            and str(record.get("project_id", "")) == principal.project_id
            and str(record.get("user_id", "")) == principal.user_id
        )
    return False


def _audit_sensitive_api_access(
    record: dict[str, Any],
    *,
    include_sensitive: bool,
    context: ApiRequestContext,
) -> None:
    """记录 Memory Query API 对敏感记忆的访问。

    中文注释：
    查询 API 是给人/后台系统看的。
    如果调用方显式 include_sensitive=true，必须留下审计。
    这里仍然不写原始敏感内容，只记录 memory id、actor 和敏感级别。
    """

    sensitivity = str(record.get("sensitivity_level", "internal"))
    if sensitivity in {"public", "internal"}:
        return
    _upsert_memory_audit_event(
        _build_audit_event(
            action="sensitive_access",
            memory_id=str(record.get("id", "")),
            reason="Memory Query API 访问敏感记忆。",
            backend="memory_api",
            metadata={
                **context_metadata(context),
                "include_sensitive": include_sensitive,
                "sensitivity_level": sensitivity,
                "visibility": record.get("visibility", ""),
                "source": "memory_query_api",
            },
        )
    )


def _paths(record: dict[str, Any]) -> set[str]:
    paths = record.get("paths", [])
    if not isinstance(paths, list):
        return set()
    return {str(path) for path in paths}


def _matches_memory_query(record: dict[str, Any], query: MemoryQuery) -> bool:
    if query.kind and str(record.get("kind", "")) != query.kind:
        return False
    if query.task_id and str(record.get("task_id", "")) != query.task_id:
        return False
    if query.tool_name and str(record.get("tool_name", "")) != query.tool_name:
        return False
    if (
        query.contradiction_key
        and str(record.get("contradiction_key", "")) != query.contradiction_key
    ):
        return False
    if query.pinned is not None and bool(record.get("pinned", False)) is not query.pinned:
        return False
    if query.file_path and query.file_path not in _paths(record):
        return False
    failure = _failure_memory(record)
    if query.failure_category and str(failure.get("category", "")) != query.failure_category:
        return False
    if query.failure_pattern_id and str(failure.get("pattern_id", "")) != query.failure_pattern_id:
        return False
    return True


def _matches_audit_query(event: dict[str, Any], query: AuditQuery) -> bool:
    if query.memory_id and str(event.get("memory_id", "")) != query.memory_id:
        return False
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if query.run_id and str(metadata.get("run_id", "")) != query.run_id:
        return False
    if query.action and str(event.get("action", "")) != query.action:
        return False
    return True


def _audit_event_visible_to_context(event: dict[str, Any], context: ApiRequestContext) -> bool:
    """判断审计事件是否属于当前 principal 的可见范围。"""

    if "admin" in context.principal.roles:
        return True
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        return True

    tenant_id = str(metadata.get("tenant_id", "")).strip()
    workspace_id = str(metadata.get("workspace_id", "")).strip()
    project_id = str(metadata.get("project_id", "")).strip()
    if tenant_id and tenant_id != context.principal.tenant_id:
        return False
    if workspace_id and workspace_id != context.principal.workspace_id:
        return False
    if project_id and project_id != context.principal.project_id:
        return False
    return True


class MemoryQueryRepository:
    """Memory Query Repository。

    中文注释：
    FastAPI 层只负责 HTTP。
    真正查询 memory / audit 的逻辑放在这里，方便以后替换成：
    - 只查 Postgres 的生产实现。
    - 加权限校验的企业实现。
    - 接 dashboard 的分页查询实现。
    """

    def list_memories(
        self,
        query: MemoryQuery,
        context: ApiRequestContext,
    ) -> RepositoryResult:
        try:
            store = _configured_store()
            records = store.list_records(MAX_MEMORY_RECORDS)
            backend = store.backend_name
            error = ""
        except Exception as exc:
            store = JsonlMemoryStore()
            records = store.list_records(MAX_MEMORY_RECORDS)
            backend = store.backend_name
            error = str(exc)

        visible_records = [
            record
            for record in records
            if _record_visible_to_api_context(record, context)
            and _matches_memory_query(record, query)
        ]
        for record in records:
            if _record_visible_to_api_context(record, context) and _matches_memory_query(
                record, query
            ):
                _audit_sensitive_api_access(
                    record,
                    include_sensitive=query.include_sensitive,
                    context=context,
                )
        filtered = [
            _safe_record(record, include_sensitive=query.include_sensitive)
            for record in visible_records
        ]
        page_items, page = _paginate(filtered, limit=query.limit, cursor=query.cursor)
        return RepositoryResult(page_items, backend, error, page)

    def get_memory(
        self,
        memory_id: str,
        *,
        include_sensitive: bool,
        context: ApiRequestContext,
    ) -> RepositoryResult:
        result = self.list_memories(
            MemoryQuery(limit=MAX_MEMORY_RECORDS, include_sensitive=include_sensitive),
            context,
        )
        for record in result.data:
            if str(record.get("id", "")) == memory_id:
                return RepositoryResult(record, result.backend, result.error)
        return RepositoryResult(None, result.backend, result.error)

    def list_audit_events(
        self,
        query: AuditQuery,
        context: ApiRequestContext,
    ) -> RepositoryResult:
        try:
            store = _configured_store()
            if isinstance(store, PostgresMemoryStore):
                store.list_records(1)
                events = self._postgres_audit_events(store, query.limit)
                backend = "postgres"
            else:
                events = _read_jsonl_audit_events(MAX_MEMORY_AUDIT_EVENTS)
                backend = store.backend_name
            error = ""
        except Exception as exc:
            events = _read_jsonl_audit_events(MAX_MEMORY_AUDIT_EVENTS)
            backend = "jsonl"
            error = str(exc)

        filtered = [
            _safe_event(event, include_sensitive=query.include_sensitive)
            for event in events
            if _matches_audit_query(event, query)
            and _audit_event_visible_to_context(event, context)
        ]
        page_items, page = _paginate(filtered, limit=query.limit, cursor=query.cursor)
        return RepositoryResult(page_items, backend, error, page)

    def _postgres_audit_events(
        self,
        store: PostgresMemoryStore,
        limit: int,
    ) -> list[dict[str, Any]]:
        with store._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, action, memory_id, reason, backend,
                       created_at::text, metadata
                FROM beginner_agent_memory_audit
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            metadata = row[6]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            events.append(
                {
                    "id": row[0],
                    "action": row[1],
                    "memory_id": row[2],
                    "reason": row[3],
                    "backend": row[4],
                    "created_at": row[5],
                    "metadata": metadata,
                }
            )
        return events

    def why_saved(
        self,
        memory_id: str,
        *,
        include_sensitive: bool,
        context: ApiRequestContext,
    ) -> RepositoryResult:
        record_result = self.get_memory(
            memory_id,
            include_sensitive=include_sensitive,
            context=context,
        )
        audit_result = self.list_audit_events(
            AuditQuery(
                memory_id=memory_id,
                include_sensitive=include_sensitive,
                limit=200,
            ),
            context,
        )
        record = record_result.data
        if record is None:
            return RepositoryResult(
                None,
                record_result.backend,
                record_result.error or audit_result.error,
            )
        metadata = _metadata(record)
        return RepositoryResult(
            {
                "memory": record,
                "memory_policy": metadata.get("memory_policy", {}),
                "quality": metadata.get("memory_quality_evaluation", {}),
                "failure_memory": metadata.get("failure_memory", {}),
                "preference_memory": metadata.get("preference_memory", {}),
                "audit_events": audit_result.data,
                "audit_backend": audit_result.backend,
            },
            record_result.backend,
            record_result.error or audit_result.error,
        )

    def usage(
        self,
        memory_id: str,
        *,
        include_sensitive: bool,
        context: ApiRequestContext,
    ) -> RepositoryResult:
        result = self.list_audit_events(
            AuditQuery(limit=MAX_MEMORY_AUDIT_EVENTS, include_sensitive=include_sensitive),
            context,
        )
        events = result.data
        used_by = [
            event
            for event in events
            if memory_id in json.dumps(event.get("metadata", {}), ensure_ascii=False)
        ]
        direct_events = [
            event for event in events if str(event.get("memory_id", "")) == memory_id
        ]
        return RepositoryResult(
            {
                "memory_id": memory_id,
                "direct_events": direct_events,
                "used_by_events": used_by,
                "effectiveness": summarize_memory_usage(memory_id),
            },
            result.backend,
            result.error,
        )

    def usage_effectiveness(
        self,
        *,
        memory_id: str = "",
        limit: int,
    ) -> RepositoryResult:
        """查询 memory 使用效果闭环。"""

        events = read_memory_usage(limit)
        if memory_id:
            events = [event for event in events if str(event.get("memory_id", "")) == memory_id]
        return RepositoryResult(
            {
                "summary": summarize_memory_usage(memory_id, limit=limit),
                "events": events[-limit:],
            },
            "jsonl-memory-usage",
        )

    def feedback(
        self,
        memory_id: str | None,
        *,
        limit: int,
    ) -> RepositoryResult:
        """查询人工/系统反馈。

        中文注释：
        feedback 是 memory 质量闭环的一部分。
        它回答“这条记忆后来是否被证明有用/有害”。
        """

        events = read_memory_feedback(limit)
        if memory_id:
            events = [
                event
                for event in events
                if str(event.get("memory_id", "")) == memory_id
            ]
        summary = feedback_summary_for_memory(memory_id) if memory_id else {}
        return RepositoryResult(
            {
                "memory_id": memory_id or "",
                "summary": summary,
                "events": events[:limit],
            },
            "jsonl-feedback",
        )

    def eval_cases(self, *, limit: int) -> RepositoryResult:
        """查询离线 memory eval cases。"""

        return RepositoryResult(read_memory_eval_cases(limit), "jsonl-eval-cases")

    def rerank_telemetry(self, *, limit: int) -> RepositoryResult:
        """查询 rerank telemetry 和 A/B bucket 统计。"""

        return RepositoryResult(
            {
                "summary": summarize_rerank_telemetry(limit),
                "events": read_rerank_telemetry(limit),
            },
            "jsonl-rerank-telemetry",
        )

    def online_eval(self, *, limit: int) -> RepositoryResult:
        """查询 retrieval online eval。"""

        return RepositoryResult(
            {
                "summary": summarize_online_eval(limit),
                "events": read_online_eval_events(limit),
            },
            "jsonl-memory-online-eval",
        )

    def memory_observability(self, *, limit: int) -> RepositoryResult:
        """查询 memory observability 本地事件。"""

        return RepositoryResult(
            read_memory_observability_events(limit),
            "jsonl-memory-observability",
        )

    def postgres_governance(self) -> RepositoryResult:
        """查询 Postgres memory schema / performance governance。"""

        import os

        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            return RepositoryResult(
                None,
                "postgres",
                "DATABASE_URL 未配置。",
            )
        try:
            return RepositoryResult(
                memory_postgres_governance_report(database_url),
                "postgres",
            )
        except Exception as exc:
            return RepositoryResult(None, "postgres", str(exc))

    def contradiction_evolution(
        self,
        contradiction_key: str,
        *,
        include_sensitive: bool,
        context: ApiRequestContext,
    ) -> RepositoryResult:
        result = self.list_memories(
            MemoryQuery(
                contradiction_key=contradiction_key,
                include_sensitive=include_sensitive,
                limit=MAX_MEMORY_RECORDS,
            ),
            context,
        )
        records = result.data
        records.sort(key=lambda record: str(record.get("created_at", "")))
        return RepositoryResult(records, result.backend, result.error)

    def run_lineage(self, run_id: str) -> RepositoryResult:
        """查询某次 run 的 lineage 报告。"""

        data = lineage_for_run_id(run_id)
        audit = data.get("audit", {}) if isinstance(data, dict) else {}
        return RepositoryResult(
            data,
            str(audit.get("backend", "")),
            str(audit.get("backend_error", "")),
        )

    def failure_patterns(
        self,
        *,
        limit: int,
        category: str | None,
        pattern_id: str | None,
        context: ApiRequestContext,
    ) -> RepositoryResult:
        """按失败模式聚合 failure memory。

        中文注释：
        真实 code agent 不只是“保存失败记录”，还要能看出：
        - 哪类失败反复出现。
        - 哪些失败不可重试。
        - 哪些失败后续有成功修复路径。
        所以这里把底层 memory records 聚合成 dashboard 更容易展示的 pattern。
        """

        result = self.list_memories(
            MemoryQuery(
                limit=MAX_MEMORY_RECORDS,
                failure_category=category,
                failure_pattern_id=pattern_id,
            ),
            context,
        )
        records = result.data
        patterns: dict[str, dict[str, Any]] = {}
        for record in records:
            failure = _failure_memory(record)
            current_pattern_id = str(failure.get("pattern_id", "")).strip()
            if not current_pattern_id:
                current_pattern_id = str(record.get("contradiction_key", "")).strip()
            if not current_pattern_id:
                current_pattern_id = str(record.get("id", "unknown"))

            pattern = patterns.setdefault(
                current_pattern_id,
                {
                    "pattern_id": current_pattern_id,
                    "category": failure.get("category", ""),
                    "count": 0,
                    "latest_at": "",
                    "memory_ids": [],
                    "sample_titles": [],
                    "non_retryable_count": 0,
                    "successful_repair_paths": [],
                },
            )
            pattern["count"] += 1
            pattern["memory_ids"].append(record.get("id", ""))
            if title := str(record.get("title", "")).strip():
                pattern["sample_titles"].append(title)
            if failure.get("retryable") is False:
                pattern["non_retryable_count"] += 1
            repair_path = failure.get("successful_repair_path")
            if repair_path:
                pattern["successful_repair_paths"].append(repair_path)
            pattern["latest_at"] = max(
                str(pattern["latest_at"]),
                str(record.get("created_at", "")),
            )

        grouped = sorted(
            patterns.values(),
            key=lambda item: (int(item["count"]), str(item["latest_at"])),
            reverse=True,
        )
        return RepositoryResult(grouped[:limit], result.backend, result.error)
