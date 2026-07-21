from __future__ import annotations

import json
from typing import Any

from beginner_agent.memory import (
    MAX_MEMORY_AUDIT_EVENTS,
    MAX_MEMORY_RECORDS,
    JsonlMemoryStore,
    PostgresMemoryStore,
    _configured_store,
    _read_jsonl_audit_events,
)

from .models import AuditQuery, MemoryQuery


SENSITIVE_REDACTION = {
    "redacted": True,
    "reason": "include_sensitive=false，敏感 metadata 默认不通过 API 展示。",
}


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
    if query.action and str(event.get("action", "")) != query.action:
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

    def list_memories(self, query: MemoryQuery) -> tuple[list[dict[str, Any]], str, str]:
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

        filtered = [
            _safe_record(record, include_sensitive=query.include_sensitive)
            for record in records
            if _matches_memory_query(record, query)
        ]
        return filtered[: query.limit], backend, error

    def get_memory(
        self,
        memory_id: str,
        *,
        include_sensitive: bool,
    ) -> tuple[dict[str, Any] | None, str, str]:
        records, backend, error = self.list_memories(
            MemoryQuery(limit=MAX_MEMORY_RECORDS, include_sensitive=include_sensitive)
        )
        for record in records:
            if str(record.get("id", "")) == memory_id:
                return record, backend, error
        return None, backend, error

    def list_audit_events(
        self,
        query: AuditQuery,
    ) -> tuple[list[dict[str, Any]], str, str]:
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
        ]
        return filtered[: query.limit], backend, error

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
    ) -> tuple[dict[str, Any] | None, str, str]:
        record, backend, error = self.get_memory(
            memory_id,
            include_sensitive=include_sensitive,
        )
        events, audit_backend, audit_error = self.list_audit_events(
            AuditQuery(
                memory_id=memory_id,
                include_sensitive=include_sensitive,
                limit=200,
            )
        )
        if record is None:
            return None, backend, error or audit_error
        metadata = _metadata(record)
        return (
            {
                "memory": record,
                "memory_policy": metadata.get("memory_policy", {}),
                "quality": metadata.get("memory_quality_evaluation", {}),
                "failure_memory": metadata.get("failure_memory", {}),
                "preference_memory": metadata.get("preference_memory", {}),
                "audit_events": events,
                "audit_backend": audit_backend,
            },
            backend,
            error or audit_error,
        )

    def usage(
        self,
        memory_id: str,
        *,
        include_sensitive: bool,
    ) -> tuple[dict[str, Any], str, str]:
        events, backend, error = self.list_audit_events(
            AuditQuery(limit=MAX_MEMORY_AUDIT_EVENTS, include_sensitive=include_sensitive)
        )
        used_by = [
            event
            for event in events
            if memory_id in json.dumps(event.get("metadata", {}), ensure_ascii=False)
        ]
        direct_events = [
            event for event in events if str(event.get("memory_id", "")) == memory_id
        ]
        return (
            {
                "memory_id": memory_id,
                "direct_events": direct_events,
                "used_by_events": used_by,
            },
            backend,
            error,
        )

    def contradiction_evolution(
        self,
        contradiction_key: str,
        *,
        include_sensitive: bool,
    ) -> tuple[list[dict[str, Any]], str, str]:
        records, backend, error = self.list_memories(
            MemoryQuery(
                contradiction_key=contradiction_key,
                include_sensitive=include_sensitive,
                limit=MAX_MEMORY_RECORDS,
            )
        )
        records.sort(key=lambda record: str(record.get("created_at", "")))
        return records, backend, error

    def failure_patterns(
        self,
        *,
        limit: int,
        category: str | None,
        pattern_id: str | None,
    ) -> tuple[list[dict[str, Any]], str, str]:
        """按失败模式聚合 failure memory。

        中文注释：
        真实 code agent 不只是“保存失败记录”，还要能看出：
        - 哪类失败反复出现。
        - 哪些失败不可重试。
        - 哪些失败后续有成功修复路径。
        所以这里把底层 memory records 聚合成 dashboard 更容易展示的 pattern。
        """

        records, backend, error = self.list_memories(
            MemoryQuery(
                limit=MAX_MEMORY_RECORDS,
                failure_category=category,
                failure_pattern_id=pattern_id,
            )
        )
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
        return grouped[:limit], backend, error
