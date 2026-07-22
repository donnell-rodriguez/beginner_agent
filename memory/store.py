from __future__ import annotations

import os
from typing import Any

from .failure import build_failure_memory_profile
from .audit import _build_audit_event
from .jsonl_store import JsonlMemoryStore
from .models import MemoryAuditEvent, MemoryRecord, MemoryStore
from .policy import _records_share_memory_pattern
from .postgres_store import PostgresMemoryStore
from .quality import MemoryEvaluator, adjusted_memory_fields
from .settings import (
    DEFAULT_MEMORY_BACKEND,
    MAX_MEMORY_RECORDS,
    MEMORY_PROMOTION_SUCCESS_THRESHOLD,
)

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

    evaluation = MemoryEvaluator().evaluate(
        record.model_dump(mode="json"),
        existing_records,
    )
    quality_updates = adjusted_memory_fields(
        record.model_dump(mode="json"),
        evaluation,
    )
    metadata_updates["memory_quality_evaluation"] = evaluation.as_dict()
    metadata_updates["memory_quality_evaluation"]["applied_updates"] = {
        key: value
        for key, value in quality_updates.items()
        if key in {"quality_score", "trust_score", "decay_score"}
    }
    updates.update(quality_updates)
    updates["metadata"] = metadata_updates
    events.append(
        _build_audit_event(
            action="store" if evaluation.quality.decision != "reject" else "discard",
            memory_id=record.id,
            reason=(
                "MemoryEvaluator 完成质量评分："
                f"quality={evaluation.quality.overall}, "
                f"trust={evaluation.trust.trust_score}, "
                f"decay={evaluation.decay.decay_score}, "
                f"decision={evaluation.quality.decision}。"
            ),
            backend=backend,
            metadata=evaluation.as_dict(),
        )
    )

    failure_profile = build_failure_memory_profile(
        {
            **record.model_dump(mode="json"),
            "metadata": metadata_updates,
        },
        existing_records,
    )
    if failure_profile:
        profile = failure_profile.as_dict()
        metadata_updates["failure_memory"] = profile
        updates["metadata"] = metadata_updates
        if failure_profile.retry_class == "repair_required":
            updates["importance"] = max(float(updates.get("importance", record.importance)), 0.78)
        if failure_profile.successful_repair_memory_ids:
            updates["importance"] = max(float(updates.get("importance", record.importance)), 0.88)
            metadata_updates["failure_memory"]["has_known_successful_repair"] = True
        events.append(
            _build_audit_event(
                action="store",
                memory_id=record.id,
                reason=(
                    "Failure Memory Library 记录失败模式："
                    f"category={failure_profile.category}, "
                    f"owner={failure_profile.owner}, "
                    f"retry_class={failure_profile.retry_class}。"
                ),
                backend=backend,
                metadata=profile,
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

    return record.model_copy(update=updates), events


def _upsert_memory_record(record: MemoryRecord) -> tuple[str, str, MemoryRecord]:
    """写入 memory record，并返回 backend 信息。"""

    def with_run_id(event: MemoryAuditEvent, run_id: str) -> MemoryAuditEvent:
        if not run_id:
            return event
        return event.model_copy(
            update={"metadata": {**event.metadata, "run_id": run_id}}
        )

    try:
        store = _configured_store()
        existing_records = store.list_records(MAX_MEMORY_RECORDS)
        governed_record, governance_events = _govern_memory_record_before_write(
            record,
            existing_records,
            backend=store.backend_name,
        )
        store.upsert_record(governed_record)
        run_id = str(governed_record.metadata.get("run_id", ""))
        for event in governance_events:
            store.upsert_audit_event(with_run_id(event, run_id))
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
                    "run_id": governed_record.metadata.get("run_id", ""),
                },
            )
        )
        return store.backend_name, "", governed_record
    except Exception as exc:
        fallback = JsonlMemoryStore()
        existing_records = fallback.list_records(MAX_MEMORY_RECORDS)
        governed_record, governance_events = _govern_memory_record_before_write(
            record,
            existing_records,
            backend=fallback.backend_name,
        )
        fallback.upsert_record(governed_record)
        run_id = str(governed_record.metadata.get("run_id", ""))
        for event in governance_events:
            fallback.upsert_audit_event(with_run_id(event, run_id))
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
                    "run_id": governed_record.metadata.get("run_id", ""),
                },
            )
        )
        return fallback.backend_name, str(exc), governed_record


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
