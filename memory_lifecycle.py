from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .memory import (
    MAX_MEMORY_RECORDS,
    JsonlMemoryStore,
    MemoryAuditEvent,
    MemoryRecord,
    MemoryStore,
    _build_audit_event,
    _configured_store,
    _safe_memory_value,
    _stable_memory_id,
)
from .memory_compaction import compact_memories
from .state_factory import create_initial_state


DEFAULT_LIFECYCLE_SOURCE_LIMIT = 300
DEFAULT_LIFECYCLE_SUMMARY_MIN_GROUP_SIZE = 6
DEFAULT_LIFECYCLE_EMBEDDING_REBUILD_LIMIT = 50
LOW_VALUE_THRESHOLD = 0.28
HIGH_VALUE_THRESHOLD = 0.86


@dataclass(frozen=True)
class LifecycleReport:
    """Memory Lifecycle Job 的结构化报告。

    中文注释：
    生产级后台任务不能只打印一句“成功”。
    它要告诉我们每个维护阶段做了什么：
    - 清理了多少过期记忆。
    - 降权了多少低价值记忆。
    - 提升了多少高价值记忆。
    - 修复了多少 contradiction。
    - 生成了多少 summary memory。
    - 重建了多少 embedding。
    """

    backend: str
    backend_warning: str
    expired_cleaned: int
    low_value_deprioritized: int
    high_value_promoted: int
    contradiction_fixed: int
    summary_created: int
    compaction_report: dict[str, Any]
    embeddings_rebuilt: int
    audit_events_written: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_warning": self.backend_warning,
            "expired_cleaned": self.expired_cleaned,
            "low_value_deprioritized": self.low_value_deprioritized,
            "high_value_promoted": self.high_value_promoted,
            "contradiction_fixed": self.contradiction_fixed,
            "summary_created": self.summary_created,
            "compaction_report": self.compaction_report,
            "embeddings_rebuilt": self.embeddings_rebuilt,
            "audit_events_written": self.audit_events_written,
        }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except ValueError:
        return default


def _store_with_fallback() -> tuple[MemoryStore, str]:
    try:
        return _configured_store(), ""
    except Exception as exc:
        return JsonlMemoryStore(), str(exc)


def _active_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if str(record.get("validity_status", "active")) == "active"
    ]


def _record_score(record: dict[str, Any]) -> float:
    """计算 lifecycle 用的价值分。

    中文注释：
    这里不是模型评分，而是稳定的本地治理规则。
    分数越高，越值得保留或提升；分数越低，越应该降权。
    """

    importance = float(record.get("importance", 0.5))
    quality = float(record.get("quality_score", 0.5))
    trust = float(record.get("trust_score", 0.5))
    decay = float(record.get("decay_score", 0.0))
    pinned_bonus = 0.1 if bool(record.get("pinned", False)) else 0.0
    score = (
        importance * 0.35
        + quality * 0.3
        + trust * 0.3
        - decay * 0.25
        + pinned_bonus
    )
    return max(0.0, min(1.0, score))


def _as_memory_record(record: dict[str, Any], updates: dict[str, Any]) -> MemoryRecord:
    """把 dict record 安全转换回 MemoryRecord。"""

    return MemoryRecord(**{**record, **updates})


def _audit(
    store: MemoryStore,
    *,
    action: str,
    memory_id: str,
    reason: str,
    metadata: dict[str, Any],
) -> int:
    event: MemoryAuditEvent = _build_audit_event(
        action=action,  # type: ignore[arg-type]
        memory_id=memory_id,
        reason=reason,
        backend=store.backend_name,
        metadata=metadata,
    )
    store.upsert_audit_event(event)
    return 1


def _deprioritize_low_value(store: MemoryStore, records: list[dict[str, Any]]) -> tuple[int, int]:
    """定期降权低价值 memory。"""

    changed = 0
    audits = 0
    for record in records:
        if bool(record.get("pinned", False)):
            continue
        score = _record_score(record)
        if score > LOW_VALUE_THRESHOLD:
            continue
        metadata = dict(record.get("metadata") if isinstance(record.get("metadata"), dict) else {})
        metadata["lifecycle"] = {
            **metadata.get("lifecycle", {}),
            "last_action": "deprioritize",
            "score": round(score, 4),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        updated = _as_memory_record(
            record,
            {
                "importance": min(float(record.get("importance", 0.5)), 0.25),
                "validity_status": "deprecated",
                "metadata": metadata,
            },
        )
        store.upsert_record(updated)
        audits += _audit(
            store,
            action="deprioritize",
            memory_id=updated.id,
            reason="Memory Lifecycle 将低价值记忆降权并标记 deprecated。",
            metadata={"score": round(score, 4), "title": updated.title},
        )
        changed += 1
    return changed, audits


def _promote_high_value(store: MemoryStore, records: list[dict[str, Any]]) -> tuple[int, int]:
    """定期提升高价值 memory。"""

    changed = 0
    audits = 0
    for record in records:
        if bool(record.get("pinned", False)):
            continue
        score = _record_score(record)
        if score < HIGH_VALUE_THRESHOLD:
            continue
        metadata = dict(record.get("metadata") if isinstance(record.get("metadata"), dict) else {})
        metadata["lifecycle"] = {
            **metadata.get("lifecycle", {}),
            "last_action": "promote",
            "score": round(score, 4),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        updated = _as_memory_record(
            record,
            {
                "importance": max(float(record.get("importance", 0.5)), 0.9),
                "retention_policy": "long_term",
                "metadata": metadata,
            },
        )
        store.upsert_record(updated)
        audits += _audit(
            store,
            action="promote",
            memory_id=updated.id,
            reason="Memory Lifecycle 将高价值记忆提升为 long_term。",
            metadata={"score": round(score, 4), "title": updated.title},
        )
        changed += 1
    return changed, audits


def _fix_contradictions(store: MemoryStore, records: list[dict[str, Any]]) -> tuple[int, int]:
    """定期检查 contradiction_key，保留最新 active 记忆。"""

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("contradiction_key") or "").strip()
        if key:
            groups.setdefault(key, []).append(record)

    fixed = 0
    audits = 0
    for key, group in groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        winner = group[0]
        loser_ids = [str(item.get("id", "")) for item in group[1:]]
        store.mark_records_status(
            loser_ids,
            "superseded",
            superseded_by=str(winner.get("id", "")),
        )
        audits += _audit(
            store,
            action="contradiction_check",
            memory_id=str(winner.get("id", "")),
            reason="Memory Lifecycle 检查 contradiction_key，并保留最新 active 记忆。",
            metadata={
                "contradiction_key": key,
                "winner": winner.get("id", ""),
                "superseded": loser_ids,
            },
        )
        fixed += len(loser_ids)
    return fixed, audits


def _summary_group_key(record: dict[str, Any]) -> str:
    return ":".join(
        [
            str(record.get("project_id", "")),
            str(record.get("kind", "")),
            str(record.get("tool_name", "")),
            str(record.get("tool_result_status", "")),
        ]
    )


def _create_summary_memories(
    store: MemoryStore,
    records: list[dict[str, Any]],
) -> tuple[int, int]:
    """定期生成 summary memory。

    中文注释：
    compaction 处理“相似记录合并”。
    lifecycle summary 处理“项目阶段总结”：
    当同一类任务累计很多条时，生成一条 project summary，
    让后续检索优先看到总结，而不是一堆碎片。
    """

    min_group_size = _env_int(
        "BEGINNER_AGENT_MEMORY_LIFECYCLE_SUMMARY_MIN_GROUP_SIZE",
        DEFAULT_LIFECYCLE_SUMMARY_MIN_GROUP_SIZE,
        minimum=2,
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if str(record.get("source", "")) in {"memory_compaction_node", "memory_lifecycle_job"}:
            continue
        groups.setdefault(_summary_group_key(record), []).append(record)

    created = 0
    audits = 0
    for key, group in groups.items():
        if len(group) < min_group_size:
            continue
        group.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        sample = group[:10]
        source_ids = [str(item.get("id", "")) for item in sample]
        summary_lines = [
            f"summary_key={key}",
            f"source_count={len(group)}",
            "recent_titles="
            + " | ".join(str(item.get("title", "")) for item in sample)[:1000],
        ]
        raw_record = {
            "kind": "project",
            "task_id": "memory_lifecycle_job",
            "title": f"生命周期项目阶段总结：{key}",
            "summary": "\n".join(summary_lines)[:2000],
            "status": "summarized",
            "tool_name": "none",
            "tool_result_status": "success",
            "paths": sorted({path for item in sample for path in item.get("paths", [])})[:50],
            "scope": "project",
            "visibility": "project",
            "tenant_id": str(sample[0].get("tenant_id", "local-tenant")),
            "workspace_id": str(sample[0].get("workspace_id", "local-workspace")),
            "project_id": str(sample[0].get("project_id", "beginner_agent")),
            "user_id": str(sample[0].get("user_id", "local-user")),
            "contradiction_key": f"lifecycle-summary:{key}",
        }
        record_id = _stable_memory_id(raw_record)
        summary_record = MemoryRecord(
            id=record_id,
            kind="project",
            task_id="memory_lifecycle_job",
            title=str(raw_record["title"]),
            summary=str(raw_record["summary"]),
            status="summarized",
            tool_name="none",
            tool_result_status="success",
            paths=raw_record["paths"],
            tags=["lifecycle_summary", f"source_count:{len(group)}"],
            confidence=0.82,
            importance=0.84,
            quality_score=0.78,
            trust_score=0.78,
            decay_score=0.02,
            scope="project",
            visibility="project",
            sensitivity_level="internal",
            tenant_id=str(raw_record["tenant_id"]),
            workspace_id=str(raw_record["workspace_id"]),
            project_id=str(raw_record["project_id"]),
            user_id=str(raw_record["user_id"]),
            retention_policy="long_term",
            validity_status="active",
            pinned=False,
            expires_at=None,
            supersedes=source_ids[0] if source_ids else None,
            contradiction_key=str(raw_record["contradiction_key"]),
            source="memory_lifecycle_job",
            metadata={
                "lifecycle_summary": {
                    "key": key,
                    "source_count": len(group),
                    "source_memory_ids": source_ids,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                "source_records": _safe_memory_value(sample),
            },
        )
        store.upsert_record(summary_record)
        audits += _audit(
            store,
            action="summarize",
            memory_id=summary_record.id,
            reason="Memory Lifecycle 生成项目阶段 summary memory。",
            metadata=summary_record.metadata["lifecycle_summary"],
        )
        created += 1
    return created, audits


def run_memory_lifecycle_job() -> LifecycleReport:
    """运行一次完整 Memory Lifecycle Job。

    中文注释：
    这就是“定期后台任务”的核心函数。
    后续可以被：
    - 本地 CLI。
    - cron。
    - Kubernetes CronJob。
    - Prefect/Celery worker。
    - FastAPI admin endpoint。
    统一调用。
    """

    store, setup_error = _store_with_fallback()
    audit_count = 0

    expired_cleaned = store.cleanup_expired_records()
    if expired_cleaned:
        audit_count += _audit(
            store,
            action="expire",
            memory_id="memory_lifecycle_job",
            reason="Memory Lifecycle 清理过期非 pinned 记忆。",
            metadata={"expired_cleaned": expired_cleaned},
        )

    source_limit = _env_int(
        "BEGINNER_AGENT_MEMORY_LIFECYCLE_SOURCE_LIMIT",
        DEFAULT_LIFECYCLE_SOURCE_LIMIT,
    )
    records = _active_records(store.list_records(min(MAX_MEMORY_RECORDS, source_limit)))

    low_value_count, low_value_audits = _deprioritize_low_value(store, records)
    high_value_count, high_value_audits = _promote_high_value(store, records)
    contradiction_count, contradiction_audits = _fix_contradictions(store, records)
    summary_count, summary_audits = _create_summary_memories(store, records)
    audit_count += (
        low_value_audits
        + high_value_audits
        + contradiction_audits
        + summary_audits
    )

    compaction_report: dict[str, Any] = {}
    if _env_bool("BEGINNER_AGENT_MEMORY_LIFECYCLE_RUN_COMPACTION", True):
        state = create_initial_state("memory lifecycle compaction")
        state["current_task_id"] = "memory_lifecycle_job"
        compaction_report = compact_memories(state)

    embeddings_rebuilt = 0
    if _env_bool("BEGINNER_AGENT_MEMORY_LIFECYCLE_REBUILD_EMBEDDINGS", True):
        rebuild_limit = _env_int(
            "BEGINNER_AGENT_MEMORY_LIFECYCLE_EMBEDDING_REBUILD_LIMIT",
            DEFAULT_LIFECYCLE_EMBEDDING_REBUILD_LIMIT,
        )
        embeddings_rebuilt = store.rebuild_embeddings(rebuild_limit)
        if embeddings_rebuilt:
            audit_count += _audit(
                store,
                action="rebuild_embedding",
                memory_id="memory_lifecycle_job",
                reason="Memory Lifecycle 定期重建 memory embeddings。",
                metadata={"embeddings_rebuilt": embeddings_rebuilt},
            )

    return LifecycleReport(
        backend=store.backend_name,
        backend_warning=setup_error,
        expired_cleaned=expired_cleaned,
        low_value_deprioritized=low_value_count,
        high_value_promoted=high_value_count,
        contradiction_fixed=contradiction_count,
        summary_created=summary_count,
        compaction_report=compaction_report,
        embeddings_rebuilt=embeddings_rebuilt,
        audit_events_written=audit_count,
    )


def memory_lifecycle_report_json() -> str:
    """运行 lifecycle job 并输出 JSON，方便脚本入口复用。"""

    return json.dumps(
        run_memory_lifecycle_job().as_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
