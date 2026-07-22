from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .models import MemoryAuditEvent, MemoryRecord, ValidityStatus
from .policy import _record_should_be_deleted
from .settings import (
    MAX_MEMORY_AUDIT_EVENTS,
    MAX_MEMORY_RECORDS,
    MEMORY_AUDIT_FILE,
    MEMORY_DIR,
    MEMORY_FILE,
)
from ..tooling.core import ensure_state_dirs

def _ensure_memory_file() -> None:
    """确保 memory 存储文件存在。"""

    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("", encoding="utf-8")

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


def _mark_jsonl_records_status(
    memory_ids: list[str],
    status: ValidityStatus,
    *,
    superseded_by: str | None = None,
) -> None:
    """批量更新 JSONL fallback 里的记忆状态。"""

    if not memory_ids:
        return
    memory_id_set = set(memory_ids)
    records = _read_jsonl_memory_records(MAX_MEMORY_RECORDS)
    updated_at = datetime.now(timezone.utc).isoformat()
    for record in records:
        if str(record.get("id", "")) not in memory_id_set:
            continue
        record["validity_status"] = status
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        compaction = metadata.get("compaction")
        if not isinstance(compaction, dict):
            compaction = {}
        compaction.update(
            {
                "superseded_by": superseded_by,
                "updated_at": updated_at,
            }
        )
        metadata["compaction"] = compaction
        record["metadata"] = metadata
    _write_jsonl_memory_records(records)

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

    def mark_records_status(
        self,
        memory_ids: list[str],
        status: ValidityStatus,
        *,
        superseded_by: str | None = None,
    ) -> None:
        _mark_jsonl_records_status(
            memory_ids,
            status,
            superseded_by=superseded_by,
        )

    def cleanup_expired_records(self) -> int:
        records = _read_jsonl_memory_records(MAX_MEMORY_RECORDS)
        kept = [record for record in records if not _record_should_be_deleted(record)]
        deleted = len(records) - len(kept)
        if deleted:
            _write_jsonl_memory_records(kept)
        return deleted

    def rebuild_embeddings(self, limit: int) -> int:
        return 0

    def search_similar_records(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        return []

    def upsert_audit_event(self, event: MemoryAuditEvent) -> None:
        _upsert_jsonl_audit_event(event.model_dump(mode="json"))
