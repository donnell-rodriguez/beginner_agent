from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .audit import _build_audit_event
from .jsonl_store import JsonlMemoryStore
from .models import (
    MemoryAuditEvent,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    MemoryStore,
    SensitivityLevel,
)
from .policy import _memory_access_context, _safe_memory_value, _stable_memory_id
from .settings import DEFAULT_PROJECT_ID, MAX_MEMORY_RECORDS
from .store import _configured_store
from ..state import State


CompactionRoute = Literal["schedule", "finish"]
CompactionGroupType = Literal["failure_pattern", "file_memory", "project_memory"]

DEFAULT_COMPACTION_MIN_GROUP_SIZE = 4
DEFAULT_COMPACTION_MAX_GROUPS = 3
DEFAULT_COMPACTION_SOURCE_LIMIT = 200


@dataclass(frozen=True)
class CompactionCandidate:
    """一组可以被压缩的 memory records。

    中文注释：
    memory compaction 不是把所有记忆粗暴合并。
    它要先找到“确实属于同一主题”的记录组，例如：
    - 同一个失败模式。
    - 同一个文件的多次修改/阅读。
    - 同一个项目阶段的多次执行经验。
    """

    group_type: CompactionGroupType
    key: str
    title: str
    records: list[dict[str, Any]]
    score: float


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """读取整数环境变量。"""

    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except ValueError:
        return default


def _memory_compaction_enabled() -> bool:
    """判断是否启用 Memory Compaction。"""

    return os.getenv("BEGINNER_AGENT_MEMORY_COMPACTION_ENABLED", "true").lower() == "true"


def _compaction_min_group_size() -> int:
    """一组至少多少条记忆才触发压缩。"""

    return _env_int(
        "BEGINNER_AGENT_MEMORY_COMPACTION_MIN_GROUP_SIZE",
        DEFAULT_COMPACTION_MIN_GROUP_SIZE,
    )


def _compaction_max_groups() -> int:
    """每次最多压缩多少组，避免单轮 agent 花太多时间做后台维护。"""

    return _env_int(
        "BEGINNER_AGENT_MEMORY_COMPACTION_MAX_GROUPS_PER_RUN",
        DEFAULT_COMPACTION_MAX_GROUPS,
    )


def _compaction_source_limit() -> int:
    """每次最多扫描多少条 memory records。"""

    return _env_int(
        "BEGINNER_AGENT_MEMORY_COMPACTION_SOURCE_LIMIT",
        DEFAULT_COMPACTION_SOURCE_LIMIT,
    )


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _failure_profile(record: dict[str, Any]) -> dict[str, Any]:
    failure = _metadata(record).get("failure_memory")
    return failure if isinstance(failure, dict) else {}


def _active_source_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛出适合作为 compaction 输入的原始记忆。"""

    result: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("validity_status", "active")) != "active":
            continue
        if bool(record.get("pinned", False)):
            continue
        if str(record.get("source", "")) == "memory_compaction_node":
            continue
        if str(record.get("sensitivity_level", "internal")) == "secret":
            continue
        if not str(record.get("summary", "")).strip():
            continue
        result.append(record)
    return result


def _paths(record: dict[str, Any]) -> list[str]:
    paths = record.get("paths", [])
    if not isinstance(paths, list):
        return []
    return [str(path) for path in paths if str(path).strip()]


def _group_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """按失败模式、文件、项目阶段建立候选分组。"""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        failure = _failure_profile(record)
        failure_pattern_id = str(failure.get("pattern_id", "")).strip()
        failure_category = str(failure.get("category", "")).strip()
        if failure_pattern_id:
            groups.setdefault(("failure_pattern", failure_pattern_id), []).append(record)
        elif str(record.get("kind", "")) == "failure" or str(
            record.get("tool_result_status", "")
        ) in {"failed", "blocked"}:
            key = f"{failure_category or 'unknown'}:{record.get('tool_name', 'none')}"
            groups.setdefault(("failure_pattern", key), []).append(record)

        for path in _paths(record)[:5]:
            groups.setdefault(("file_memory", path), []).append(record)

        project_key = ":".join(
            [
                str(record.get("project_id", DEFAULT_PROJECT_ID)),
                str(record.get("kind", "task")),
                str(record.get("tool_name", "none")),
                str(record.get("tool_result_status", "none")),
            ]
        )
        groups.setdefault(("project_memory", project_key), []).append(record)
    return groups


def _candidate_score(records: list[dict[str, Any]]) -> float:
    """给候选组打分，越值得压缩越靠前。"""

    count_score = len(records)
    age_score = min(2.0, len({str(item.get("task_id", "")) for item in records}) / 3)
    quality_score = sum(float(item.get("quality_score", 0.5)) for item in records)
    return count_score + age_score + quality_score


def _build_candidates(records: list[dict[str, Any]]) -> list[CompactionCandidate]:
    min_group_size = _compaction_min_group_size()
    candidates: list[CompactionCandidate] = []
    for (group_type, key), group_records in _group_records(records).items():
        unique_records = {str(record.get("id", "")): record for record in group_records}
        compactable = list(unique_records.values())
        if len(compactable) < min_group_size:
            continue
        candidates.append(
            CompactionCandidate(
                group_type=group_type,  # type: ignore[arg-type]
                key=key,
                title=_candidate_title(group_type, key),
                records=compactable,
                score=_candidate_score(compactable),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _candidate_title(group_type: str, key: str) -> str:
    if group_type == "failure_pattern":
        return f"压缩失败模式：{key}"
    if group_type == "file_memory":
        return f"压缩文件记忆：{key}"
    return f"压缩项目阶段记忆：{key}"


def _select_non_overlapping_candidates(
    candidates: list[CompactionCandidate],
) -> list[CompactionCandidate]:
    """选择互不重叠的候选组，避免同一条旧记忆一轮内被压缩多次。"""

    selected: list[CompactionCandidate] = []
    used_ids: set[str] = set()
    for candidate in candidates:
        source_ids = {str(record.get("id", "")) for record in candidate.records}
        if source_ids.intersection(used_ids):
            continue
        selected.append(candidate)
        used_ids.update(source_ids)
        if len(selected) >= _compaction_max_groups():
            break
    return selected


def _compact_summary(candidate: CompactionCandidate) -> str:
    """把多条记忆压缩成一段稳定摘要。"""

    lines = [
        f"compaction_type={candidate.group_type}",
        f"key={candidate.key}",
        f"source_count={len(candidate.records)}",
    ]
    status_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    for record in candidate.records:
        status = str(record.get("tool_result_status", "none"))
        tool = str(record.get("tool_name", "none"))
        status_counts[status] = status_counts.get(status, 0) + 1
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    lines.append(f"statuses={status_counts}")
    lines.append(f"tools={tool_counts}")
    sample_reasons = [
        str(_metadata(record).get("source_memory", {}).get("reason", "")).strip()
        or str(record.get("summary", "")).strip()
        for record in candidate.records[:5]
    ]
    sample_reasons = [reason for reason in sample_reasons if reason]
    if sample_reasons:
        lines.append("evidence=" + " | ".join(sample_reasons)[:800])
    return "\n".join(lines)[:2000]


def _combined_sensitivity(records: list[dict[str, Any]]) -> SensitivityLevel:
    order = ["public", "internal", "confidential", "secret"]
    highest = "internal"
    for record in records:
        current = str(record.get("sensitivity_level", "internal"))
        if current in order and order.index(current) > order.index(highest):
            highest = current
    return highest  # type: ignore[return-value]


def _combined_paths(records: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for record in records:
        for path in _paths(record):
            if path not in paths:
                paths.append(path)
    return paths[:50]


def _combined_tags(candidate: CompactionCandidate) -> list[str]:
    tags = [
        "compacted",
        candidate.group_type,
        f"source_count:{len(candidate.records)}",
    ]
    for record in candidate.records:
        for tag in record.get("tags", []):
            text = str(tag)
            if text and text not in tags:
                tags.append(text)
    return tags[:50]


def _record_kind_for_candidate(candidate: CompactionCandidate) -> MemoryKind:
    if candidate.group_type == "failure_pattern":
        return "failure"
    if candidate.group_type == "file_memory":
        return "project"
    return "project"


def _record_scope_for_candidate(candidate: CompactionCandidate) -> MemoryScope:
    if candidate.group_type == "file_memory":
        return "file"
    return "project"


def _build_compacted_record(
    candidate: CompactionCandidate,
    state: State,
) -> MemoryRecord:
    """根据候选组生成新的压缩记忆。"""

    now = datetime.now(timezone.utc).isoformat()
    source_ids = [str(record.get("id", "")) for record in candidate.records]
    access_context = _memory_access_context(state)
    raw_record = {
        "kind": _record_kind_for_candidate(candidate),
        "task_id": state.get("current_task_id", "memory_compaction"),
        "title": candidate.title,
        "summary": _compact_summary(candidate),
        "status": "compacted",
        "tool_name": "none",
        "tool_result_status": "success",
        "paths": _combined_paths(candidate.records),
        "scope": _record_scope_for_candidate(candidate),
        "visibility": "project",
        "tenant_id": access_context["tenant_id"],
        "workspace_id": access_context["workspace_id"],
        "project_id": access_context["project_id"],
        "user_id": access_context["user_id"],
        "contradiction_key": f"compaction:{candidate.group_type}:{candidate.key}",
    }
    record_id = _stable_memory_id(raw_record)
    source_quality = [
        float(record.get("quality_score", 0.5)) for record in candidate.records
    ]
    source_trust = [float(record.get("trust_score", 0.5)) for record in candidate.records]
    return MemoryRecord(
        id=record_id,
        kind=raw_record["kind"],
        task_id=str(raw_record["task_id"]),
        title=str(raw_record["title"]),
        summary=str(raw_record["summary"]),
        status="compacted",
        tool_name="none",
        tool_result_status="success",
        paths=raw_record["paths"],
        tags=_combined_tags(candidate),
        confidence=round(min(0.95, 0.65 + len(candidate.records) * 0.03), 4),
        importance=round(min(0.95, 0.65 + len(candidate.records) * 0.04), 4),
        quality_score=round(sum(source_quality) / max(1, len(source_quality)), 4),
        trust_score=round(sum(source_trust) / max(1, len(source_trust)), 4),
        decay_score=0.05,
        scope=raw_record["scope"],
        visibility="project",
        sensitivity_level=_combined_sensitivity(candidate.records),
        tenant_id=access_context["tenant_id"],
        workspace_id=access_context["workspace_id"],
        project_id=access_context["project_id"],
        user_id=access_context["user_id"],
        retention_policy="long_term",
        validity_status="active",
        pinned=False,
        expires_at=None,
        supersedes=source_ids[0] if source_ids else None,
        contradiction_key=str(raw_record["contradiction_key"]),
        source="memory_compaction_node",
        created_at=now,
        metadata={
            "compaction": {
                "type": candidate.group_type,
                "key": candidate.key,
                "source_memory_ids": source_ids,
                "source_count": len(source_ids),
                "strategy": "deterministic_group_compaction",
                "created_at": now,
            },
            "source_records": _safe_memory_value(candidate.records[:20]),
        },
    )


def _store_with_fallback() -> tuple[MemoryStore, str]:
    """读取配置的 store；失败时回退 JSONL。"""

    try:
        store = _configured_store()
        return store, ""
    except Exception as exc:
        return JsonlMemoryStore(), str(exc)


def compact_memories(state: State) -> dict[str, Any]:
    """执行一次 Memory Compaction。

    中文注释：
    这是可控的后台维护步骤。
    它不会修改用户代码，只会整理 memory store：
    - 写入新的 compacted memory。
    - 把被压缩的旧 memory 标记为 superseded。
    - 写 audit event，方便 API 查询为什么压缩。
    """

    if not _memory_compaction_enabled():
        return {
            "enabled": False,
            "backend": "disabled",
            "compacted_count": 0,
            "superseded_count": 0,
            "reason": "BEGINNER_AGENT_MEMORY_COMPACTION_ENABLED=false。",
        }

    store, setup_error = _store_with_fallback()
    try:
        records = store.list_records(min(MAX_MEMORY_RECORDS, _compaction_source_limit()))
    except Exception as exc:
        store = JsonlMemoryStore()
        records = store.list_records(min(MAX_MEMORY_RECORDS, _compaction_source_limit()))
        setup_error = str(exc)

    source_records = _active_source_records(records)
    candidates = _build_candidates(source_records)
    selected = _select_non_overlapping_candidates(candidates)
    if not selected:
        return {
            "enabled": True,
            "backend": store.backend_name,
            "backend_warning": setup_error,
            "compacted_count": 0,
            "superseded_count": 0,
            "candidate_count": len(candidates),
            "reason": "没有达到压缩阈值的 memory group。",
        }

    compacted_records: list[dict[str, Any]] = []
    superseded_ids: list[str] = []
    for candidate in selected:
        compacted_record = _build_compacted_record(candidate, state)
        source_ids = [str(record.get("id", "")) for record in candidate.records]
        store.upsert_record(compacted_record)
        store.mark_records_status(
            source_ids,
            "superseded",
            superseded_by=compacted_record.id,
        )
        audit_event: MemoryAuditEvent = _build_audit_event(
            action="compact",
            memory_id=compacted_record.id,
            reason=(
                "Memory Compaction 把多条相似记忆合并成一条长期记忆，"
                "旧记录已标记为 superseded。"
            ),
            backend=store.backend_name,
            metadata={
                "compaction": compacted_record.metadata["compaction"],
                "source_memory_ids": source_ids,
            },
        )
        store.upsert_audit_event(audit_event)
        compacted_records.append(compacted_record.model_dump(mode="json"))
        superseded_ids.extend(source_ids)

    return {
        "enabled": True,
        "backend": store.backend_name,
        "backend_warning": setup_error,
        "compacted_count": len(compacted_records),
        "superseded_count": len(superseded_ids),
        "candidate_count": len(candidates),
        "compacted_memory_ids": [record["id"] for record in compacted_records],
        "superseded_memory_ids": superseded_ids,
        "groups": [
            {
                "type": candidate.group_type,
                "key": candidate.key,
                "source_count": len(candidate.records),
            }
            for candidate in selected
        ],
    }


def memory_compaction_node(state: State) -> dict[str, object]:
    """Memory Compaction Node：压缩长期记忆，降低检索噪声和速度成本。"""

    report = compact_memories(state)
    compacted_count = int(report.get("compacted_count", 0))
    superseded_count = int(report.get("superseded_count", 0))
    return {
        "memory_compaction_report": report,
        "next_action": "finish" if state["done"] else "schedule",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Memory Compaction：生成 "
                    f"{compacted_count} 条压缩记忆，"
                    f"归档 {superseded_count} 条旧记忆，"
                    f"backend={report.get('backend', '')}。"
                ),
            }
        ],
    }


def route_after_memory_compaction(state: State) -> CompactionRoute:
    """Memory Compaction 后的路由。"""

    if state["done"] or state["next_action"] == "finish":
        return "finish"
    return "schedule"
