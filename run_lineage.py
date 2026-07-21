from __future__ import annotations

import json
from typing import Any

from .memory_jsonl_store import JsonlMemoryStore, _read_jsonl_audit_events
from .memory_postgres_store import PostgresMemoryStore
from .memory_settings import MAX_MEMORY_AUDIT_EVENTS
from .memory_store import _configured_store
from .observability_store import ObservabilityStore
from .state import State


def _metadata(value: dict[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _run_id_from_event(event: dict[str, Any]) -> str:
    metadata = _metadata(event)
    return str(metadata.get("run_id", ""))


def _postgres_audit_events(limit: int) -> list[dict[str, Any]]:
    store = _configured_store()
    if not isinstance(store, PostgresMemoryStore):
        return []
    store.list_records(1)
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


def audit_events_for_run(run_id: str, *, limit: int = 200) -> tuple[list[dict[str, Any]], str, str]:
    """查询某次 run 相关的 memory audit events。

    中文注释：
    这是 run lineage 的关键连接点。
    memory 写入、memory 检索、敏感访问、生命周期治理都会写 audit。
    只要 audit metadata 里带 run_id，这里就能把它们串回同一次运行。
    """

    try:
        store = _configured_store()
        if isinstance(store, PostgresMemoryStore):
            events = _postgres_audit_events(min(limit, MAX_MEMORY_AUDIT_EVENTS))
            backend = "postgres"
        else:
            events = _read_jsonl_audit_events(min(limit, MAX_MEMORY_AUDIT_EVENTS))
            backend = store.backend_name
        error = ""
    except Exception as exc:
        events = _read_jsonl_audit_events(min(limit, MAX_MEMORY_AUDIT_EVENTS))
        backend = JsonlMemoryStore.backend_name
        error = str(exc)
    matched = [event for event in events if _run_id_from_event(event) == run_id]
    return matched[:limit], backend, error


def _memory_context_summary(state: State) -> dict[str, Any]:
    memory_context = state.get("memory_context", {})
    if not isinstance(memory_context, dict):
        memory_context = {}
    records = memory_context.get("relevant_records", [])
    if not isinstance(records, list):
        records = []
    return {
        "backend": memory_context.get("backend", ""),
        "backend_error": memory_context.get("backend_error", ""),
        "retrieved_memory_ids": [str(record.get("id", "")) for record in records],
        "retrieved_count": len(records),
        "preference_count": memory_context.get("user_preferences", {}).get("count", 0),
        "state_note_count": memory_context.get("state_note_count", 0),
    }


def _generated_memory_summary(state: State) -> dict[str, Any]:
    memory_notes = list(state.get("memory_notes", []))
    generated = [
        note for note in memory_notes if isinstance(note, dict) and note.get("backend")
    ]
    return {
        "generated_memory_ids": [str(note.get("id", "")) for note in generated],
        "generated_count": len(generated),
        "latest": generated[-3:],
    }


def _tool_result_summary(state: State) -> dict[str, Any]:
    tool_result_data = state.get("tool_result_data", {})
    if not isinstance(tool_result_data, dict):
        tool_result_data = {}
    return {
        "tool_name": state.get("tool_name", "none"),
        "tool_args": state.get("tool_args", {}),
        "tool_result_status": state.get("tool_result_status", "none"),
        "execution_status": state.get("execution_status", "not_started"),
        "changed_files": tool_result_data.get("changed_files", []),
        "artifact_paths": tool_result_data.get("artifact_paths", []),
        "duration_ms": tool_result_data.get("duration_ms", 0),
        "retryable": tool_result_data.get("retryable", False),
    }


def build_run_lineage_report(state: State) -> dict[str, Any]:
    """生成当前 State 的 run lineage 报告。

    中文注释：
    这份报告回答一个生产系统常见问题：

        某次 run 到底发生了什么？

    它把原来分散的几类数据串起来：
    - checkpoint：这次 run 的恢复后端。
    - memory：用了哪些记忆，生成了哪些记忆。
    - tool result：执行了哪个工具，结果怎样。
    - audit event：有哪些治理事件。
    - observability：最后是否成功，目标进度如何。
    """

    run_id = state["run_id"]
    audit_events, audit_backend, audit_error = audit_events_for_run(run_id)
    return {
        "run_id": run_id,
        "checkpoint": state.get("checkpoint_report", {}),
        "memory": {
            "used": _memory_context_summary(state),
            "generated": _generated_memory_summary(state),
            "compaction": state.get("memory_compaction_report", {}),
        },
        "tool_result": _tool_result_summary(state),
        "audit": {
            "backend": audit_backend,
            "backend_error": audit_error,
            "event_count": len(audit_events),
            "events": audit_events[-50:],
        },
        "observability": {
            "done": state.get("done", False),
            "step_count": state.get("step_count", 0),
            "goal_progress": state.get("goal_progress", {}),
            "evaluation": {
                "decision": state.get("evaluation_decision", "none"),
                "reason": state.get("evaluation_reason", ""),
            },
            "recovery": {
                "action": state.get("recovery_action", "none"),
                "reason": state.get("recovery_reason", ""),
            },
        },
        "success": bool(state.get("done")) and state.get("tool_result_status") != "failed",
    }


def lineage_for_run_id(run_id: str) -> dict[str, Any]:
    """从已落地的 observability/audit 数据重建某次 run 的 lineage。"""

    reports = ObservabilityStore().recent_reports(run_id=run_id, limit=1)
    latest_report = reports[0] if reports else {}
    audit_events, audit_backend, audit_error = audit_events_for_run(run_id)
    lineage = latest_report.get("lineage", {}) if isinstance(latest_report, dict) else {}
    return {
        "run_id": run_id,
        "observability_report_found": bool(latest_report),
        "latest_observability_report": latest_report,
        "lineage": lineage,
        "audit": {
            "backend": audit_backend,
            "backend_error": audit_error,
            "event_count": len(audit_events),
            "events": audit_events[-100:],
        },
    }
