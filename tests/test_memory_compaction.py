from __future__ import annotations

from typing import Any

from beginner_agent import memory_compaction
from beginner_agent.state_factory import create_initial_state


def _record(memory_id: str, *, path: str = "memory.py") -> dict[str, Any]:
    return {
        "id": memory_id,
        "kind": "failure",
        "task_id": "task-1",
        "title": f"测试失败 {memory_id}",
        "summary": "pytest failure in memory.py",
        "status": "failed",
        "tool_name": "run_tests",
        "tool_result_status": "failed",
        "paths": [path],
        "tags": ["failure"],
        "confidence": 0.8,
        "importance": 0.7,
        "quality_score": 0.8,
        "trust_score": 0.8,
        "decay_score": 0.0,
        "scope": "project",
        "visibility": "project",
        "sensitivity_level": "internal",
        "tenant_id": "local-tenant",
        "workspace_id": "local-workspace",
        "project_id": "beginner_agent",
        "user_id": "local-user",
        "retention_policy": "ttl",
        "validity_status": "active",
        "pinned": False,
        "expires_at": None,
        "supersedes": None,
        "contradiction_key": None,
        "source": "memory_writer_node",
        "created_at": "2026-07-21T00:00:00+00:00",
        "metadata": {
            "failure_memory": {
                "category": "test_failure",
                "pattern_id": "pytest-memory-failure",
            }
        },
    }


def test_compaction_creates_summary_and_supersedes_sources(
    monkeypatch,
    fake_memory_store,
) -> None:
    fake_memory_store.records = [_record("m1"), _record("m2"), _record("m3")]
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_COMPACTION_MIN_GROUP_SIZE", "2")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_COMPACTION_MAX_GROUPS_PER_RUN", "1")
    monkeypatch.setattr(memory_compaction, "_store_with_fallback", lambda: (fake_memory_store, ""))

    state = create_initial_state("整理失败记忆")
    state["current_task_id"] = "task-compact"
    report = memory_compaction.compact_memories(state)

    assert report["compacted_count"] == 1
    assert report["superseded_count"] == 3
    assert fake_memory_store.upserts
    assert fake_memory_store.status_updates[0]["status"] == "superseded"
    assert fake_memory_store.audit_events
