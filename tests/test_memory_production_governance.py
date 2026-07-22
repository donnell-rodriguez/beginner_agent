from __future__ import annotations

import beginner_agent.memory.effectiveness as effectiveness
import beginner_agent.memory.feedback as feedback
import beginner_agent.memory.observability_sinks as observability_sinks
import beginner_agent.memory.online_eval as online_eval
from beginner_agent.memory.effectiveness import (
    MemoryUsageEvent,
    append_memory_usage,
    close_memory_usage_loop,
    read_memory_usage,
    summarize_memory_usage,
)
from beginner_agent.memory.observability_sinks import (
    append_memory_observability_event,
    read_memory_observability_events,
)
from beginner_agent.memory.online_eval import (
    read_online_eval_events,
    record_retrieval_online_eval,
    summarize_online_eval,
)
from beginner_agent.memory.postgres_performance import _recommendations
from fastapi.testclient import TestClient

from beginner_agent.memory_api.app import create_app
from beginner_agent.state_factory import create_initial_state


def test_memory_usage_effectiveness_closes_pending_and_writes_feedback(
    monkeypatch,
    tmp_path,
) -> None:
    """Memory 使用效果要能反向形成 feedback。"""

    usage_file = tmp_path / "memory_usage.jsonl"
    feedback_file = tmp_path / "memory_feedback.jsonl"
    monkeypatch.setattr(effectiveness, "MEMORY_USAGE_FILE", usage_file)
    monkeypatch.setattr(effectiveness, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(feedback, "MEMORY_FEEDBACK_FILE", feedback_file)
    monkeypatch.setattr(feedback, "MEMORY_DIR", tmp_path)

    state = create_initial_state("请修复 memory 检索问题")
    state["run_id"] = "run-usage-1"
    state["current_task_id"] = "task-1"
    state["done"] = True
    state["tool_result_status"] = "success"

    append_memory_usage(
        MemoryUsageEvent(
            memory_id="memory-1",
            run_id="run-usage-1",
            task_id="task-1",
            outcome="pending",
            reason="retrieved",
            retrieval_score=0.8,
            rerank_score=0.9,
        )
    )

    result = close_memory_usage_loop(state)

    assert result["updated"] == 1
    assert summarize_memory_usage("memory-1")["counts"]["helped"] == 1
    assert read_memory_usage()[-1]["outcome"] == "helped"
    assert feedback.read_memory_feedback()[-1]["signal"] == "useful"


def test_memory_online_eval_records_real_retrieval_quality(
    monkeypatch,
    tmp_path,
) -> None:
    """真实检索后要能沉淀 online eval 事件。"""

    online_file = tmp_path / "memory_online_eval.jsonl"
    monkeypatch.setattr(online_eval, "MEMORY_ONLINE_EVAL_FILE", online_file)
    monkeypatch.setattr(online_eval, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(
        online_eval,
        "read_memory_eval_cases",
        lambda limit: [
            {
                "query": "memory 检索",
                "expected_memory_ids": ["memory-1"],
                "negative_memory_ids": ["memory-bad"],
            }
        ],
    )

    state = create_initial_state("帮我检查 memory 检索质量")
    state["run_id"] = "run-eval-1"
    state["current_task_id"] = "task-1"
    returned = [{"id": "memory-1"}, {"id": "memory-2"}]

    record_retrieval_online_eval(
        state,
        returned,
        backend="jsonl",
        backend_error="",
    )

    summary = summarize_online_eval()
    event = read_online_eval_events()[-1]

    assert online_file.exists()
    assert summary["matched_case_count"] == 1
    assert summary["passed_case_count"] == 1
    assert event["returned_ids"] == ["memory-1", "memory-2"]


def test_memory_observability_supports_kafka_and_otel_spool(
    monkeypatch,
    tmp_path,
) -> None:
    """Observability sink 失败不能耦合主流程，同时要支持可插拔 spool。"""

    monkeypatch.setattr(observability_sinks, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(
        observability_sinks,
        "MEMORY_OBSERVABILITY_FILE",
        tmp_path / "memory_observability.jsonl",
    )
    monkeypatch.setattr(
        observability_sinks,
        "MEMORY_KAFKA_SPOOL_FILE",
        tmp_path / "memory_kafka_spool.jsonl",
    )
    monkeypatch.setattr(
        observability_sinks,
        "MEMORY_OTEL_SPOOL_FILE",
        tmp_path / "memory_otel_spool.jsonl",
    )
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_OBSERVABILITY_ENABLED", "true")

    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_OBSERVABILITY_SINK", "kafka_spool")
    append_memory_observability_event(
        {"event_type": "memory_retrieved", "run_id": "run-1", "created_at": "1"}
    )

    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_OBSERVABILITY_SINK", "otel_spool")
    append_memory_observability_event(
        {"event_type": "memory_rerank", "run_id": "run-1", "created_at": "2"}
    )

    events = read_memory_observability_events()

    assert any(event.get("_sink") == "kafka_spool" for event in events)
    assert any(event.get("_sink") == "otel_spool" for event in events)


def test_postgres_governance_recommendations_are_actionable() -> None:
    """Postgres 治理报告要能给出迁移和索引建议。"""

    recommendations = _recommendations(
        pending=[{"version": 2}],
        missing_indexes=["idx_beginner_agent_memory_acl"],
        counts={
            "beginner_agent_memory": 100001,
            "beginner_agent_memory_audit": 500001,
        },
    )

    assert any("upgrade" in item for item in recommendations)
    assert any("索引" in item for item in recommendations)
    assert any("分区" in item for item in recommendations)
    assert any("数据仓库" in item for item in recommendations)


def test_memory_api_exposes_new_governance_endpoints(monkeypatch, tmp_path) -> None:
    """Memory API 要暴露 usage/eval/observability/governance 查询入口。"""

    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_AUDITOR_TOKEN", "auditor-token")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_ADMIN_TOKEN", "admin-token")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_BACKEND", "jsonl")
    monkeypatch.setattr(effectiveness, "MEMORY_USAGE_FILE", tmp_path / "usage.jsonl")
    monkeypatch.setattr(effectiveness, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(online_eval, "MEMORY_ONLINE_EVAL_FILE", tmp_path / "online.jsonl")
    monkeypatch.setattr(online_eval, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(
        observability_sinks,
        "MEMORY_OBSERVABILITY_FILE",
        tmp_path / "observability.jsonl",
    )
    monkeypatch.setattr(observability_sinks, "MEMORY_DIR", tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    client = TestClient(create_app())

    usage = client.get(
        "/usage/effectiveness",
        headers={"Authorization": "Bearer auditor-token"},
    )
    online = client.get(
        "/eval/online",
        headers={"Authorization": "Bearer auditor-token"},
    )
    observability = client.get(
        "/observability/events",
        headers={"Authorization": "Bearer auditor-token"},
    )
    governance = client.get(
        "/postgres/governance",
        headers={"Authorization": "Bearer admin-token"},
    )

    assert usage.status_code == 200
    assert online.status_code == 200
    assert observability.status_code == 200
    assert governance.status_code == 200
    assert governance.json()["ok"] is False
    assert "DATABASE_URL" in governance.json()["error"]
