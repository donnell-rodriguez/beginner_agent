from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from beginner_agent.checkpoint_node import postgres_checkpoint_node, route_after_postgres_checkpoint
import beginner_agent.checkpoint_observability as checkpoint_observability
from beginner_agent.checkpointing import build_checkpointer, checkpoint_backend_config
from beginner_agent.state_factory import create_initial_state


def _clear_checkpoint_env(monkeypatch) -> None:
    """清理 checkpoint 相关 env，避免本地 .env 影响测试。"""

    for name in (
        "BEGINNER_AGENT_CHECKPOINT_BACKEND",
        "BEGINNER_AGENT_CHECKPOINT_DATABASE_URL",
        "BEGINNER_AGENT_CHECKPOINT_ALLOW_MEMORY_FALLBACK",
        "BEGINNER_AGENT_CHECKPOINT_REQUIRE_THREAD_ID",
        "BEGINNER_AGENT_CHECKPOINT_HEALTHCHECK_ENABLED",
        "BEGINNER_AGENT_CHECKPOINT_NAMESPACE",
        "BEGINNER_AGENT_CHECKPOINT_REQUIRE_PERSISTENCE_FOR_AGENT",
        "BEGINNER_AGENT_CHECKPOINT_REQUIRE_PERSISTENCE_FOR_APPROVAL",
        "BEGINNER_AGENT_CHECKPOINT_REQUIRE_PERSISTENCE_FOR_WRITE_TOOLS",
        "BEGINNER_AGENT_CHECKPOINT_LONG_TASK_STEP_THRESHOLD",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_checkpoint_node_reports_memory_backend(monkeypatch) -> None:
    """memory backend 要报告为非持久化，并提示只适合本地实验。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "memory")

    result = postgres_checkpoint_node(create_initial_state("hello"))
    report = result["checkpoint_report"]

    assert report["backend"] == "memory"
    assert report["persistent"] is False
    assert report["health"]["status"] == "warning"
    assert report["recovery_contract"]["resume_supported"] is False


def test_checkpoint_node_blocks_postgres_without_database_url(monkeypatch) -> None:
    """Postgres backend 缺少 database_url 且不允许 fallback 时要 blocked。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_ALLOW_MEMORY_FALLBACK", "false")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_HEALTHCHECK_ENABLED", "false")

    result = postgres_checkpoint_node(create_initial_state("hello"))
    report = result["checkpoint_report"]

    assert report["backend"] == "postgres"
    assert report["health"]["status"] == "blocked"
    assert report["health"]["database_url_configured"] is False
    assert report["recovery_contract"]["fallback_policy"] == "fail_fast"


def test_checkpoint_can_fallback_to_memory_when_explicitly_allowed(monkeypatch) -> None:
    """本地开发允许 fallback 时，Postgres 缺配置会降级到 memory。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_ALLOW_MEMORY_FALLBACK", "true")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_LONG_TASK_STEP_THRESHOLD", "30")

    config = checkpoint_backend_config()
    checkpointer = build_checkpointer()
    state = create_initial_state("hello")
    state["max_steps"] = 4
    result = postgres_checkpoint_node(state)
    report = result["checkpoint_report"]

    assert config.requested_backend == "postgres"
    assert config.effective_backend == "memory"
    assert isinstance(checkpointer, MemorySaver)
    assert report["health"]["status"] == "degraded"
    assert report["requested_backend"] == "postgres"
    assert report["backend"] == "memory"
    assert report["fallback_risk_decision"]["allowed"] is True


def test_checkpoint_fallback_blocks_high_risk_agent_task(monkeypatch) -> None:
    """高风险复杂 agent 任务不能在 memory fallback 下继续执行。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_ALLOW_MEMORY_FALLBACK", "true")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_HEALTHCHECK_ENABLED", "false")

    state = create_initial_state("请修改代码并运行测试")
    state["task_type"] = "agent"
    state["risk_level"] = "high"
    state["tool_name"] = "apply_patch"

    result = postgres_checkpoint_node(state)
    report = result["checkpoint_report"]

    assert report["backend"] == "memory"
    assert report["health"]["status"] == "blocked"
    assert report["fallback_risk_decision"]["allowed"] is False
    assert report["fallback_risk_decision"]["requires_persistence"] is True
    assert "risk_level_high" in report["fallback_risk_decision"]["risk_factors"]
    assert result["next_action"] == "finish"
    assert result["done"] is True


def test_checkpoint_fallback_blocks_long_task(monkeypatch) -> None:
    """长任务即使风险不高，也不应该降级到不可恢复的 memory checkpoint。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_ALLOW_MEMORY_FALLBACK", "true")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_LONG_TASK_STEP_THRESHOLD", "5")

    state = create_initial_state("梳理整个项目")
    state["task_type"] = "chat"
    state["risk_level"] = "low"
    state["max_steps"] = 8

    result = postgres_checkpoint_node(state)
    report = result["checkpoint_report"]

    assert report["health"]["status"] == "blocked"
    assert "long_task_max_steps_8" in report["fallback_risk_decision"]["risk_factors"]


def test_route_after_postgres_checkpoint_finishes_when_blocked() -> None:
    """checkpoint blocked 时，graph 要进入最终 summary，而不是继续 scheduler。"""

    state = create_initial_state("hello")
    state["checkpoint_report"] = {"health": {"status": "blocked"}}

    assert route_after_postgres_checkpoint(state) == "finish"


def test_route_after_postgres_checkpoint_schedules_when_healthy() -> None:
    """checkpoint 没有阻断时，graph 继续进入 Scheduler。"""

    state = create_initial_state("hello")
    state["checkpoint_report"] = {"health": {"status": "healthy"}}

    assert route_after_postgres_checkpoint(state) == "schedule"


def test_checkpoint_node_reports_postgres_when_configured_without_live_healthcheck(
    monkeypatch,
) -> None:
    """关闭真实 healthcheck 时，只做配置层检查，不连接本地数据库。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_HEALTHCHECK_ENABLED", "false")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_NAMESPACE", "unit-test")

    result = postgres_checkpoint_node(create_initial_state("hello"))
    report = result["checkpoint_report"]

    assert report["backend"] == "postgres"
    assert report["persistent"] is True
    assert report["health"]["status"] == "healthy"
    assert report["health"]["setup_status"] == "assumed_runtime_setup"
    assert report["health"]["diagnostics"]["roundtrip_status"] == "not_run"
    assert report["recovery_contract"]["checkpoint_namespace"] == "unit-test"
    assert report["recovery_contract"]["resume_supported"] is True
    assert report["observability_event"]["event_type"] == "checkpoint_health"
    assert report["observability_event"]["diagnostics"]["roundtrip_status"] == "not_run"


def test_checkpoint_observability_writes_jsonl_event(monkeypatch, tmp_path) -> None:
    """checkpoint_node 要把事件写入独立 checkpoint_events.jsonl。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_OBSERVABILITY_SINK", "jsonl")
    monkeypatch.setattr(checkpoint_observability, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(
        checkpoint_observability,
        "CHECKPOINT_EVENTS_FILE",
        tmp_path / "checkpoint_events.jsonl",
    )

    postgres_checkpoint_node(create_initial_state("hello"))
    events = checkpoint_observability.read_checkpoint_events()

    assert events[-1]["event_type"] == "checkpoint_health"
    assert events[-1]["backend"] == "memory"
    assert events[-1]["status"] == "warning"
    assert events[-1]["alerts"]


def test_checkpoint_observability_kafka_spool_sink(monkeypatch, tmp_path) -> None:
    """kafka_spool sink 要写入 spool 文件，方便后续 producer 转发。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_ALLOW_MEMORY_FALLBACK", "true")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_OBSERVABILITY_SINK", "kafka_spool")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_KAFKA_TOPIC", "test.checkpoint")
    monkeypatch.setattr(checkpoint_observability, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(
        checkpoint_observability,
        "CHECKPOINT_KAFKA_SPOOL_FILE",
        tmp_path / "checkpoint_kafka_spool.jsonl",
    )

    postgres_checkpoint_node(create_initial_state("hello"))
    spool = checkpoint_observability.CHECKPOINT_KAFKA_SPOOL_FILE.read_text(encoding="utf-8")

    assert '"_sink": "kafka_spool"' in spool
    assert '"_topic": "test.checkpoint"' in spool
    assert "checkpoint_fallback_to_memory" in spool


def test_checkpoint_observability_null_sink_does_not_write(monkeypatch, tmp_path) -> None:
    """null sink 不写文件，适合测试或隐私场景。"""

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_OBSERVABILITY_SINK", "null")
    monkeypatch.setattr(checkpoint_observability, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(
        checkpoint_observability,
        "CHECKPOINT_EVENTS_FILE",
        tmp_path / "checkpoint_events.jsonl",
    )

    postgres_checkpoint_node(create_initial_state("hello"))

    assert not checkpoint_observability.CHECKPOINT_EVENTS_FILE.exists()


def test_checkpoint_observability_failure_does_not_break_node(monkeypatch) -> None:
    """checkpoint observability 写入失败不能拖垮 checkpoint node 主路径。"""

    class FailingSink:
        def append_event(self, event):
            raise OSError("disk full")

        def read_events(self, limit=None):
            return []

    _clear_checkpoint_env(monkeypatch)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setattr(
        checkpoint_observability,
        "resolve_checkpoint_observability_sink",
        lambda: FailingSink(),
    )

    result = postgres_checkpoint_node(create_initial_state("hello"))

    assert result["checkpoint_report"]["backend"] == "memory"
