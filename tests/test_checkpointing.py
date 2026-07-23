from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from beginner_agent.checkpoint_node import postgres_checkpoint_node
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

    config = checkpoint_backend_config()
    checkpointer = build_checkpointer()
    result = postgres_checkpoint_node(create_initial_state("hello"))
    report = result["checkpoint_report"]

    assert config.requested_backend == "postgres"
    assert config.effective_backend == "memory"
    assert isinstance(checkpointer, MemorySaver)
    assert report["health"]["status"] == "degraded"
    assert report["requested_backend"] == "postgres"
    assert report["backend"] == "memory"


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
    assert report["recovery_contract"]["checkpoint_namespace"] == "unit-test"
    assert report["recovery_contract"]["resume_supported"] is True
    assert report["observability_event"]["event_type"] == "checkpoint_health"
