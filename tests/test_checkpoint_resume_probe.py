from __future__ import annotations

import pytest

from beginner_agent.checkpoint_resume_probe import run_checkpoint_resume_probe


@pytest.mark.postgres
def test_postgres_checkpoint_resume_probe(postgres_database_url, monkeypatch) -> None:
    """真实 Postgres checkpoint 要能 interrupt 后用同一个 thread_id 恢复。

    中文注释：
    这是 checkpoint 的最终闭环测试：

    1. 第一次 graph.stream(...) 触发 interrupt。
    2. Postgres checkpointer 保存中间状态。
    3. 重新 build graph，模拟进程重启。
    4. 用同一个 thread_id + Command(resume=...) 恢复。
    5. 确认恢复后继续执行 finish 节点。
    """

    monkeypatch.setenv("DATABASE_URL", postgres_database_url)
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("BEGINNER_AGENT_CHECKPOINT_HEALTHCHECK_ROUNDTRIP_ENABLED", "true")

    result = run_checkpoint_resume_probe("pytest-postgres-resume-probe")

    assert result.backend == "postgres"
    assert result.interrupted is True
    assert result.interrupt_payload["probe"] == "checkpoint_resume"
    assert result.state_before_resume["marker"] == "started-before-interrupt"
    assert result.state_after_resume["marker"] == "started-before-interrupt"
    assert result.state_after_resume["approved"] is True
    assert result.state_after_resume["final"] == "resumed"
    assert result.resumed is True

