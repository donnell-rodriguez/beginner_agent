from __future__ import annotations

from beginner_agent import memory_lifecycle_scheduler as scheduler
from beginner_agent.memory_lifecycle import LifecycleReport


def _report() -> LifecycleReport:
    return LifecycleReport(
        backend="fake",
        backend_warning="",
        expired_cleaned=1,
        low_value_deprioritized=0,
        high_value_promoted=0,
        contradiction_fixed=0,
        summary_created=0,
        compaction_report={},
        embeddings_rebuilt=0,
        audit_events_written=1,
    )


def test_lifecycle_scheduler_records_success_and_skips_duplicate(
    tmp_path,
    monkeypatch,
) -> None:
    memory_dir = tmp_path / "memory"
    monkeypatch.setattr(scheduler, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(
        scheduler,
        "MEMORY_LIFECYCLE_RUNS_FILE",
        memory_dir / "memory_lifecycle_runs.jsonl",
    )
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_HISTORY_BACKEND", "jsonl")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_RUN_KEY", "unit-run-key")
    monkeypatch.setattr(scheduler, "run_memory_lifecycle_job", _report)

    first = scheduler.run_memory_lifecycle_scheduled()
    second = scheduler.run_memory_lifecycle_scheduled()

    assert first.status == "success"
    assert first.locked is True
    assert first.report["expired_cleaned"] == 1
    assert second.status == "skipped"
    assert "已经成功执行过" in second.skipped_reason


def test_lifecycle_scheduler_retries_then_succeeds(tmp_path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    attempts = {"count": 0}

    def flaky_report() -> LifecycleReport:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        return _report()

    monkeypatch.setattr(scheduler, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(
        scheduler,
        "MEMORY_LIFECYCLE_RUNS_FILE",
        memory_dir / "memory_lifecycle_runs.jsonl",
    )
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_HISTORY_BACKEND", "jsonl")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_RUN_KEY", "retry-run-key")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(scheduler, "run_memory_lifecycle_job", flaky_report)

    record = scheduler.run_memory_lifecycle_scheduled()

    assert record.status == "success"
    assert record.attempt == 2
    assert attempts["count"] == 2
