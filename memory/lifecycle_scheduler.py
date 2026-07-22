from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import load_project_env
from .lifecycle import LifecycleReport, run_memory_lifecycle_job
from .migrations import run_memory_migrations
from .settings import MEMORY_DIR, MEMORY_LIFECYCLE_RUNS_FILE


JOB_NAME = "memory_lifecycle"
LOCK_NAME = "beginner_agent:memory_lifecycle"


@dataclass(frozen=True)
class LifecycleScheduleConfig:
    """Lifecycle 调度配置。

    中文注释：
    大厂生产环境里，定时任务通常不是“想起来手动跑一下”。
    它应该有明确调度配置：
    - cron：什么时候应该跑。
    - max_attempts：失败最多重试几次。
    - retry_backoff_seconds：失败后等多久再重试。
    - lock_ttl_seconds：锁最长持有时间，防止任务挂死后永远卡住。
    """

    cron: str
    max_attempts: int
    retry_backoff_seconds: float
    lock_ttl_seconds: int
    force: bool = False


@dataclass(frozen=True)
class LifecycleRunRecord:
    """一次 lifecycle 调度运行记录。"""

    run_id: str
    run_key: str
    job_name: str
    status: str
    attempt: int
    max_attempts: int
    started_at: str
    finished_at: str | None = None
    locked: bool = False
    skipped_reason: str = ""
    backend: str = ""
    error: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_key": self.run_key,
            "job_name": self.job_name,
            "status": self.status,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "locked": self.locked,
            "skipped_reason": self.skipped_reason,
            "backend": self.backend,
            "error": self.error,
            "report": self.report,
            "metadata": self.metadata,
        }


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default)).strip()))
    except ValueError:
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _schedule_config(force: bool = False) -> LifecycleScheduleConfig:
    cron = os.getenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_CRON", "*/30 * * * *").strip()
    if len(cron.split()) != 5:
        raise ValueError("BEGINNER_AGENT_MEMORY_LIFECYCLE_CRON 必须是 5 段 cron 表达式。")
    return LifecycleScheduleConfig(
        cron=cron,
        max_attempts=_env_int("BEGINNER_AGENT_MEMORY_LIFECYCLE_MAX_ATTEMPTS", 3, minimum=1),
        retry_backoff_seconds=_env_float(
            "BEGINNER_AGENT_MEMORY_LIFECYCLE_RETRY_BACKOFF_SECONDS",
            2.0,
            minimum=0.0,
        ),
        lock_ttl_seconds=_env_int(
            "BEGINNER_AGENT_MEMORY_LIFECYCLE_LOCK_TTL_SECONDS",
            900,
            minimum=30,
        ),
        force=force,
    )


def _run_key(config: LifecycleScheduleConfig) -> str:
    """生成幂等 run key。

    中文注释：
    同一个调度窗口内，同一个 job 不应该被重复执行成功多次。
    本地先按分钟生成 key；如果外部系统已经提供 run key，则优先使用外部 key。
    """

    explicit = os.getenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_RUN_KEY", "").strip()
    if explicit:
        return explicit
    minute_bucket = _utc_now().strftime("%Y%m%d%H%M")
    return f"{JOB_NAME}:{config.cron}:{minute_bucket}"


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _history_backend() -> str:
    backend = os.getenv("BEGINNER_AGENT_MEMORY_LIFECYCLE_HISTORY_BACKEND", "auto").strip()
    if backend == "auto":
        return "postgres" if _database_url() else "jsonl"
    return backend


def _jsonl_history_records() -> list[dict[str, Any]]:
    if not MEMORY_LIFECYCLE_RUNS_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in MEMORY_LIFECYCLE_RUNS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _write_jsonl_record(record: LifecycleRunRecord) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with MEMORY_LIFECYCLE_RUNS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _postgres_connect():
    import psycopg

    database_url = _database_url()
    if not database_url:
        raise RuntimeError("缺少 DATABASE_URL，无法使用 Postgres lifecycle history。")
    run_memory_migrations(database_url)
    return psycopg.connect(database_url)


def _successful_run_exists(run_key: str) -> bool:
    backend = _history_backend()
    if backend == "postgres":
        with _postgres_connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM beginner_agent_memory_lifecycle_runs
                WHERE job_name = %s AND run_key = %s AND status = 'success'
                LIMIT 1
                """,
                (JOB_NAME, run_key),
            ).fetchone()
            return row is not None
    return any(
        record.get("job_name") == JOB_NAME
        and record.get("run_key") == run_key
        and record.get("status") == "success"
        for record in _jsonl_history_records()
    )


def _save_run_record(record: LifecycleRunRecord) -> None:
    backend = _history_backend()
    if backend == "postgres":
        with _postgres_connect() as conn:
            conn.execute(
                """
                INSERT INTO beginner_agent_memory_lifecycle_runs (
                    run_id, run_key, job_name, status, attempt, max_attempts,
                    started_at, finished_at, locked, skipped_reason,
                    backend, error, report, metadata
                )
                VALUES (
                    %(run_id)s, %(run_key)s, %(job_name)s, %(status)s,
                    %(attempt)s, %(max_attempts)s, %(started_at)s,
                    %(finished_at)s, %(locked)s, %(skipped_reason)s,
                    %(backend)s, %(error)s, %(report)s::jsonb, %(metadata)s::jsonb
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    finished_at = EXCLUDED.finished_at,
                    locked = EXCLUDED.locked,
                    skipped_reason = EXCLUDED.skipped_reason,
                    backend = EXCLUDED.backend,
                    error = EXCLUDED.error,
                    report = EXCLUDED.report,
                    metadata = EXCLUDED.metadata
                """,
                {
                    **record.as_dict(),
                    "report": json.dumps(record.report, ensure_ascii=False),
                    "metadata": json.dumps(record.metadata, ensure_ascii=False),
                },
            )
        return
    _write_jsonl_record(record)


@contextmanager
def _postgres_advisory_lock() -> Iterator[bool]:
    conn = _postgres_connect()
    locked = False
    try:
        locked = bool(
            conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (LOCK_NAME,)).fetchone()[0]
        )
        yield locked
    finally:
        if locked:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (LOCK_NAME,))
        conn.close()


@contextmanager
def _file_lock(lock_ttl_seconds: int) -> Iterator[bool]:
    """JSONL fallback 使用的本地文件锁。

    中文注释：
    这只能保护同一台机器上的并发。
    如果是多机器/多 Pod，就必须使用 Postgres advisory lock、Redis lock 或调度系统锁。
    """

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MEMORY_DIR / "memory_lifecycle.lock"
    now = time.time()
    if lock_path.exists():
        age = now - lock_path.stat().st_mtime
        if age <= lock_ttl_seconds:
            yield False
            return
        lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": _iso_now()}))
        yield True
    finally:
        lock_path.unlink(missing_ok=True)


@contextmanager
def _distributed_lock(config: LifecycleScheduleConfig) -> Iterator[bool]:
    backend = _history_backend()
    if backend == "postgres":
        with _postgres_advisory_lock() as locked:
            yield locked
        return
    with _file_lock(config.lock_ttl_seconds) as locked:
        yield locked


def _skipped_record(
    *,
    run_id: str,
    run_key: str,
    config: LifecycleScheduleConfig,
    reason: str,
) -> LifecycleRunRecord:
    return LifecycleRunRecord(
        run_id=run_id,
        run_key=run_key,
        job_name=JOB_NAME,
        status="skipped",
        attempt=0,
        max_attempts=config.max_attempts,
        started_at=_iso_now(),
        finished_at=_iso_now(),
        locked=False,
        skipped_reason=reason,
        backend=_history_backend(),
        metadata={"cron": config.cron},
    )


def run_memory_lifecycle_scheduled(*, force: bool = False) -> LifecycleRunRecord:
    """带调度治理地运行 Memory Lifecycle Job。

    中文注释：
    这个函数解决的是“后台任务工程化”：
    - cron 配置：记录当前 job 应该由什么 cron 触发。
    - 任务锁：同一时间只允许一个 lifecycle job 跑。
    - 幂等运行记录：同一个 run_key 成功后不重复执行。
    - 失败重试：失败时按配置重试。
    - 运行历史：每次 skipped/success/failed 都落库或写 JSONL。
    - 分布式并发保护：Postgres backend 使用 advisory lock。
    """

    load_project_env()
    config = _schedule_config(force=force)
    run_key = _run_key(config)
    run_id = uuid.uuid4().hex

    if not force and _successful_run_exists(run_key):
        record = _skipped_record(
            run_id=run_id,
            run_key=run_key,
            config=config,
            reason="同一个 run_key 已经成功执行过，按幂等规则跳过。",
        )
        _save_run_record(record)
        return record

    with _distributed_lock(config) as locked:
        if not locked:
            record = _skipped_record(
                run_id=run_id,
                run_key=run_key,
                config=config,
                reason="另一个 lifecycle job 正在运行，当前任务跳过。",
            )
            _save_run_record(record)
            return record

        last_error = ""
        started_at = _iso_now()
        for attempt in range(1, config.max_attempts + 1):
            try:
                report: LifecycleReport = run_memory_lifecycle_job()
                record = LifecycleRunRecord(
                    run_id=run_id,
                    run_key=run_key,
                    job_name=JOB_NAME,
                    status="success",
                    attempt=attempt,
                    max_attempts=config.max_attempts,
                    started_at=started_at,
                    finished_at=_iso_now(),
                    locked=True,
                    backend=report.backend,
                    report=report.as_dict(),
                    metadata={
                        "cron": config.cron,
                        "history_backend": _history_backend(),
                    },
                )
                _save_run_record(record)
                return record
            except Exception as exc:
                last_error = str(exc)
                if attempt < config.max_attempts:
                    time.sleep(config.retry_backoff_seconds)

        record = LifecycleRunRecord(
            run_id=run_id,
            run_key=run_key,
            job_name=JOB_NAME,
            status="failed",
            attempt=config.max_attempts,
            max_attempts=config.max_attempts,
            started_at=started_at,
            finished_at=_iso_now(),
            locked=True,
            backend=_history_backend(),
            error=last_error,
            metadata={"cron": config.cron},
        )
        _save_run_record(record)
        return record


def memory_lifecycle_scheduled_report_json(*, force: bool = False) -> str:
    """运行带调度治理的 lifecycle job，并输出 JSON。"""

    return json.dumps(
        run_memory_lifecycle_scheduled(force=force).as_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
