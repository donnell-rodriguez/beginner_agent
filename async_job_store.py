from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import PROJECT_DIR, load_project_env


AsyncJobStatus = Literal["queued", "running", "success", "failed", "cancelled", "timeout"]


load_project_env()


def async_job_db_path() -> Path:
    """返回本地 async job 数据库路径。"""

    configured = os.getenv(
        "BEGINNER_AGENT_ASYNC_JOB_DB_PATH",
        ".agent_state/async_jobs.sqlite3",
    ).strip()
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def async_job_timeout_seconds() -> int:
    """读取异步 job 最大等待时间。"""

    raw = os.getenv("BEGINNER_AGENT_ASYNC_JOB_TIMEOUT_SECONDS", "300").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 300


def async_job_poll_seconds() -> float:
    """读取异步 job 轮询间隔。"""

    raw = os.getenv("BEGINNER_AGENT_ASYNC_JOB_POLL_SECONDS", "1.0").strip()
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 1.0


class AsyncJobStore:
    """本地异步 job 状态库。

    中文注释：
    这不是完整远程 worker。
    它提供的是 worker contract：
    - Executor 如果未来提交远程 job，可以写 job_id。
    - Worker 更新 queued/running/success/failed。
    - Async Job Waiter 根据 job_id 轮询。

    这样 graph.py 不需要知道远程 worker 是 Celery、Kafka、Prefect 还是自研系统。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else async_job_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    def _connect(self) -> sqlite3.Connection:
        """创建 SQLite 连接。"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup(self) -> None:
        """初始化 job 表。"""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS async_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )

    def upsert_job(
        self,
        *,
        job_id: str,
        status: AsyncJobStatus,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        """创建或更新 job 状态。"""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO async_jobs (job_id, status, result_json, error, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    result_json = excluded.result_json,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    status,
                    json.dumps(result or {}, ensure_ascii=False, sort_keys=True),
                    error,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """读取 job 状态。"""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM async_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "result": json.loads(row["result_json"] or "{}"),
            "error": row["error"],
            "updated_at": row["updated_at"],
        }

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: int | None = None,
        poll_seconds: float | None = None,
    ) -> dict[str, Any]:
        """等待 job 进入终态。"""

        timeout = timeout_seconds or async_job_timeout_seconds()
        poll = poll_seconds or async_job_poll_seconds()
        deadline = time.monotonic() + timeout
        while time.monotonic() <= deadline:
            job = self.get_job(job_id)
            if job is None:
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": "job_id 不存在。",
                    "result": {},
                }
            if job["status"] in {"success", "failed", "cancelled"}:
                return job
            time.sleep(poll)
        self.upsert_job(job_id=job_id, status="timeout", error="等待异步 job 超时。")
        return self.get_job(job_id) or {
            "job_id": job_id,
            "status": "timeout",
            "error": "等待异步 job 超时。",
            "result": {},
        }
