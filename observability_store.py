from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_DIR, load_project_env


load_project_env()


def observability_db_path() -> Path:
    """返回 observability SQLite 路径。"""

    configured = os.getenv(
        "BEGINNER_AGENT_OBSERVABILITY_DB_PATH",
        ".agent_state/observability.sqlite3",
    ).strip()
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


class ObservabilityStore:
    """本地 observability 事件库。

    中文注释：
    大厂里通常会把这些数据写到 metrics/logs/traces 平台。
    当前项目先用 SQLite 落地：
    - 每轮 loop 一条 report。
    - 能按 run_id 查询。
    - 后续可以迁移到 OpenTelemetry / Prometheus / ClickHouse / Postgres。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else observability_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    def _connect(self) -> sqlite3.Connection:
        """创建 SQLite 连接。"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup(self) -> None:
        """初始化 observability 表。"""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observability_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_count INTEGER NOT NULL,
                    done INTEGER NOT NULL,
                    next_action TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_observability_reports_run_id
                ON observability_reports(run_id, id)
                """
            )

    def record_report(self, *, run_id: str, report: dict[str, Any]) -> dict[str, Any]:
        """写入一条 observability report。"""

        recorded_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO observability_reports (
                    run_id, step_count, done, next_action, recorded_at, report_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(report.get("step_count", 0)),
                    1 if report.get("done") else 0,
                    str(report.get("next_action", "")),
                    recorded_at,
                    json.dumps(report, ensure_ascii=False, sort_keys=True),
                ),
            )
        return {
            "storage": "sqlite",
            "db_path": self.db_path.as_posix(),
            "record_id": cursor.lastrowid,
            "recorded_at": recorded_at,
        }

    def recent_reports(self, *, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """读取某次 run 最近的 observability report。"""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM observability_reports
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [json.loads(row["report_json"]) for row in rows]
