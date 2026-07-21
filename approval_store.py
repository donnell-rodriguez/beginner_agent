from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from .config import PROJECT_DIR, load_project_env


ApprovalStatus = Literal["pending", "approved", "denied", "expired"]


load_project_env()


def _utc_now() -> datetime:
    """返回 UTC 时间。

    中文注释：
    审批超时、审计日志、跨机器记录都应该用 UTC。
    这样不会被本地时区变化影响。
    """

    return datetime.now(timezone.utc)


def _json_dumps(value: Any) -> str:
    """把审批 payload / 参数稳定地保存成 JSON 字符串。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str) -> Any:
    """读取 SQLite 里的 JSON 字符串。"""

    if not value:
        return {}
    return json.loads(value)


def approval_db_path() -> Path:
    """返回审批数据库路径。

    中文注释：
    这是本地开发版的审批持久化。
    默认写到项目内 `.agent_state/approvals.sqlite3`，
    生产级可以换成 Postgres 表，但上层 CLI/API 不需要改。
    """

    configured = os.getenv("BEGINNER_AGENT_APPROVAL_DB_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else PROJECT_DIR / path
    return PROJECT_DIR / ".agent_state" / "approvals.sqlite3"


def default_approver_id() -> str:
    """读取默认审批人身份。"""

    return os.getenv("BEGINNER_AGENT_APPROVER_ID", "local-cli-user").strip()


def default_timeout_seconds() -> int:
    """读取默认审批超时时间。"""

    raw = os.getenv("BEGINNER_AGENT_APPROVAL_TIMEOUT_SECONDS", "300").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 300


@dataclass(frozen=True)
class ApprovalRecord:
    """一条审批请求记录。

    中文注释：
    你可以把它理解成“审批工单”：
    - payload：原始审批内容。
    - status：pending / approved / denied / expired。
    - approver_id：谁审批的。
    - modified_tool_args：审批人是否修改了工具参数。
    """

    approval_id: str
    thread_id: str
    task_id: str
    status: ApprovalStatus
    approver_id: str
    requested_at: str
    expires_at: str
    decided_at: str
    payload: dict[str, Any]
    reason: str
    modified_tool_args: dict[str, Any]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ApprovalRecord":
        """从 SQLite row 转成 Python 对象。"""

        return cls(
            approval_id=str(row["approval_id"]),
            thread_id=str(row["thread_id"]),
            task_id=str(row["task_id"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            approver_id=str(row["approver_id"]),
            requested_at=str(row["requested_at"]),
            expires_at=str(row["expires_at"]),
            decided_at=str(row["decided_at"]),
            payload=dict(_json_loads(str(row["payload_json"]))),
            reason=str(row["reason"]),
            modified_tool_args=dict(_json_loads(str(row["modified_tool_args_json"]))),
        )

    def to_resume_value(self) -> dict[str, Any]:
        """转换成 Command(resume=...) 需要的数据。"""

        return {
            "approved": self.status == "approved",
            "approval_id": self.approval_id,
            "task_id": self.task_id,
            "approver_id": self.approver_id,
            "reason": self.reason,
            "modified_tool_args": self.modified_tool_args,
        }


class ApprovalStore:
    """审批持久化与审计日志。

    中文注释：
    这一层不依赖 LangGraph。
    CLI、HTTP API、未来 Web UI 都可以共用它。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else approval_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    def _connect(self) -> sqlite3.Connection:
        """创建 SQLite 连接。"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup(self) -> None:
        """初始化审批表和审计表。"""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_requests (
                    approval_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    approver_id TEXT NOT NULL DEFAULT '',
                    requested_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    decided_at TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    modified_tool_args_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    approval_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    event_at TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )

    def _audit(
        self,
        approval_id: str,
        event_type: str,
        actor_id: str,
        details: dict[str, Any],
    ) -> None:
        """写一条审批审计日志。"""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approval_audit_events (
                    approval_id, event_type, actor_id, event_at, details_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    event_type,
                    actor_id,
                    _utc_now().isoformat(),
                    _json_dumps(details),
                ),
            )

    def create_or_update_request(
        self,
        payload: dict[str, Any],
        *,
        thread_id: str,
        timeout_seconds: int | None = None,
    ) -> ApprovalRecord:
        """保存 pending 审批请求。

        中文注释：
        graph 每次 interrupt 时，CLI 会先调用这个函数。
        这样即使用户还没决定，审批请求也已经落库，可以被 API/UI 看到。
        """

        base_approval_id = str(payload.get("approval_id") or f"approval-{payload['task_id']}")
        approval_id = base_approval_id
        task_id = str(payload.get("task_id", ""))
        now = _utc_now()
        timeout = timeout_seconds or default_timeout_seconds()
        expires_at = now + timedelta(seconds=timeout)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT status FROM approval_requests WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if existing and existing["status"] != "pending":
                # 中文注释：
                # policy.py 生成的 approval_id 对同一个 task/tool 是稳定的。
                # 但不同 graph thread 可能重复出现 root.1/read_file。
                # 如果旧审批已经结束，就生成一个新的持久化 id，避免串单。
                approval_id = f"{base_approval_id}-{uuid.uuid4().hex[:8]}"

            normalized_payload = {**payload, "approval_id": approval_id}

            conn.execute(
                """
                INSERT INTO approval_requests (
                    approval_id, thread_id, task_id, status, requested_at,
                    expires_at, payload_json
                )
                VALUES (?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(approval_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    task_id = excluded.task_id,
                    status = 'pending',
                    requested_at = excluded.requested_at,
                    expires_at = excluded.expires_at,
                    payload_json = excluded.payload_json
                """,
                (
                    approval_id,
                    thread_id,
                    task_id,
                    now.isoformat(),
                    expires_at.isoformat(),
                    _json_dumps(normalized_payload),
                ),
            )

        self._audit(
            approval_id,
            "requested",
            "agent",
            {"thread_id": thread_id, "timeout_seconds": timeout},
        )
        return self.get(approval_id)  # type: ignore[return-value]

    def get(self, approval_id: str) -> ApprovalRecord | None:
        """读取审批记录，如果过期会自动标记 expired。"""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approval_requests WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        record = ApprovalRecord.from_row(row)
        if record.status == "pending" and record.expires_at <= _utc_now().isoformat():
            return self.mark_expired(approval_id)
        return record

    def list(self, *, status: str | None = None, limit: int = 20) -> list[ApprovalRecord]:
        """列出最近审批请求。"""

        query = "SELECT * FROM approval_requests"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY requested_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        records = [ApprovalRecord.from_row(row) for row in rows]
        return [self.get(record.approval_id) or record for record in records]

    def decide(
        self,
        approval_id: str,
        *,
        approved: bool,
        approver_id: str,
        reason: str,
        modified_tool_args: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        """批准或拒绝审批请求。"""

        current = self.get(approval_id)
        if current is None:
            raise KeyError(f"审批请求不存在：{approval_id}")
        if current.status != "pending":
            return current

        status: ApprovalStatus = "approved" if approved else "denied"
        args = modified_tool_args or {}
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE approval_requests
                SET status = ?, approver_id = ?, decided_at = ?,
                    reason = ?, modified_tool_args_json = ?
                WHERE approval_id = ?
                """,
                (
                    status,
                    approver_id,
                    _utc_now().isoformat(),
                    reason,
                    _json_dumps(args),
                    approval_id,
                ),
            )

        self._audit(
            approval_id,
            "decided",
            approver_id,
            {"approved": approved, "reason": reason, "modified_tool_args": args},
        )
        return self.get(approval_id)  # type: ignore[return-value]

    def mark_expired(self, approval_id: str) -> ApprovalRecord:
        """把 pending 审批标记为 expired。"""

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE approval_requests
                SET status = 'expired', decided_at = ?, reason = ?
                WHERE approval_id = ? AND status = 'pending'
                """,
                (
                    _utc_now().isoformat(),
                    "审批超时，自动拒绝。",
                    approval_id,
                ),
            )
        self._audit(approval_id, "expired", "system", {"reason": "timeout"})
        return self.get(approval_id)  # type: ignore[return-value]

    def wait_for_decision(
        self,
        approval_id: str,
        *,
        timeout_seconds: int | None = None,
        poll_seconds: float = 1.0,
    ) -> ApprovalRecord:
        """等待外部 API/UI 写入审批结果。"""

        deadline = time.monotonic() + (timeout_seconds or default_timeout_seconds())
        while time.monotonic() <= deadline:
            record = self.get(approval_id)
            if record is None:
                raise KeyError(f"审批请求不存在：{approval_id}")
            if record.status != "pending":
                return record
            time.sleep(poll_seconds)
        return self.mark_expired(approval_id)

    def audit_events(self, approval_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """读取某个审批请求的审计日志。"""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM approval_audit_events
                WHERE approval_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (approval_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "approval_id": row["approval_id"],
                "event_type": row["event_type"],
                "actor_id": row["actor_id"],
                "event_at": row["event_at"],
                "details": _json_loads(row["details_json"]),
            }
            for row in rows
        ]
