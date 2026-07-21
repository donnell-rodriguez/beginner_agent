from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .memory_settings import (
    MAX_MEMORY_FEEDBACK_EVENTS,
    MEMORY_DIR,
    MEMORY_FEEDBACK_FILE,
)
from .tooling.core import ensure_state_dirs


FeedbackSignal = Literal["useful", "not_useful", "harmful", "fixed_later", "confirmed"]


@dataclass(frozen=True)
class MemoryFeedbackEvent:
    """人工或系统反馈事件。

    中文注释：
    大厂 memory 系统不会只看“写入时的规则分数”。
    后续任务、人类审核、测试结果都可以反过来证明：
    - 这条记忆有用。
    - 这条记忆没用。
    - 这条记忆误导了 agent。

    这些反馈会影响 trust / quality / rerank。
    """

    memory_id: str
    signal: FeedbackSignal
    reason: str
    run_id: str = ""
    task_id: str = ""
    actor_id: str = "local-system"
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "signal": self.signal,
            "reason": self.reason,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "actor_id": self.actor_id,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }


def _ensure_feedback_file() -> None:
    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FEEDBACK_FILE.exists():
        MEMORY_FEEDBACK_FILE.write_text("", encoding="utf-8")


def append_memory_feedback(event: MemoryFeedbackEvent) -> None:
    """追加一条 memory feedback。

    中文注释：
    当前实现是 JSONL，本地可运行。
    后续可以把同样接口换成 Postgres 表或在线反馈系统。
    """

    _ensure_feedback_file()
    events = read_memory_feedback(MAX_MEMORY_FEEDBACK_EVENTS)
    events.append(event.as_dict())
    trimmed = events[-MAX_MEMORY_FEEDBACK_EVENTS:]
    MEMORY_FEEDBACK_FILE.write_text(
        "".join(
            f"{json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            for item in trimmed
        ),
        encoding="utf-8",
    )


def read_memory_feedback(limit: int = MAX_MEMORY_FEEDBACK_EVENTS) -> list[dict[str, Any]]:
    """读取最近的 feedback events。"""

    _ensure_feedback_file()
    events: list[dict[str, Any]] = []
    for line in MEMORY_FEEDBACK_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events[-limit:]


def feedback_summary_for_memory(memory_id: str) -> dict[str, Any]:
    """聚合某条 memory 的反馈信号。"""

    summary = {
        "useful": 0,
        "not_useful": 0,
        "harmful": 0,
        "fixed_later": 0,
        "confirmed": 0,
    }
    recent_reasons: list[str] = []
    for event in read_memory_feedback():
        if str(event.get("memory_id", "")) != memory_id:
            continue
        signal = str(event.get("signal", ""))
        if signal in summary:
            summary[signal] += 1
        reason = str(event.get("reason", "")).strip()
        if reason:
            recent_reasons.append(reason[:200])
    positive = summary["useful"] + summary["confirmed"] + summary["fixed_later"]
    negative = summary["not_useful"] + (summary["harmful"] * 2)
    return {
        **summary,
        "positive": positive,
        "negative": negative,
        "net": positive - negative,
        "recent_reasons": recent_reasons[-3:],
    }

