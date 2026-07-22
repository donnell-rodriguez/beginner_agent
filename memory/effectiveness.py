from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .feedback import MemoryFeedbackEvent, append_memory_feedback
from .settings import MEMORY_DIR
from ..state import State
from ..tooling.core import ensure_state_dirs


# 中文注释：
# effectiveness.py 负责“Memory 使用效果闭环”。
#
# 生产级 memory 不是“检索出来就结束”。
# 还要回答：
#
#   这条 memory 被哪次 run 用过？
#   用完后任务是否成功？
#   它是否可能帮助了任务？
#   它是否可能误导了 agent？
#
# 当前实现先用本地 JSONL 保存 usage event。
# 后续可以把同样模型写到 Postgres / Kafka / 数据仓库。


MemoryUsageOutcome = Literal["pending", "helped", "neutral", "hurt"]

MEMORY_USAGE_FILE = MEMORY_DIR / "memory_usage.jsonl"
MAX_MEMORY_USAGE_EVENTS = 3000


@dataclass(frozen=True)
class MemoryUsageEvent:
    """一条 memory 被使用后的效果记录。"""

    memory_id: str
    run_id: str
    task_id: str
    outcome: MemoryUsageOutcome
    reason: str
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    tool_result_status: str = "none"
    done: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
            "tool_result_status": self.tool_result_status,
            "done": self.done,
            "created_at": self.created_at,
        }


def append_memory_usage(event: MemoryUsageEvent) -> None:
    """追加一条 memory usage event。"""

    _ensure_usage_file()
    events = read_memory_usage(MAX_MEMORY_USAGE_EVENTS)
    events.append(event.as_dict())
    MEMORY_USAGE_FILE.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in events[-MAX_MEMORY_USAGE_EVENTS:]
        ),
        encoding="utf-8",
    )


def read_memory_usage(limit: int = MAX_MEMORY_USAGE_EVENTS) -> list[dict[str, Any]]:
    """读取最近的 memory usage events。"""

    _ensure_usage_file()
    events: list[dict[str, Any]] = []
    for line in MEMORY_USAGE_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events[-limit:]


def record_retrieved_memory_usage(
    state: State,
    records: list[dict[str, Any]],
) -> None:
    """Memory Retriever 召回后先记录 pending usage。

    中文注释：
    这一步只说明“这些 memory 被放进上下文候选/结果里了”。
    后续 Memory Writer 根据 run 是否完成，再补 helped / neutral / hurt。
    """

    for record in records:
        memory_id = str(record.get("id", ""))
        if not memory_id:
            continue
        append_memory_usage(
            MemoryUsageEvent(
                memory_id=memory_id,
                run_id=str(state.get("run_id", "")),
                task_id=str(state.get("current_task_id", "")),
                outcome="pending",
                reason="Memory Retriever 将该记忆放入 memory_context。",
                retrieval_score=_float_value(record.get("retrieval_score")),
                rerank_score=_float_value(record.get("rerank_score")),
                tool_result_status=str(state.get("tool_result_status", "none")),
                done=bool(state.get("done", False)),
            )
        )


def close_memory_usage_loop(state: State) -> dict[str, Any]:
    """根据当前 run 结果，把 pending usage 转成反馈信号。

    中文注释：
    这是“使用效果 -> feedback -> quality/trust”的桥。
    - run 完成且工具成功：认为相关 memory 可能 helped。
    - 工具失败/阻塞：先记 neutral，避免轻率惩罚 memory。
    - 明确 recovery_action=replan/retry：认为可能 hurt，降低信任。
    """

    run_id = str(state.get("run_id", ""))
    if not run_id:
        return {"updated": 0, "reason": "缺少 run_id。"}
    all_events = read_memory_usage()
    closed_keys = {
        (str(event.get("run_id", "")), str(event.get("memory_id", "")))
        for event in all_events
        if str(event.get("run_id", "")) == run_id and event.get("outcome") != "pending"
    }
    pending = [
        event
        for event in all_events
        if str(event.get("run_id", "")) == run_id and event.get("outcome") == "pending"
        and (run_id, str(event.get("memory_id", ""))) not in closed_keys
    ]
    if not pending:
        return {"updated": 0, "reason": "没有 pending memory usage。"}

    outcome, signal, reason = _outcome_for_state(state)
    updated = 0
    for event in pending:
        memory_id = str(event.get("memory_id", ""))
        if not memory_id:
            continue
        append_memory_usage(
            MemoryUsageEvent(
                memory_id=memory_id,
                run_id=run_id,
                task_id=str(event.get("task_id", state.get("current_task_id", ""))),
                outcome=outcome,
                reason=reason,
                retrieval_score=_float_value(event.get("retrieval_score")),
                rerank_score=_float_value(event.get("rerank_score")),
                tool_result_status=str(state.get("tool_result_status", "none")),
                done=bool(state.get("done", False)),
            )
        )
        append_memory_feedback(
            MemoryFeedbackEvent(
                memory_id=memory_id,
                signal=signal,
                reason=f"usage_effectiveness: {reason}",
                run_id=run_id,
                task_id=str(event.get("task_id", state.get("current_task_id", ""))),
                actor_id="memory_effectiveness_loop",
            )
        )
        updated += 1
    return {"updated": updated, "outcome": outcome, "feedback_signal": signal}


def summarize_memory_usage(memory_id: str = "", *, limit: int = 500) -> dict[str, Any]:
    """聚合 memory usage 统计。"""

    events = read_memory_usage(limit)
    if memory_id:
        events = [event for event in events if str(event.get("memory_id", "")) == memory_id]
    counts: dict[str, int] = {"pending": 0, "helped": 0, "neutral": 0, "hurt": 0}
    for event in events:
        outcome = str(event.get("outcome", "neutral"))
        if outcome in counts:
            counts[outcome] += 1
    total_closed = counts["helped"] + counts["neutral"] + counts["hurt"]
    return {
        "memory_id": memory_id,
        "event_count": len(events),
        "counts": counts,
        "help_rate": counts["helped"] / total_closed if total_closed else 0.0,
        "hurt_rate": counts["hurt"] / total_closed if total_closed else 0.0,
        "recent_events": events[-20:],
    }


def _outcome_for_state(state: State) -> tuple[MemoryUsageOutcome, str, str]:
    tool_status = str(state.get("tool_result_status", "none"))
    recovery_action = str(state.get("recovery_action", "none"))
    if recovery_action in {"retry", "replan"}:
        return "hurt", "not_useful", "本轮触发 retry/replan，相关记忆可能没有帮助。"
    if state.get("done") and tool_status in {"success", "none"}:
        return "helped", "useful", "本轮任务完成，相关记忆被视为正向信号。"
    if tool_status in {"failed", "blocked"}:
        return "neutral", "not_useful", "工具失败/阻塞，先记录弱负反馈。"
    return "neutral", "confirmed", "本轮未明确证明好坏，记录中性确认。"


def _ensure_usage_file() -> None:
    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_USAGE_FILE.exists():
        MEMORY_USAGE_FILE.write_text("", encoding="utf-8")


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
