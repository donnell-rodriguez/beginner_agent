from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

from ..state import RiskLevel, TaskType
from .models import RouterEvalCase
from .observability import (
    append_router_eval_case,
    append_router_feedback_event,
    read_router_events,
    read_router_feedback_events,
)


# 中文注释：
# feedback.py 负责 Router 的人工反馈闭环。
#
# 大厂不会只做“离线 eval case 文件”，还会把线上错误持续回流：
#
#     某次 Router 决策错了
#       -> 用户/开发者提交纠错
#       -> 系统记录 RouterFeedbackEvent
#       -> 自动生成 RouterEvalCase
#       -> 下一次 replay 覆盖这个错误场景
#
# 这个模块只处理 Router 反馈，不掺杂 Memory/Tool/Graph 逻辑。


@dataclass(frozen=True)
class RouterFeedbackEvent:
    """一次针对 Router 错误决策的人工纠错事件。"""

    feedback_id: str
    user_input: str
    expected_task_type: TaskType
    expected_risk_level: RiskLevel
    expected_needs_tool: bool
    correction_reason: str

    # 中文注释：
    # decision_id/run_id 用来把 feedback 追溯到某一次真实 RouterEvent。
    # 如果是从历史样本手动补录，可以为空。
    decision_id: str = ""
    run_id: str = ""

    # 中文注释：
    # actual_* 保存当时 Router 的错误输出。
    # 这让后续分析可以知道它到底错在 intent、risk 还是 tool_needs。
    actual_task_type: str = ""
    actual_risk_level: str = ""
    actual_needs_tool: bool | None = None

    source: str = "manual_feedback"
    actor_id: str = "local-user"
    eval_case_created: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "user_input": self.user_input,
            "actual_task_type": self.actual_task_type,
            "actual_risk_level": self.actual_risk_level,
            "actual_needs_tool": self.actual_needs_tool,
            "expected_task_type": self.expected_task_type,
            "expected_risk_level": self.expected_risk_level,
            "expected_needs_tool": self.expected_needs_tool,
            "correction_reason": self.correction_reason,
            "source": self.source,
            "actor_id": self.actor_id,
            "eval_case_created": self.eval_case_created,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class RouterFeedbackResult:
    """一次反馈闭环写入结果。"""

    event: RouterFeedbackEvent
    eval_case: RouterEvalCase
    duplicate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "duplicate": self.duplicate,
            "event": self.event.as_dict(),
            "eval_case": self.eval_case.as_dict(),
        }


def record_router_correction(
    *,
    expected_task_type: str,
    expected_risk_level: str,
    expected_needs_tool: bool,
    correction_reason: str,
    router_report: dict[str, Any] | None = None,
    decision_id: str = "",
    user_input: str = "",
    source: str = "manual_feedback",
    actor_id: str = "local-user",
) -> RouterFeedbackResult:
    """记录一次 Router 人工纠错，并自动生成 eval case。

    中文注释：
    推荐传 router_report，因为它包含：
    - decision_id
    - run_id
    - user_input
    - 当时的 decision

    如果手上只有 decision_id，也可以通过历史 router events 反查。
    """

    _validate_expected(expected_task_type, expected_risk_level)
    report = router_report or _find_router_report(decision_id)
    extracted = _extract_router_report(report)
    resolved_user_input = user_input or extracted["user_input"]
    if not resolved_user_input:
        raise ValueError("Router feedback 必须提供 user_input 或可追溯的 router_report。")

    event = RouterFeedbackEvent(
        feedback_id=_feedback_id(
            decision_id=extracted["decision_id"] or decision_id,
            user_input=resolved_user_input,
            expected_task_type=expected_task_type,
            expected_risk_level=expected_risk_level,
            expected_needs_tool=expected_needs_tool,
        ),
        decision_id=extracted["decision_id"] or decision_id,
        run_id=extracted["run_id"],
        user_input=resolved_user_input,
        actual_task_type=extracted["actual_task_type"],
        actual_risk_level=extracted["actual_risk_level"],
        actual_needs_tool=extracted["actual_needs_tool"],
        expected_task_type=cast(TaskType, expected_task_type),
        expected_risk_level=cast(RiskLevel, expected_risk_level),
        expected_needs_tool=expected_needs_tool,
        correction_reason=correction_reason,
        source=source,
        actor_id=actor_id,
    )
    case = _feedback_event_to_eval_case(event)
    if _feedback_already_recorded(event.feedback_id):
        return RouterFeedbackResult(event=event, eval_case=case, duplicate=True)

    append_router_feedback_event(event.as_dict())
    append_router_eval_case(case)
    return RouterFeedbackResult(event=event, eval_case=case)


def read_router_feedback(limit: int | None = None) -> list[dict[str, Any]]:
    """读取 Router 人工纠错反馈。"""

    return read_router_feedback_events(limit)


def _feedback_event_to_eval_case(event: RouterFeedbackEvent) -> RouterEvalCase:
    return RouterEvalCase(
        user_input=event.user_input,
        expected_task_type=event.expected_task_type,
        expected_risk_level=event.expected_risk_level,
        expected_needs_tool=event.expected_needs_tool,
        reason=(
            f"{event.source}: {event.correction_reason}; "
            f"feedback_id={event.feedback_id}; "
            f"decision_id={event.decision_id or 'none'}; "
            f"actual={event.actual_task_type}/{event.actual_risk_level}/{event.actual_needs_tool}"
        ),
        category="regression_cases",
        tags=("feedback", event.source),
        created_at=event.created_at,
    )


def _find_router_report(decision_id: str) -> dict[str, Any]:
    if not decision_id:
        return {}
    for event in reversed(read_router_events()):
        if str(event.get("decision_id", "")) == decision_id:
            return event
    raise ValueError(f"找不到 decision_id={decision_id} 对应的 RouterEvent。")


def _extract_router_report(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    decision = report.get("decision", {})
    if not isinstance(decision, dict):
        decision = {}
    return {
        "decision_id": str(report.get("decision_id", "")),
        "run_id": str(report.get("run_id", "")),
        "user_input": str(report.get("user_input", "")),
        "actual_task_type": str(decision.get("task_type", "")),
        "actual_risk_level": str(decision.get("risk_level", "")),
        "actual_needs_tool": _optional_bool(decision.get("needs_tool")),
    }


def _feedback_already_recorded(feedback_id: str) -> bool:
    return any(str(item.get("feedback_id", "")) == feedback_id for item in read_router_feedback())


def _feedback_id(
    *,
    decision_id: str,
    user_input: str,
    expected_task_type: str,
    expected_risk_level: str,
    expected_needs_tool: bool,
) -> str:
    raw = json.dumps(
        {
            "decision_id": decision_id,
            "user_input": user_input,
            "expected_task_type": expected_task_type,
            "expected_risk_level": expected_risk_level,
            "expected_needs_tool": expected_needs_tool,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _validate_expected(task_type: str, risk_level: str) -> None:
    if task_type not in {"search", "write", "chat", "agent"}:
        raise ValueError(f"Invalid expected_task_type: {task_type}")
    if risk_level not in {"low", "medium", "high"}:
        raise ValueError(f"Invalid expected_risk_level: {risk_level}")


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None
