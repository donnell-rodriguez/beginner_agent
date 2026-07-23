from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import RouterDecision, RouterSecuritySignal
from .rules import RouterRuleSet


# 中文注释：
# conflicts.py 负责检测 Router 内部不同判断来源是否打架。
# 例如：
# - LLM 认为 chat，但规则认为 agent。
# - LLM 认为 low，但规则/security 认为 high。
# - 最终 decision 和 security signal 不一致。


@dataclass(frozen=True)
class RouterConflict:
    kind: str
    severity: str
    expected: str
    actual: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "expected": self.expected,
            "actual": self.actual,
            "reason": self.reason,
        }


def detect_router_conflicts(
    *,
    text: str,
    decision: RouterDecision,
    rules: RouterRuleSet,
    security: RouterSecuritySignal,
    context_policy_reason: str,
    low_confidence_threshold: float,
) -> tuple[RouterConflict, ...]:
    """检测 Router 决策中的冲突。"""

    conflicts: list[RouterConflict] = []
    rule_task = rules.explain_task_type(text)
    rule_risk = rules.explain_risk_level(text)

    if rule_task.matched and decision.task_type != rule_task.outcome:
        conflicts.append(
            RouterConflict(
                kind="task_type_rule_conflict",
                severity="medium",
                expected=rule_task.outcome,
                actual=decision.task_type,
                reason=f"规则 {rule_task.selected_rule_id} 判断为 {rule_task.outcome}。",
            )
        )

    if rule_risk.matched and decision.risk_level != rule_risk.outcome:
        conflicts.append(
            RouterConflict(
                kind="risk_rule_conflict",
                severity="high" if rule_risk.outcome == "high" else "medium",
                expected=rule_risk.outcome,
                actual=decision.risk_level,
                reason=f"规则 {rule_risk.selected_rule_id} 判断为 {rule_risk.outcome}。",
            )
        )

    if security.malicious_intent != "none" and decision.risk_level != "high":
        conflicts.append(
            RouterConflict(
                kind="security_risk_conflict",
                severity="high",
                expected="high",
                actual=decision.risk_level,
                reason=f"安全分类命中 {security.malicious_intent}，风险不应低于 high。",
            )
        )

    if decision.confidence < low_confidence_threshold:
        conflicts.append(
            RouterConflict(
                kind="low_confidence",
                severity="medium",
                expected=f">={low_confidence_threshold:.2f}",
                actual=f"{decision.confidence:.2f}",
                reason="Router 最终置信度低于阈值，需要进入复核或更保守策略。",
            )
        )

    if context_policy_reason and decision.risk_level != "high":
        conflicts.append(
            RouterConflict(
                kind="context_policy_conflict",
                severity="high",
                expected="high",
                actual=decision.risk_level,
                reason=context_policy_reason,
            )
        )

    return tuple(conflicts)
