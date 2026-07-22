from __future__ import annotations

from .models import RouterDecision, RouterSecuritySignal, RouterStageReport
from .rules import RouterRuleSet


def build_stage_reports(
    *,
    text: str,
    decision: RouterDecision,
    rules: RouterRuleSet,
    security: RouterSecuritySignal,
    context_policy_reason: str,
) -> list[RouterStageReport]:
    """构造多级 Router 子决策报告。

    中文注释：
    这让 Router 不只是给一个最终结果，
    还明确暴露每一层判断：
    intent -> risk -> tool_needs -> security -> context_policy。
    """

    rule_intent = rules.classify_task_type(text)
    rule_risk = rules.classify_risk_level(text)
    return [
        RouterStageReport(
            stage="intent",
            decision=decision.task_type,
            reason=f"最终 intent={decision.task_type}；规则兜底会判为 {rule_intent}。",
            confidence=decision.confidence,
        ),
        RouterStageReport(
            stage="risk",
            decision=decision.risk_level,
            reason=f"最终 risk={decision.risk_level}；规则兜底会判为 {rule_risk}。",
            confidence=decision.confidence,
        ),
        RouterStageReport(
            stage="tool_needs",
            decision=str(decision.needs_tool).lower(),
            reason="agent 分支通常需要工具；search/write/chat 通常不进入复杂工具 loop。",
            confidence=decision.confidence,
        ),
        RouterStageReport(
            stage="security",
            decision=security.malicious_intent,
            reason=security.reason,
            confidence=0.9 if security.malicious_intent != "none" else 0.7,
        ),
        RouterStageReport(
            stage="context_policy",
            decision="high_risk_override" if context_policy_reason else "none",
            reason=context_policy_reason or "未命中 tenant/project/user 路由策略。",
            confidence=0.9,
        ),
    ]
