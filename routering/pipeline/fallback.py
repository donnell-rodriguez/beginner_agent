from __future__ import annotations

from typing import cast

from ...state import RiskLevel, TaskType
from ..failure_policy import (
    RouterFailurePolicy,
    conservative_risk_level,
    load_router_failure_policy,
)
from ..rules import RouterRuleSet
from .models import RouterStageDecision


# 中文注释：
# fallback.py 集中放子 Router 失败后的兜底逻辑。
# 这样每个阶段失败时怎么保守处理，有一个统一位置可以审计。


def fallback_intent(
    text: str,
    rules: RouterRuleSet,
    fallback_reason: str,
    model_response: str,
) -> RouterStageDecision:
    rule_decision = rules.explain_task_type(text)
    return RouterStageDecision(
        stage="intent_router",
        decision=rule_decision.outcome,
        reason=f"Intent Router 兜底规则：{rule_decision.selected_rule_reason}",
        confidence=0.45,
        source="fallback",
        model_response=model_response,
        model_error=fallback_reason,
        fallback_reason=fallback_reason,
    )


def fallback_risk(
    text: str,
    rules: RouterRuleSet,
    fallback_reason: str,
    model_response: str,
    *,
    failure_policy: RouterFailurePolicy | None = None,
) -> RouterStageDecision:
    policy = failure_policy or load_router_failure_policy()
    if policy.risk_failure_policy == "conservative":
        risk_level, risk_reason = conservative_risk_level(text, rules)
        policy_applied = "risk_conservative_fallback"
    else:
        rule_decision = rules.explain_risk_level(text)
        risk_level = cast(RiskLevel, rule_decision.outcome)
        risk_reason = rule_decision.selected_rule_reason
        policy_applied = "risk_rule_fallback"
    return RouterStageDecision(
        stage="risk_router",
        decision=risk_level,
        reason=f"Risk Router 兜底规则：{risk_reason}",
        confidence=0.45,
        source="fallback",
        model_response=model_response,
        model_error=fallback_reason,
        fallback_reason=fallback_reason,
        failure_policy_applied=policy_applied,
    )


def fallback_tool_needs(
    intent: TaskType,
    fallback_reason: str,
    model_response: str,
) -> RouterStageDecision:
    needs_tool = intent == "agent"
    return RouterStageDecision(
        stage="tool_needs_router",
        decision=str(needs_tool).lower(),
        reason="Tool Needs Router 兜底规则：agent 分支需要工具，其他简单分支默认不需要。",
        confidence=0.45,
        source="fallback",
        model_response=model_response,
        model_error=fallback_reason,
        fallback_reason=fallback_reason,
    )
