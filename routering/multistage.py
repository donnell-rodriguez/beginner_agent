from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from pydantic import ValidationError

from ..state import RiskLevel, TaskType
from .failure_policy import RouterFailurePolicy, load_router_failure_policy
from .models import DecisionSource, RouterDecision, RouterSecuritySignal, RouterStageReport
from .prompts import RouterPromptSpec
from .rules import RouterRuleSet
from .security_classifier import run_llm_security_classifier
from .pipeline.fallback import fallback_intent, fallback_risk, fallback_tool_needs
from .pipeline.models import (
    IntentStageModel,
    MultiStageRouterResult,
    RiskStageModel,
    RouterStageDecision,
    ToolNeedsStageModel,
)
from .pipeline.repair import parse_stage_model_with_repair
from .pipeline.reporting import (
    build_multistage_reports,
    combine_model_responses,
    combine_stage_field,
)
from .pipeline.runtime import call_stage_router


# 中文注释：
# multistage.py 现在只负责“编排多个 Router 子阶段”。
#
# 每个职责已经拆到单独模块：
# - pipeline/models.py：子阶段数据结构。
# - pipeline/runtime.py：子阶段 LLM 调用。
# - pipeline/repair.py：JSON/schema repair。
# - pipeline/fallback.py：失败兜底策略。
# - pipeline/reporting.py：可观测报告聚合。
#
# 这样你阅读时可以先看本文件理解主流程，
# 再按需进入对应模块看细节。


def run_multistage_router(
    text: str,
    *,
    rules: RouterRuleSet,
    security: RouterSecuritySignal,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
) -> MultiStageRouterResult:
    """运行多个独立 Router 子阶段，并聚合成最终 RouterDecision。"""

    failure_policy = load_router_failure_policy()
    intent = _run_intent_router(
        text,
        rules=rules,
        prompt=prompt,
        chat_completion=chat_completion,
        min_confidence=min_confidence,
        failure_policy=failure_policy,
    )
    risk = _run_risk_router(
        text,
        rules=rules,
        prompt=prompt,
        chat_completion=chat_completion,
        min_confidence=min_confidence,
        failure_policy=failure_policy,
    )
    tool_needs = _run_tool_needs_router(
        text,
        intent=cast(TaskType, intent.decision),
        prompt=prompt,
        chat_completion=chat_completion,
        min_confidence=min_confidence,
        failure_policy=failure_policy,
    )
    merged_security, security_classifier_decision = run_llm_security_classifier(
        text,
        local_security=security,
        prompt=prompt,
        chat_completion=chat_completion,
    )
    security_decision = _security_stage_decision(
        merged_security,
        failure_policy=failure_policy,
    )

    stage_decisions = tuple(
        stage
        for stage in (intent, risk, tool_needs, security_classifier_decision, security_decision)
        if stage is not None
    )
    source: DecisionSource = "llm"
    if any(stage.source == "fallback" for stage in stage_decisions):
        source = "fallback"

    decision = _aggregate_decision(
        intent=intent,
        risk=risk,
        tool_needs=tool_needs,
        security=merged_security,
    )

    if merged_security.malicious_intent != "none":
        decision = decision.model_copy(
            update={
                "task_type": "agent",
                "risk_level": "high",
                "needs_tool": True,
                "reason": f"{decision.reason}；Security Router 保守升级：{merged_security.reason}",
                "confidence": min(decision.confidence, 0.65),
            }
        )
        source = "security_override"
    elif security_decision.failure_policy_applied == "security_suspected_raise_risk":
        decision = decision.model_copy(
            update={
                "task_type": "agent",
                "risk_level": "medium",
                "needs_tool": True,
                "reason": f"{decision.reason}；Security Router 疑似风险保守升级：{merged_security.reason}",
                "confidence": min(decision.confidence, 0.6),
            }
        )
        source = "security_override"

    return MultiStageRouterResult(
        decision=decision,
        stage_decisions=stage_decisions,
        source=source,
        security=merged_security,
        model_response=combine_model_responses(stage_decisions),
        model_error=combine_stage_field(stage_decisions, "model_error"),
        fallback_reason=combine_stage_field(stage_decisions, "fallback_reason"),
        failure_audit=tuple(
            audit for stage in stage_decisions if (audit := stage.failure_audit()) is not None
        ),
    )


def _aggregate_decision(
    *,
    intent: RouterStageDecision,
    risk: RouterStageDecision,
    tool_needs: RouterStageDecision,
    security: RouterSecuritySignal,
) -> RouterDecision:
    """把各个子阶段的结果聚合成 RouterDecision。"""

    stage_decisions = (intent, risk, tool_needs)
    confidence = min(stage.confidence for stage in stage_decisions)
    reason = "；".join(
        f"{stage.stage}={stage.decision}（{stage.reason}）" for stage in stage_decisions
    )
    reason = f"{reason}；security_router={security.malicious_intent}（{security.reason}）"
    return RouterDecision(
        task_type=cast(TaskType, intent.decision),
        risk_level=cast(RiskLevel, risk.decision),
        needs_tool=_bool_from_stage_decision(tool_needs.decision),
        reason=reason,
        confidence=confidence,
    )


def _run_intent_router(
    text: str,
    *,
    rules: RouterRuleSet,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
    failure_policy: RouterFailurePolicy,
) -> RouterStageDecision:
    try:
        response = call_stage_router(
            text,
            prompt=prompt,
            chat_completion=chat_completion,
            stage_title="Intent Router",
            instruction=(
                "你只负责判断 task_type。"
                "只返回 JSON："
                '{"task_type":"chat|search|write|agent","reason":"一句话原因","confidence":0.0-1.0}'
            ),
            max_tokens_env="BEGINNER_AGENT_ROUTER_INTENT_MAX_TOKENS",
            timeout_ms_env="BEGINNER_AGENT_ROUTER_INTENT_TIMEOUT_MS",
        )
        parsed, repair = parse_stage_model_with_repair(
            response,
            model_cls=IntentStageModel,
            required_fields={"task_type"},
            stage_title="Intent Router",
            schema_hint='{"task_type":"chat|search|write|agent","reason":"...","confidence":0.0-1.0}',
            prompt=prompt,
            chat_completion=chat_completion,
            failure_policy=failure_policy,
            max_tokens_env="BEGINNER_AGENT_ROUTER_INTENT_MAX_TOKENS",
        )
        if parsed.confidence < min_confidence and failure_policy.low_confidence_policy == "fallback":
            return fallback_intent(text, rules, f"置信度 {parsed.confidence:.2f} 低于阈值。", response)
        return RouterStageDecision(
            stage="intent_router",
            decision=parsed.task_type,
            reason=parsed.reason,
            confidence=parsed.confidence,
            source="llm",
            model_response=repair.final_response or response,
            repair_attempt_count=repair.attempt_count,
            repair_success=repair.success,
            raw_invalid_response=repair.raw_invalid_response,
            validation_error_type=repair.validation_error_type,
        )
    except (
        RuntimeError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        AttributeError,
        ValidationError,
    ) as exc:
        return fallback_intent(text, rules, f"{type(exc).__name__}: {exc}", "")


def _run_risk_router(
    text: str,
    *,
    rules: RouterRuleSet,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
    failure_policy: RouterFailurePolicy,
) -> RouterStageDecision:
    try:
        response = call_stage_router(
            text,
            prompt=prompt,
            chat_completion=chat_completion,
            stage_title="Risk Router",
            instruction=(
                "你只负责判断 risk_level。"
                "修改代码、删除文件、执行命令、读取敏感信息通常是 high。"
                "只返回 JSON："
                '{"risk_level":"low|medium|high","reason":"一句话原因","confidence":0.0-1.0}'
            ),
            max_tokens_env="BEGINNER_AGENT_ROUTER_RISK_MAX_TOKENS",
            timeout_ms_env="BEGINNER_AGENT_ROUTER_RISK_TIMEOUT_MS",
        )
        parsed, repair = parse_stage_model_with_repair(
            response,
            model_cls=RiskStageModel,
            required_fields={"risk_level"},
            stage_title="Risk Router",
            schema_hint='{"risk_level":"low|medium|high","reason":"...","confidence":0.0-1.0}',
            prompt=prompt,
            chat_completion=chat_completion,
            failure_policy=failure_policy,
            max_tokens_env="BEGINNER_AGENT_ROUTER_RISK_MAX_TOKENS",
        )
        if parsed.confidence < min_confidence and failure_policy.low_confidence_policy == "fallback":
            return fallback_risk(text, rules, f"置信度 {parsed.confidence:.2f} 低于阈值。", response)
        return RouterStageDecision(
            stage="risk_router",
            decision=parsed.risk_level,
            reason=parsed.reason,
            confidence=parsed.confidence,
            source="llm",
            model_response=repair.final_response or response,
            repair_attempt_count=repair.attempt_count,
            repair_success=repair.success,
            raw_invalid_response=repair.raw_invalid_response,
            validation_error_type=repair.validation_error_type,
        )
    except (
        RuntimeError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        AttributeError,
        ValidationError,
    ) as exc:
        return fallback_risk(
            text,
            rules,
            f"{type(exc).__name__}: {exc}",
            "",
            failure_policy=failure_policy,
        )


def _run_tool_needs_router(
    text: str,
    *,
    intent: TaskType,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
    failure_policy: RouterFailurePolicy,
) -> RouterStageDecision:
    try:
        response = call_stage_router(
            text,
            prompt=prompt,
            chat_completion=chat_completion,
            stage_title="Tool Needs Router",
            instruction=(
                "你只负责判断 needs_tool。"
                "需要读取文件、搜索代码、运行测试、修改代码、分析项目时返回 true。"
                "普通聊天或纯写作通常返回 false。"
                "只返回 JSON："
                '{"needs_tool":true,"reason":"一句话原因","confidence":0.0-1.0}'
            ),
            max_tokens_env="BEGINNER_AGENT_ROUTER_TOOL_NEEDS_MAX_TOKENS",
            timeout_ms_env="BEGINNER_AGENT_ROUTER_TOOL_NEEDS_TIMEOUT_MS",
        )
        parsed, repair = parse_stage_model_with_repair(
            response,
            model_cls=ToolNeedsStageModel,
            required_fields={"needs_tool"},
            stage_title="Tool Needs Router",
            schema_hint='{"needs_tool":true,"reason":"...","confidence":0.0-1.0}',
            prompt=prompt,
            chat_completion=chat_completion,
            failure_policy=failure_policy,
            max_tokens_env="BEGINNER_AGENT_ROUTER_TOOL_NEEDS_MAX_TOKENS",
        )
        if parsed.confidence < min_confidence and failure_policy.low_confidence_policy == "fallback":
            return fallback_tool_needs(intent, f"置信度 {parsed.confidence:.2f} 低于阈值。", response)
        return RouterStageDecision(
            stage="tool_needs_router",
            decision=str(parsed.needs_tool).lower(),
            reason=parsed.reason,
            confidence=parsed.confidence,
            source="llm",
            model_response=repair.final_response or response,
            repair_attempt_count=repair.attempt_count,
            repair_success=repair.success,
            raw_invalid_response=repair.raw_invalid_response,
            validation_error_type=repair.validation_error_type,
        )
    except (
        RuntimeError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        AttributeError,
        ValidationError,
    ) as exc:
        return fallback_tool_needs(intent, f"{type(exc).__name__}: {exc}", "")


def _security_stage_decision(
    security: RouterSecuritySignal,
    *,
    failure_policy: RouterFailurePolicy,
) -> RouterStageDecision:
    policy_applied = ""
    if security.malicious_intent != "none":
        policy_applied = "security_override_raise_high"
    elif security.injection_risk != "none" and failure_policy.security_failure_policy == "raise_risk":
        policy_applied = "security_suspected_raise_risk"
    return RouterStageDecision(
        stage="security_router",
        decision=security.malicious_intent,
        reason=security.reason,
        confidence=0.9 if security.malicious_intent != "none" else 0.7,
        source="security_override" if security.malicious_intent != "none" else "llm",
        failure_policy_applied=policy_applied,
    )


def _bool_from_stage_decision(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


__all__ = [
    "MultiStageRouterResult",
    "RouterStageDecision",
    "RouterStageReport",
    "build_multistage_reports",
    "run_multistage_router",
]
