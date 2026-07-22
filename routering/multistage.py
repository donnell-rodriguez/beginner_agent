from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import load_project_env
from ..node_utils import json_loads_from_model
from ..state import RiskLevel, TaskType
from .models import DecisionSource, RouterDecision, RouterSecuritySignal, RouterStageReport
from .prompts import RouterPromptSpec
from .rules import RouterRuleSet


# 中文注释：
# multistage.py 是真正的“多阶段 Router”。
#
# 之前 router.py 是一次 LLM 调用直接返回：
#
#     task_type + risk_level + needs_tool
#
# 现在拆成多个独立子路由器：
#
#     Intent Router      -> 判断用户想做什么
#     Risk Router        -> 判断风险等级
#     Tool Needs Router  -> 判断是否需要工具
#     Security Router    -> 使用本地安全分类器做保守覆盖
#
# 这样更接近生产系统：
# - 每个阶段可以单独换模型。
# - 每个阶段可以单独 eval。
# - 每个阶段都有自己的 fallback。
# - 审计时能看到到底是哪一层出的问题。


StageName = Literal["intent_router", "risk_router", "tool_needs_router", "security_router"]

ROUTER_DECISION_FIELDS = frozenset(
    {
        "task_type",
        "risk_level",
        "needs_tool",
        "reason",
        "confidence",
    }
)


class _IntentStageModel(BaseModel):
    """Intent Router 的模型输出 schema。"""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    reason: str = Field(default="Intent Router 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class _RiskStageModel(BaseModel):
    """Risk Router 的模型输出 schema。"""

    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel
    reason: str = Field(default="Risk Router 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class _ToolNeedsStageModel(BaseModel):
    """Tool Needs Router 的模型输出 schema。"""

    model_config = ConfigDict(extra="forbid")

    needs_tool: bool
    reason: str = Field(default="Tool Needs Router 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


@dataclass(frozen=True)
class RouterStageDecision:
    """一个独立 Router 子阶段的判断结果。"""

    stage: StageName
    decision: str
    reason: str
    confidence: float
    source: DecisionSource
    model_response: str = ""
    model_error: str = ""
    fallback_reason: str = ""


@dataclass(frozen=True)
class MultiStageRouterResult:
    """多阶段 Router 聚合后的结果。"""

    decision: RouterDecision
    stage_decisions: tuple[RouterStageDecision, ...]
    source: DecisionSource
    model_response: str = ""
    model_error: str = ""
    fallback_reason: str = ""


def run_multistage_router(
    text: str,
    *,
    rules: RouterRuleSet,
    security: RouterSecuritySignal,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
) -> MultiStageRouterResult:
    """运行多个独立 Router 子阶段，并聚合成最终 RouterDecision。

    中文注释：
    这里不是“一次 prompt 让模型全都判断完”，而是：

    1. Intent Router 单独判断 task_type。
    2. Risk Router 单独判断 risk_level。
    3. Tool Needs Router 单独判断 needs_tool。
    4. Security Router 用本地安全分类结果做保守覆盖。

    每个 LLM 子阶段都有自己的解析和 fallback。
    所以即使 Risk Router 失败，Intent Router 的结果仍然可以保留。
    """

    intent = _run_intent_router(
        text,
        rules=rules,
        prompt=prompt,
        chat_completion=chat_completion,
        min_confidence=min_confidence,
    )
    risk = _run_risk_router(
        text,
        rules=rules,
        prompt=prompt,
        chat_completion=chat_completion,
        min_confidence=min_confidence,
    )
    tool_needs = _run_tool_needs_router(
        text,
        intent=cast(TaskType, intent.decision),
        prompt=prompt,
        chat_completion=chat_completion,
        min_confidence=min_confidence,
    )
    security_decision = _security_stage_decision(security)

    stage_decisions = (intent, risk, tool_needs, security_decision)
    source: DecisionSource = "llm"
    if any(stage.source == "fallback" for stage in stage_decisions):
        source = "fallback"

    confidence = min(stage.confidence for stage in stage_decisions)
    reason = "；".join(
        f"{stage.stage}={stage.decision}（{stage.reason}）" for stage in stage_decisions
    )
    decision = RouterDecision(
        task_type=cast(TaskType, intent.decision),
        risk_level=cast(RiskLevel, risk.decision),
        needs_tool=_bool_from_stage_decision(tool_needs.decision),
        reason=reason,
        confidence=confidence,
    )

    if security.malicious_intent != "none":
        # 中文注释：
        # 安全阶段只允许“保守升级”，不允许把高风险降级。
        # 例如模型判断 chat/low，但本地安全分类命中读取 secret，
        # 最终仍然必须进入 agent/high/needs_tool。
        decision = decision.model_copy(
            update={
                "task_type": "agent",
                "risk_level": "high",
                "needs_tool": True,
                "reason": f"{decision.reason}；Security Router 保守升级：{security.reason}",
                "confidence": min(decision.confidence, 0.65),
            }
        )
        source = "security_override"

    return MultiStageRouterResult(
        decision=decision,
        stage_decisions=stage_decisions,
        source=source,
        model_response=_combine_model_responses(stage_decisions),
        model_error=_combine_stage_field(stage_decisions, "model_error"),
        fallback_reason=_combine_stage_field(stage_decisions, "fallback_reason"),
    )


def build_multistage_reports(result: MultiStageRouterResult) -> list[RouterStageReport]:
    """把独立子阶段结果转成 RouterEvent 使用的 stage_reports。"""

    reports: list[RouterStageReport] = []
    for stage in result.stage_decisions:
        reason = stage.reason
        if stage.source == "fallback":
            reason = f"{reason}；fallback_reason={stage.fallback_reason}"
        if stage.model_error:
            reason = f"{reason}；model_error={stage.model_error}"
        reports.append(
            RouterStageReport(
                stage=stage.stage,
                decision=stage.decision,
                reason=f"source={stage.source}；{reason}",
                confidence=stage.confidence,
            )
        )
    return reports


def _run_intent_router(
    text: str,
    *,
    rules: RouterRuleSet,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
) -> RouterStageDecision:
    try:
        response = _call_stage_router(
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
        )
        parsed = _IntentStageModel.model_validate(_stage_payload(response, {"task_type"}))
        if parsed.confidence < min_confidence:
            return _fallback_intent(text, rules, f"置信度 {parsed.confidence:.2f} 低于阈值。", response)
        return RouterStageDecision(
            stage="intent_router",
            decision=parsed.task_type,
            reason=parsed.reason,
            confidence=parsed.confidence,
            source="llm",
            model_response=response,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError, AttributeError, ValidationError) as exc:
        return _fallback_intent(text, rules, f"{type(exc).__name__}: {exc}", "")


def _run_risk_router(
    text: str,
    *,
    rules: RouterRuleSet,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
) -> RouterStageDecision:
    try:
        response = _call_stage_router(
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
        )
        parsed = _RiskStageModel.model_validate(_stage_payload(response, {"risk_level"}))
        if parsed.confidence < min_confidence:
            return _fallback_risk(text, rules, f"置信度 {parsed.confidence:.2f} 低于阈值。", response)
        return RouterStageDecision(
            stage="risk_router",
            decision=parsed.risk_level,
            reason=parsed.reason,
            confidence=parsed.confidence,
            source="llm",
            model_response=response,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError, AttributeError, ValidationError) as exc:
        return _fallback_risk(text, rules, f"{type(exc).__name__}: {exc}", "")


def _run_tool_needs_router(
    text: str,
    *,
    intent: TaskType,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    min_confidence: float,
) -> RouterStageDecision:
    try:
        response = _call_stage_router(
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
        )
        parsed = _ToolNeedsStageModel.model_validate(_stage_payload(response, {"needs_tool"}))
        if parsed.confidence < min_confidence:
            return _fallback_tool_needs(intent, f"置信度 {parsed.confidence:.2f} 低于阈值。", response)
        return RouterStageDecision(
            stage="tool_needs_router",
            decision=str(parsed.needs_tool).lower(),
            reason=parsed.reason,
            confidence=parsed.confidence,
            source="llm",
            model_response=response,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError, AttributeError, ValidationError) as exc:
        return _fallback_tool_needs(intent, f"{type(exc).__name__}: {exc}", "")


def _security_stage_decision(security: RouterSecuritySignal) -> RouterStageDecision:
    return RouterStageDecision(
        stage="security_router",
        decision=security.malicious_intent,
        reason=security.reason,
        confidence=0.9 if security.malicious_intent != "none" else 0.7,
        source="security_override" if security.malicious_intent != "none" else "llm",
    )


def _call_stage_router(
    text: str,
    *,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    stage_title: str,
    instruction: str,
    max_tokens_env: str,
) -> str:
    # 中文注释：
    # 这里复用 prompt registry 的基础模板，
    # 再追加当前子 Router 的专门职责。
    #
    # 后续如果要做“每个阶段使用不同模型”，
    # 可以把这个函数扩展成按 stage_title 读取不同 model/env。
    return chat_completion(
        [
            {
                "role": "system",
                "content": f"{prompt.template}\n\n{stage_title}：{instruction}",
            },
            {"role": "user", "content": text},
        ],
        temperature=prompt.temperature,
        max_tokens=_stage_max_tokens(max_tokens_env, prompt.max_tokens),
    )


def _stage_payload(response: str, required_fields: set[str]) -> dict[str, Any]:
    data = json_loads_from_model(response)
    if not isinstance(data, dict):
        raise ValueError("Router 子阶段输出不是 JSON object。")
    extra = set(data) - ROUTER_DECISION_FIELDS
    if extra:
        raise ValueError(f"Router 子阶段输出包含未治理字段：{sorted(extra)}")
    missing = required_fields - set(data)
    if missing:
        raise ValueError(f"Router 子阶段缺少字段：{sorted(missing)}")
    return {
        key: value
        for key, value in data.items()
        if key in required_fields or key in {"reason", "confidence"}
    }


def _fallback_intent(
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


def _fallback_risk(
    text: str,
    rules: RouterRuleSet,
    fallback_reason: str,
    model_response: str,
) -> RouterStageDecision:
    rule_decision = rules.explain_risk_level(text)
    return RouterStageDecision(
        stage="risk_router",
        decision=rule_decision.outcome,
        reason=f"Risk Router 兜底规则：{rule_decision.selected_rule_reason}",
        confidence=0.45,
        source="fallback",
        model_response=model_response,
        model_error=fallback_reason,
        fallback_reason=fallback_reason,
    )


def _fallback_tool_needs(
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


def _bool_from_stage_decision(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def _stage_max_tokens(name: str, default: int) -> int:
    load_project_env()
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _combine_model_responses(stages: tuple[RouterStageDecision, ...]) -> str:
    payload = {
        stage.stage: stage.model_response
        for stage in stages
        if stage.model_response
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _combine_stage_field(stages: tuple[RouterStageDecision, ...], field_name: str) -> str:
    values = [
        f"{stage.stage}: {getattr(stage, field_name)}"
        for stage in stages
        if getattr(stage, field_name)
    ]
    return "；".join(values)
