from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from ..config import load_project_env
from ..state import RiskLevel
from .rules import DEFAULT_HIGH_RISK_KEYWORDS, DEFAULT_MEDIUM_RISK_KEYWORDS, RouterRuleSet


# 中文注释：
# failure_policy.py 负责 Router 失败时的“治理策略”。
#
# 生产级 Router 不应该把失败处理散落在每个 if/except 里。
# 更合理的是把策略集中起来：
# - 是否允许 repair retry。
# - repair 最多尝试几次。
# - risk 阶段失败时是否保守升级。
# - security 阶段失败时是否保守处理。
#
# 这样以后要修改策略，优先改配置和这个模块，而不是改 graph/node 主流程。


RiskFailurePolicy = Literal["rule", "conservative"]
SecurityFailurePolicy = Literal["none", "raise_risk"]
LowConfidencePolicy = Literal["fallback", "accept"]


@dataclass(frozen=True)
class RouterFailurePolicy:
    """Router 失败处理策略。

    中文注释：
    它不是 Router 的业务判断结果，而是“模型输出不可靠时怎么办”的治理合同。
    """

    repair_retry_enabled: bool = True
    max_repair_attempts: int = 1
    risk_failure_policy: RiskFailurePolicy = "conservative"
    security_failure_policy: SecurityFailurePolicy = "raise_risk"
    low_confidence_policy: LowConfidencePolicy = "fallback"


def load_router_failure_policy() -> RouterFailurePolicy:
    """从 .env 读取 Router 失败处理策略。"""

    load_project_env()
    return RouterFailurePolicy(
        repair_retry_enabled=_env_bool(
            "BEGINNER_AGENT_ROUTER_REPAIR_RETRY_ENABLED",
            True,
        ),
        max_repair_attempts=_env_int(
            "BEGINNER_AGENT_ROUTER_MAX_REPAIR_ATTEMPTS",
            1,
            min_value=0,
            max_value=3,
        ),
        risk_failure_policy=_env_choice(
            "BEGINNER_AGENT_ROUTER_RISK_FAILURE_POLICY",
            "conservative",
            {"rule", "conservative"},
        ),
        security_failure_policy=_env_choice(
            "BEGINNER_AGENT_ROUTER_SECURITY_FAILURE_POLICY",
            "raise_risk",
            {"none", "raise_risk"},
        ),
        low_confidence_policy=_env_choice(
            "BEGINNER_AGENT_ROUTER_LOW_CONFIDENCE_POLICY",
            "fallback",
            {"fallback", "accept"},
        ),
    )


def conservative_risk_level(text: str, rules: RouterRuleSet) -> tuple[RiskLevel, str]:
    """risk_router 失败时的保守风险判断。

    中文注释：
    规则系统仍然是第一层依据。
    但如果模型风险阶段失败，生产系统不能轻易降成 low。
    所以这里会额外检查高风险/中风险关键词。
    """

    rule_decision = rules.explain_risk_level(text)
    if rule_decision.outcome == "high":
        return "high", f"规则已命中 high：{rule_decision.selected_rule_reason}"
    if _contains_any(text, DEFAULT_HIGH_RISK_KEYWORDS):
        return "high", "risk_router 失败，输入包含代码修改/命令/写入类高风险关键词。"
    if rule_decision.outcome == "medium":
        return "medium", f"规则已命中 medium：{rule_decision.selected_rule_reason}"
    if _contains_any(text, DEFAULT_MEDIUM_RISK_KEYWORDS):
        return "medium", "risk_router 失败，输入包含测试/构建/编译类中风险关键词。"
    if _looks_like_tool_request(text):
        return "medium", "risk_router 失败，输入看起来需要工具访问，保守提升到 medium。"
    return "low", f"未命中风险关键词，沿用规则结果：{rule_decision.selected_rule_reason}"


def _looks_like_tool_request(text: str) -> bool:
    return _contains_any(
        text,
        (
            "文件",
            "目录",
            "源码",
            "读取",
            "列出",
            "运行",
            "测试",
            "build",
            "lint",
            "grep",
            "search",
        ),
    )


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(max(value, min_value), max_value)


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in allowed else default
