from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from ..config import load_project_env


# 中文注释：
# model_strategy.py 负责 Router 多模型策略。
#
# 它解决的问题是：
# - 每个 Router stage 可以有自己的模型。
# - 可以配置 cheap model 先判断。
# - 低置信度或高风险时升级到 strong model。
# - 模型选择结果进入治理合同和 stage report，方便审计。

RouterModelTier = Literal["primary", "cheap", "strong"]


@dataclass(frozen=True)
class RouterModelSelection:
    stage_key: str
    tier: RouterModelTier
    model: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "stage_key": self.stage_key,
            "tier": self.tier,
            "model": self.model,
            "reason": self.reason,
        }


def router_model_escalation_enabled() -> bool:
    """是否启用 cheap -> strong 的升级策略。"""

    load_project_env()
    return os.getenv(
        "BEGINNER_AGENT_ROUTER_MODEL_ESCALATION_ENABLED",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}


def router_primary_model_tier() -> RouterModelTier:
    """读取 Router 首次调用使用 primary 还是 cheap。"""

    load_project_env()
    raw = os.getenv("BEGINNER_AGENT_ROUTER_PRIMARY_MODEL_TIER", "primary").strip().lower()
    return "cheap" if raw == "cheap" else "primary"


def router_strong_confidence_threshold() -> float:
    """低于这个置信度时升级 strong model。"""

    load_project_env()
    try:
        value = float(os.getenv("BEGINNER_AGENT_ROUTER_STRONG_CONFIDENCE_THRESHOLD", "0.70"))
    except ValueError:
        return 0.70
    return min(max(value, 0.0), 1.0)


def router_high_risk_strong_validation_enabled() -> bool:
    """risk=high 时是否强制 strong model 复核。"""

    load_project_env()
    return os.getenv(
        "BEGINNER_AGENT_ROUTER_HIGH_RISK_STRONG_VALIDATION_ENABLED",
        "true",
    ).strip().lower() in {"1", "true", "yes", "on"}


def select_router_stage_model(
    stage_title: str,
    *,
    tier: RouterModelTier = "primary",
) -> RouterModelSelection:
    """选择某个 Router stage 使用的模型。"""

    load_project_env()
    stage_key = _stage_key(stage_title)
    if tier == "strong":
        model = _first_env(
            f"BEGINNER_AGENT_ROUTER_{stage_key}_STRONG_MODEL",
            "BEGINNER_AGENT_ROUTER_STRONG_MODEL",
            _stage_model_env(stage_key),
            "BEGINNER_AGENT_ROUTER_MODEL",
        )
        return RouterModelSelection(stage_key, tier, model, "strong model escalation")
    if tier == "cheap":
        model = _first_env(
            f"BEGINNER_AGENT_ROUTER_{stage_key}_CHEAP_MODEL",
            "BEGINNER_AGENT_ROUTER_CHEAP_MODEL",
            _stage_model_env(stage_key),
            "BEGINNER_AGENT_ROUTER_MODEL",
        )
        return RouterModelSelection(stage_key, tier, model, "cheap model first pass")
    model = _first_env(_stage_model_env(stage_key), "BEGINNER_AGENT_ROUTER_MODEL")
    return RouterModelSelection(stage_key, tier, model, "stage primary model")


def should_retry_with_strong_model(
    *,
    stage_title: str,
    confidence: float,
    primary_model: str,
    risk_level: str = "",
) -> tuple[bool, str, RouterModelSelection]:
    """判断当前 stage 是否应该升级 strong model。"""

    strong = select_router_stage_model(stage_title, tier="strong")
    if not router_model_escalation_enabled():
        return False, "model escalation disabled", strong
    if not strong.model:
        return False, "strong model not configured", strong
    if primary_model and strong.model == primary_model:
        return False, "strong model equals primary model", strong
    if risk_level == "high" and router_high_risk_strong_validation_enabled():
        return True, "risk_level=high requires strong validation", strong
    threshold = router_strong_confidence_threshold()
    if confidence < threshold:
        return True, f"confidence {confidence:.2f} < strong threshold {threshold:.2f}", strong
    return False, "strong escalation not required", strong


def _stage_key(stage_title: str) -> str:
    key = stage_title.upper().replace(" ", "_").replace("-", "_")
    return key.replace("_ROUTER", "").replace("JSON_REPAIR", "REPAIR")


def _stage_model_env(stage_key: str) -> str:
    env_by_key = {
        "INTENT": "BEGINNER_AGENT_ROUTER_INTENT_MODEL",
        "RISK": "BEGINNER_AGENT_ROUTER_RISK_MODEL",
        "TOOL_NEEDS": "BEGINNER_AGENT_ROUTER_TOOL_NEEDS_MODEL",
        "SECURITY_CLASSIFIER": "BEGINNER_AGENT_ROUTER_SECURITY_CLASSIFIER_MODEL",
        "INTENT_REPAIR": "BEGINNER_AGENT_ROUTER_REPAIR_MODEL",
        "RISK_REPAIR": "BEGINNER_AGENT_ROUTER_REPAIR_MODEL",
        "TOOL_NEEDS_REPAIR": "BEGINNER_AGENT_ROUTER_REPAIR_MODEL",
    }
    return env_by_key.get(stage_key, "BEGINNER_AGENT_ROUTER_MODEL")


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""
