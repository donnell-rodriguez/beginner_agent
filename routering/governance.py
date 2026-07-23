from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..config import load_project_env
from .models import RouterContext
from .model_strategy import select_router_stage_model
from .prompts import RouterPromptSpec
from .rules import RouterRuleSet
from .security_models import SecurityPolicy


# 中文注释：
# governance.py 负责 Router 的“版本合同”和“阶段预算合同”。
#
# 大厂级 Router 不能只知道结果，还要知道：
# - 用了哪一版 Router。
# - 用了哪一版 prompt / rules / security policy。
# - 每个子阶段理论上允许用多少时间、多少 token、哪个模型。
#
# 当前先用 .env + 本地对象实现，后续可以替换成配置中心。


@dataclass(frozen=True)
class RouterStageBudget:
    """单个 Router 子阶段的预算配置。"""

    stage: str
    model: str
    cheap_model: str
    strong_model: str
    max_tokens: int
    timeout_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "max_tokens": self.max_tokens,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True)
class RouterGovernanceContract:
    """一次 Router 决策使用的治理版本合同。"""

    router_version: str
    prompt_version: str
    prompt_source: str
    prompt_experiment_group: str
    ruleset_version: str
    ruleset_source: str
    security_policy_version: str
    security_policy_source: str
    tenant_id: str
    workspace_id: str
    project_id: str
    user_id: str
    stage_budgets: tuple[RouterStageBudget, ...]
    max_total_latency_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "router_version": self.router_version,
            "prompt_version": self.prompt_version,
            "prompt_source": self.prompt_source,
            "prompt_experiment_group": self.prompt_experiment_group,
            "ruleset_version": self.ruleset_version,
            "ruleset_source": self.ruleset_source,
            "security_policy_version": self.security_policy_version,
            "security_policy_source": self.security_policy_source,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "stage_budgets": [budget.as_dict() for budget in self.stage_budgets],
            "max_total_latency_ms": self.max_total_latency_ms,
        }


def load_router_governance_contract(
    *,
    prompt: RouterPromptSpec,
    rules: RouterRuleSet,
    security_policy: SecurityPolicy,
    context: RouterContext,
) -> RouterGovernanceContract:
    """组装本次 Router 决策的治理合同。"""

    load_project_env()
    return RouterGovernanceContract(
        router_version=os.getenv("BEGINNER_AGENT_ROUTER_VERSION", "router-local").strip()
        or "router-local",
        prompt_version=prompt.version,
        prompt_source=prompt.source,
        prompt_experiment_group=prompt.experiment_group,
        ruleset_version=rules.version,
        ruleset_source=rules.source,
        security_policy_version=security_policy.version,
        security_policy_source=security_policy.source,
        tenant_id=context.tenant_id,
        workspace_id=context.workspace_id,
        project_id=context.project_id,
        user_id=context.user_id,
        stage_budgets=(
            _stage_budget(
                "intent_router",
                model_env="BEGINNER_AGENT_ROUTER_INTENT_MODEL",
                max_tokens_env="BEGINNER_AGENT_ROUTER_INTENT_MAX_TOKENS",
                timeout_env="BEGINNER_AGENT_ROUTER_INTENT_TIMEOUT_MS",
                default_max_tokens=80,
            ),
            _stage_budget(
                "risk_router",
                model_env="BEGINNER_AGENT_ROUTER_RISK_MODEL",
                max_tokens_env="BEGINNER_AGENT_ROUTER_RISK_MAX_TOKENS",
                timeout_env="BEGINNER_AGENT_ROUTER_RISK_TIMEOUT_MS",
                default_max_tokens=80,
            ),
            _stage_budget(
                "tool_needs_router",
                model_env="BEGINNER_AGENT_ROUTER_TOOL_NEEDS_MODEL",
                max_tokens_env="BEGINNER_AGENT_ROUTER_TOOL_NEEDS_MAX_TOKENS",
                timeout_env="BEGINNER_AGENT_ROUTER_TOOL_NEEDS_TIMEOUT_MS",
                default_max_tokens=80,
            ),
            _stage_budget(
                "security_classifier",
                model_env="BEGINNER_AGENT_ROUTER_SECURITY_CLASSIFIER_MODEL",
                max_tokens_env="BEGINNER_AGENT_ROUTER_SECURITY_CLASSIFIER_MAX_TOKENS",
                timeout_env="BEGINNER_AGENT_ROUTER_SECURITY_CLASSIFIER_TIMEOUT_MS",
                default_max_tokens=120,
            ),
            _stage_budget(
                "repair_router",
                model_env="BEGINNER_AGENT_ROUTER_REPAIR_MODEL",
                max_tokens_env="BEGINNER_AGENT_ROUTER_REPAIR_MAX_TOKENS",
                timeout_env="BEGINNER_AGENT_ROUTER_REPAIR_TIMEOUT_MS",
                default_max_tokens=120,
            ),
        ),
        max_total_latency_ms=_env_int("BEGINNER_AGENT_ROUTER_MAX_TOTAL_LATENCY_MS", 3000),
    )


def router_stage_model(stage_title: str) -> str:
    """读取某个 Router 子阶段应该使用的模型。"""

    return select_router_stage_model(stage_title).model


def _stage_budget(
    stage: str,
    *,
    model_env: str,
    max_tokens_env: str,
    timeout_env: str,
    default_max_tokens: int,
) -> RouterStageBudget:
    primary = select_router_stage_model(stage)
    cheap = select_router_stage_model(stage, tier="cheap")
    strong = select_router_stage_model(stage, tier="strong")
    return RouterStageBudget(
        stage=stage,
        model=primary.model or os.getenv(model_env, os.getenv("BEGINNER_AGENT_ROUTER_MODEL", "")).strip(),
        cheap_model=cheap.model,
        strong_model=strong.model,
        max_tokens=_env_int(max_tokens_env, default_max_tokens),
        timeout_ms=_env_int(timeout_env, 1000),
    )


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
