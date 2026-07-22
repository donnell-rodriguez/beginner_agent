from __future__ import annotations

import os

from ..config import load_project_env
from .models import RouterContext, RouterDecision


DEFAULT_TENANT_ID = "local-tenant"
DEFAULT_WORKSPACE_ID = "local-workspace"
DEFAULT_PROJECT_ID = "beginner_agent"
DEFAULT_USER_ID = "local-user"


def load_router_context() -> RouterContext:
    """读取 Router 决策上下文。

    中文注释：
    这些字段来自 env，而不是写死在业务代码里。
    这样后续做多租户、多项目、多用户时，可以按上下文区分策略。
    """

    load_project_env()
    return RouterContext(
        tenant_id=os.getenv("BEGINNER_AGENT_TENANT_ID", DEFAULT_TENANT_ID).strip()
        or DEFAULT_TENANT_ID,
        workspace_id=os.getenv("BEGINNER_AGENT_WORKSPACE_ID", DEFAULT_WORKSPACE_ID).strip()
        or DEFAULT_WORKSPACE_ID,
        project_id=os.getenv("BEGINNER_AGENT_PROJECT_ID", DEFAULT_PROJECT_ID).strip()
        or DEFAULT_PROJECT_ID,
        user_id=os.getenv("BEGINNER_AGENT_USER_ID", DEFAULT_USER_ID).strip() or DEFAULT_USER_ID,
    )


def _csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def apply_context_policy(
    decision: RouterDecision,
    context: RouterContext,
) -> tuple[RouterDecision, str]:
    """根据 tenant/project/user 策略提升 Router 风险。

    中文注释：
    当前先支持“强制高风险”的上下文策略：
    - BEGINNER_AGENT_ROUTER_HIGH_RISK_TENANTS
    - BEGINNER_AGENT_ROUTER_HIGH_RISK_PROJECTS
    - BEGINNER_AGENT_ROUTER_HIGH_RISK_USERS

    这些值用逗号分隔。命中后 Router 不会拒绝任务，
    但会把 risk_level 提升为 high，让后续 policy/approval 接管。
    """

    high_risk_tenants = _csv_env("BEGINNER_AGENT_ROUTER_HIGH_RISK_TENANTS")
    high_risk_projects = _csv_env("BEGINNER_AGENT_ROUTER_HIGH_RISK_PROJECTS")
    high_risk_users = _csv_env("BEGINNER_AGENT_ROUTER_HIGH_RISK_USERS")
    reasons: list[str] = []
    if context.tenant_id in high_risk_tenants:
        reasons.append(f"tenant={context.tenant_id} 被配置为高风险。")
    if context.project_id in high_risk_projects:
        reasons.append(f"project={context.project_id} 被配置为高风险。")
    if context.user_id in high_risk_users:
        reasons.append(f"user={context.user_id} 被配置为高风险。")

    if not reasons:
        return decision, ""

    updated = decision.model_copy(
        update={
            "task_type": "agent",
            "risk_level": "high",
            "needs_tool": True,
            "reason": f"{decision.reason} Router 上下文策略：{'；'.join(reasons)}",
            "confidence": min(decision.confidence, 0.65),
        }
    )
    return updated, "；".join(reasons)
