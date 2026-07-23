from __future__ import annotations

import os
from typing import Any

from .checkpoint_models import CheckpointBackendConfig, CheckpointFallbackRiskDecision
from .config import load_project_env
from .state import State
from .tooling.registry import TOOL_SPECS


load_project_env()


def checkpoint_long_task_step_threshold() -> int:
    """读取“长任务”步数阈值。

    中文注释：
    max_steps 超过这个阈值，就说明任务可能运行较久。
    长任务一旦进程退出，如果只用 memory checkpoint，就无法可靠恢复，
    所以默认要求持久化 checkpoint。
    """

    raw = os.getenv("BEGINNER_AGENT_CHECKPOINT_LONG_TASK_STEP_THRESHOLD", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def evaluate_checkpoint_fallback_policy(
    state: State,
    config: CheckpointBackendConfig,
) -> CheckpointFallbackRiskDecision:
    """根据当前 State 判断 memory fallback 是否可以接受。

    中文注释：
    checkpoint_backend_config() 负责读 env，决定“技术配置”。
    这个函数负责读 State，决定“业务风险”。

    两层要分开：

        env/config 层：Postgres 没配时是否允许 fallback。
        runtime policy 层：这一次任务风险高不高，能不能接受 fallback。

    如果当前任务是高风险代码修改、长任务、需要审批，
    即使 env 允许 fallback，也会被这里拦住。
    """

    threshold = checkpoint_long_task_step_threshold()
    fallback_active = (
        config.requested_backend == "postgres" and config.effective_backend == "memory"
    )
    risk_factors = _checkpoint_persistence_risk_factors(state, threshold)
    requires_persistence = bool(risk_factors)

    if not fallback_active:
        return CheckpointFallbackRiskDecision(
            allowed=True,
            requires_persistence=requires_persistence,
            reason="当前没有发生 Postgres -> memory fallback。",
            risk_factors=risk_factors,
            long_task_step_threshold=threshold,
        )

    if not config.allow_memory_fallback:
        return CheckpointFallbackRiskDecision(
            allowed=False,
            requires_persistence=True,
            reason="env 不允许 memory fallback。",
            risk_factors=[*risk_factors, "env_disallows_memory_fallback"],
            long_task_step_threshold=threshold,
        )

    if requires_persistence:
        return CheckpointFallbackRiskDecision(
            allowed=False,
            requires_persistence=True,
            reason="当前任务需要持久化 checkpoint，不能降级到 memory。",
            risk_factors=risk_factors,
            long_task_step_threshold=threshold,
        )

    return CheckpointFallbackRiskDecision(
        allowed=True,
        requires_persistence=False,
        reason="当前是低风险短任务，env 允许本地 memory fallback。",
        risk_factors=[],
        long_task_step_threshold=threshold,
    )


def _checkpoint_persistence_risk_factors(state: State, threshold: int) -> list[str]:
    """收集“为什么必须持久化”的风险因子。

    中文注释：
    生产级 policy 不建议只返回 True/False。
    它应该告诉你具体命中了哪些风险规则，方便审计和后续调参。
    """

    factors: list[str] = []
    risk_level = str(state.get("risk_level", "low"))
    task_type = str(state.get("task_type", "chat"))
    tool_name = str(state.get("tool_name", "none"))
    max_steps = _int_state(state.get("max_steps"), default=0)
    pending_approval = state.get("pending_approval") or {}
    policy_decision = str(state.get("policy_decision", ""))

    if risk_level == "high":
        factors.append("risk_level_high")
    if task_type == "agent" and _bool_env(
        "BEGINNER_AGENT_CHECKPOINT_REQUIRE_PERSISTENCE_FOR_AGENT", True
    ):
        factors.append("task_type_agent")
    if max_steps >= threshold:
        factors.append(f"long_task_max_steps_{max_steps}")
    if (
        pending_approval or policy_decision == "ask"
    ) and _bool_env("BEGINNER_AGENT_CHECKPOINT_REQUIRE_PERSISTENCE_FOR_APPROVAL", True):
        factors.append("approval_required")
    if _tool_requires_persistent_checkpoint(tool_name):
        factors.append(f"tool_requires_persistence_{tool_name}")

    return factors


def _tool_requires_persistent_checkpoint(tool_name: str) -> bool:
    """判断当前工具是否属于需要持久化 checkpoint 的工具。

    中文注释：
    这里优先读 ToolSpec，而不是靠工具名猜测。
    如果工具是 write，或者风险是 high，通常说明它可能修改项目状态，
    这种任务必须能恢复、能审计，不能只靠进程内存。
    """

    if not _bool_env("BEGINNER_AGENT_CHECKPOINT_REQUIRE_PERSISTENCE_FOR_WRITE_TOOLS", True):
        return False
    spec = TOOL_SPECS.get(tool_name)
    if spec is None:
        return False
    return spec.access == "write" or spec.risk == "high"


def _int_state(value: Any, *, default: int) -> int:
    """把 State 里的数字字段安全转成 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    """读取布尔 env。"""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
