from __future__ import annotations

from typing import Any, Literal

from .checkpoint_models import CheckpointRecoveryContract, CheckpointReport
from .checkpoint_observability import append_checkpoint_event
from .checkpoint_policy import evaluate_checkpoint_fallback_policy
from .checkpointing import check_checkpoint_health, checkpoint_backend_config
from .state import State


CheckpointRoute = Literal["schedule", "finish"]


def postgres_checkpoint_node(state: State) -> dict[str, Any]:
    """Postgres Checkpoint：检查并记录当前 LangGraph checkpoint 后端。

    中文注释：
    真正的 checkpoint 不是在这个节点里手动保存的。
    真正保存发生在 graph.py 的：

        builder.compile(checkpointer=build_checkpointer())

    这个节点现在负责工程化检查：

    - 当前请求的是 memory 还是 postgres。
    - 实际生效的是 memory 还是 postgres。
    - Postgres database_url 是否配置。
    - 当前是否有 thread_id / run_id 可用于恢复。
    - checkpoint setup 是否健康。
    - 恢复合约和 fallback 策略是什么。
    """

    config = checkpoint_backend_config()
    health = check_checkpoint_health()
    thread_id = str(state.get("thread_id", "") or state.get("run_id", ""))
    thread_id_present = bool(thread_id)
    fallback_risk_decision = evaluate_checkpoint_fallback_policy(state, config)
    if not fallback_risk_decision.allowed:
        health = health.model_copy(
            update={
                "status": "blocked",
                "errors": [
                    *health.errors,
                    fallback_risk_decision.reason,
                ],
            }
        )

    recovery_contract = CheckpointRecoveryContract(
        requires_thread_id=config.require_thread_id,
        thread_id_present=thread_id_present,
        thread_id_source="state.thread_id" if state.get("thread_id") else "state.run_id",
        checkpoint_namespace=config.checkpoint_namespace,
        backend_name=config.effective_backend,
        persistence_guarantee="durable" if config.effective_backend == "postgres" else "process",
        fallback_policy=config.fallback_policy,
        resume_supported=(
            config.effective_backend == "postgres"
            and thread_id_present
            and health.status in {"healthy", "warning"}
        ),
    )
    observability_event = {
        "event_type": "checkpoint_health",
        "run_id": state["run_id"],
        "backend": config.effective_backend,
        "requested_backend": config.requested_backend,
        "status": health.status,
        "persistent": health.persistent,
        "setup_status": health.setup_status,
        "setup_mode": config.setup_mode,
        "auto_setup_enabled": config.auto_setup_enabled,
        "diagnostics": health.diagnostics.model_dump(mode="json"),
        "resume_supported": recovery_contract.resume_supported,
        "fallback_risk_decision": fallback_risk_decision.model_dump(mode="json"),
        "warnings": health.warnings,
        "errors": health.errors,
    }
    report = CheckpointReport(
        run_id=state["run_id"],
        backend=config.effective_backend,
        requested_backend=config.requested_backend,
        persistent=health.persistent,
        health=health,
        recovery_contract=recovery_contract,
        fallback_risk_decision=fallback_risk_decision,
        observability_event=observability_event,
        reason=_checkpoint_reason(
            health.status,
            config.effective_backend,
            fallback_allowed=fallback_risk_decision.allowed,
        ),
    ).model_dump(mode="json")
    append_checkpoint_event(report["observability_event"])
    updates: dict[str, Any] = {
        "checkpoint_report": report,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Postgres Checkpoint："
                    f"backend={config.effective_backend}，"
                    f"status={health.status}，"
                    f"setup_mode={config.setup_mode}，"
                    f"resume_supported={recovery_contract.resume_supported}。"
                ),
            }
        ],
    }
    if health.status == "blocked":
        updates.update(
            {
                "next_action": "finish",
                "done": True,
                "final_answer": (
                    "Checkpoint 策略阻断：当前任务需要持久化 checkpoint，"
                    "但运行时无法提供可靠恢复能力。请配置 Postgres checkpoint 后重试。"
                ),
            }
        )
    return updates


def route_after_postgres_checkpoint(state: State) -> CheckpointRoute:
    """Checkpoint 后的路由。

    中文注释：
    以前 graph.py 固定从 postgres_checkpoint 进入 scheduler。
    现在如果 checkpoint policy 判定 blocked，就直接进入最终总结，
    避免高风险/长任务在没有持久化恢复能力时继续执行。
    """

    report = state.get("checkpoint_report", {})
    health = report.get("health", {}) if isinstance(report, dict) else {}
    if health.get("status") == "blocked":
        return "finish"
    return "schedule"


def _checkpoint_reason(status: str, backend: str, *, fallback_allowed: bool) -> str:
    """生成给 summary 阅读的 checkpoint 状态说明。"""

    if not fallback_allowed:
        return "Checkpoint fallback 被风险策略阻断，当前任务不能继续。"
    if status == "healthy" and backend == "postgres":
        return "Postgres checkpoint 健康，可支持长任务恢复。"
    if status == "warning" and backend == "postgres":
        return "Postgres checkpoint 可用但存在 warning，建议查看 checkpoint_report.health。"
    if status == "degraded":
        return "Checkpoint 已降级运行，当前恢复能力弱于请求配置。"
    if status == "blocked":
        return "Checkpoint 配置或健康检查阻塞，需要修复后再运行长任务。"
    return "Memory checkpoint 只适合本地单进程实验。"
