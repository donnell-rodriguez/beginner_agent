from __future__ import annotations

from typing import Any

from .checkpoint_models import CheckpointRecoveryContract, CheckpointReport
from .checkpoint_observability import append_checkpoint_event
from .checkpointing import check_checkpoint_health, checkpoint_backend_config
from .state import State


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
        "resume_supported": recovery_contract.resume_supported,
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
        observability_event=observability_event,
        reason=_checkpoint_reason(health.status, config.effective_backend),
    ).model_dump(mode="json")
    append_checkpoint_event(report["observability_event"])
    return {
        "checkpoint_report": report,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Postgres Checkpoint："
                    f"backend={config.effective_backend}，"
                    f"status={health.status}，"
                    f"resume_supported={recovery_contract.resume_supported}。"
                ),
            }
        ],
    }


def _checkpoint_reason(status: str, backend: str) -> str:
    """生成给 summary 阅读的 checkpoint 状态说明。"""

    if status == "healthy" and backend == "postgres":
        return "Postgres checkpoint 健康，可支持长任务恢复。"
    if status == "warning" and backend == "postgres":
        return "Postgres checkpoint 可用但存在 warning，建议查看 checkpoint_report.health。"
    if status == "degraded":
        return "Checkpoint 已降级运行，当前恢复能力弱于请求配置。"
    if status == "blocked":
        return "Checkpoint 配置或健康检查阻塞，需要修复后再运行长任务。"
    return "Memory checkpoint 只适合本地单进程实验。"
