from __future__ import annotations

from typing import Any

from .async_job_store import AsyncJobStore
from .state import State


def async_job_waiter_node(state: State) -> dict[str, Any]:
    """Async Job Waiter：等待或记录异步任务状态。

    中文注释：
    当前工具还是同步执行，所以大多数情况下不需要真正等待。

    但生产级 agent 常见流程是：

        Executor 提交 job
          -> Async Job Waiter 轮询 job 状态
          -> Execution Monitor 判断成功/失败/超时

    这个节点先把 async job contract 显式接进图里。
    后续如果 Executor 改成远程 worker，只需要扩展这里的等待逻辑，
    不需要大改 graph.py。
    """

    active_execution = dict(state.get("active_execution", {}))
    worker_contract = dict(active_execution.get("future_worker_contract", {}))
    job_id = str(worker_contract.get("job_id", ""))
    execution_status = str(active_execution.get("execution_status", state["execution_status"]))

    if job_id and execution_status == "waiting_external":
        job = AsyncJobStore().wait_for_job(job_id)
        status = str(job["status"])
        reason = (
            f"远程 job {job_id} 状态：{status}。"
            if status in {"success", "failed", "cancelled", "timeout"}
            else f"等待远程 job {job_id} 完成。"
        )
    else:
        job = {}
        status = "not_required"
        reason = "当前工具同步完成，不需要等待异步 job。"

    update: dict[str, Any] = {
        "async_job_report": {
            "status": status,
            "reason": reason,
            "job_id": job_id,
            "execution_status": execution_status,
            "worker_contract": worker_contract,
            "job": job,
        },
        "next_action": "monitor",
        "messages": [
            {
                "role": "assistant",
                "content": f"Async Job Waiter：{reason}",
            }
        ],
    }
    if status == "success":
        result = dict(job.get("result") or {})
        update.update(
            {
                "execution_status": "completed",
                "tool_result_status": "success",
                "tool_result": str(result.get("output", "远程 job 执行成功。")),
                "tool_result_data": result,
            }
        )
    elif status in {"failed", "cancelled", "timeout"}:
        update.update(
            {
                "execution_status": "failed",
                "tool_result_status": "failed",
                "tool_result": str(job.get("error") or reason),
                "tool_result_data": dict(job.get("result") or {}),
            }
        )
    return update
