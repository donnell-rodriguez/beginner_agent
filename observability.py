from __future__ import annotations

from typing import Any, Literal

from .observability_store import ObservabilityStore
from .run_lineage import build_run_lineage_report
from .state import State


ObservabilityRoute = Literal["schedule", "finish"]


def observability_reporter_node(state: State) -> dict[str, Any]:
    """Observability Reporter：生成当前 agent loop 的可观测性报告。

    中文注释：
    生产级 agent 需要知道自己运行得怎么样：
    - 做了几步。
    - 当前目标进度。
    - 最近工具是否成功。
    - 是否触发审批、恢复、checkpoint、sandbox。

    这个节点不改变业务决策，只把这些信息归档到 State。
    """

    task_tree = state.get("task_tree", {})
    statuses: dict[str, int] = {}
    for task in task_tree.values():
        if isinstance(task, dict):
            status = str(task.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1

    report: dict[str, Any] = {
        "step_count": state["step_count"],
        "run_id": state["run_id"],
        "done": state["done"],
        "next_action": state["next_action"],
        "task_status_counts": statuses,
        "goal_progress": state.get("goal_progress", {}),
        "policy": {
            "decision": state.get("policy_decision"),
            "reason": state.get("policy_reason"),
        },
        "execution": {
            "status": state.get("execution_status"),
            "monitor_status": state.get("execution_monitor_status"),
            "monitor_reason": state.get("execution_monitor_reason"),
        },
        "recovery": {
            "action": state.get("recovery_action"),
            "reason": state.get("recovery_reason"),
        },
        "checkpoint": state.get("checkpoint_report", {}),
        "sandbox": state.get("sandbox_report", {}),
        "async_job": state.get("async_job_report", {}),
        "artifacts": state.get("artifact_report", {}),
    }
    report["lineage"] = build_run_lineage_report(state)
    report["storage"] = ObservabilityStore().record_report(
        run_id=state["run_id"],
        report=report,
    )
    return {
        "observability_report": report,
        "run_lineage_report": report["lineage"],
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Observability Reporter："
                    f"step={state['step_count']}，done={state['done']}，"
                    f"next_action={state['next_action']}。"
                ),
            }
        ],
    }


def route_after_observability_reporter(state: State) -> ObservabilityRoute:
    """Observability Reporter 后的路由。"""

    if state["done"] or state["next_action"] == "finish":
        return "finish"
    return "schedule"
