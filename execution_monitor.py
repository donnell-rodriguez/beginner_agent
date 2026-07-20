from __future__ import annotations

from typing import Any, Literal

from .state import State


ExecutionMonitorRoute = Literal["evaluate", "recover"]


def _monitor_status(state: State) -> tuple[str, str]:
    """根据最近一次 execution attempt 判断是否需要恢复。

    中文注释：
    Execution Monitor / Watchdog 不负责“怎么恢复”。
    它只回答一个问题：

        这次执行过程是否值得直接进入 Evaluator？

    如果只是正常完成，就进入 Evaluator。
    如果超预算、失败、空结果、部分结果，就交给 Recovery Planner。
    """

    active_execution = state.get("active_execution") or {}
    execution_status = str(active_execution.get("execution_status") or state["execution_status"])
    tool_result_status = str(
        active_execution.get("tool_result_status") or state["tool_result_status"]
    )
    duration_ms = int(active_execution.get("duration_ms") or 0)
    budget_ms = int(active_execution.get("budget_ms") or state["max_tool_duration_ms"])

    if execution_status == "blocked" or tool_result_status == "blocked":
        return "blocked", "工具被权限层、参数校验或安全策略阻断。"
    if execution_status == "failed" or tool_result_status == "failed":
        return "failed", "工具执行失败，需要 Recovery Planner 判断重试、换方案或停止。"
    if tool_result_status == "empty":
        return "empty", "工具返回空结果，需要判断是否重试或换方案。"
    if tool_result_status == "partial":
        return "partial", "工具只返回部分结果，需要判断是否继续补充或先接受部分结果。"
    if execution_status == "completed_over_budget" or (
        budget_ms > 0 and duration_ms > budget_ms
    ):
        return (
            "over_budget",
            f"工具耗时 {duration_ms}ms，超过预算 {budget_ms}ms，需要判断是否换更小范围策略。",
        )
    return "ok", "执行过程正常，可以进入 Evaluator。"


def execution_monitor_node(state: State) -> dict[str, Any]:
    """Execution Monitor / Watchdog：监控执行尝试。

    中文注释：
    大厂式长任务不会只做：

        Executor -> Evaluator

    因为执行可能超时、卡住、空结果、部分结果。
    所以中间加一层 Watchdog：

        Executor -> Execution Monitor -> Evaluator / Recovery Planner

    当前项目还没有真正后台 worker。
    因此这里主要基于 execution_attempt 记录做同步判断。
    后续如果接入异步 worker，这里可以扩展成：
    - 查询 job_id 状态。
    - 判断是否需要继续等待。
    - 判断是否取消任务。
    - 判断是否恢复执行。
    """

    status, reason = _monitor_status(state)
    next_action = "evaluate" if status == "ok" else "recover"
    return {
        "execution_monitor_status": status,
        "execution_monitor_reason": reason,
        "next_action": next_action,
        "messages": [
            {
                "role": "assistant",
                "content": f"Execution Monitor：状态 {status}。原因：{reason}",
            }
        ],
    }


def route_after_execution_monitor(state: State) -> ExecutionMonitorRoute:
    """Execution Monitor 后的条件边。"""

    if state["next_action"] == "recover":
        return "recover"
    return "evaluate"
