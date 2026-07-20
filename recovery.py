from __future__ import annotations

import json
from typing import Any, Literal

from .llm_client import chat_completion
from .node_utils import json_loads_from_model
from .state import RecoveryAction, State


RecoveryRoute = Literal["evaluate"]


def _local_recovery_action(task: dict[str, Any], state: State) -> tuple[RecoveryAction, str]:
    """本地规则恢复策略。

    中文注释：
    生产级 agent 不会每次失败都直接问 LLM。
    常见做法是：
    - 简单明确的问题用规则处理。
    - 复杂或连续失败时再让 LLM 参与重新规划。

    这样系统更稳定，也更容易做安全控制。
    """

    monitor_status = state["execution_monitor_status"]
    tool_result_status = state["tool_result_status"]
    active_execution = state.get("active_execution") or {}
    tool_name = state["tool_name"]
    retry_count = int(task.get("retry_count", 0))
    max_retries = int(state["max_task_retries"])
    long_running = bool(active_execution.get("long_running_tool"))

    if monitor_status == "blocked":
        return "ask_human", "执行被阻断，应该请求人工确认或调整权限。"

    if retry_count < max_retries and tool_result_status in ("failed", "empty"):
        if long_running:
            return "retry_with_new_args", "长任务失败或空结果，优先尝试缩小范围或调整参数。"
        return "retry_same", "普通工具失败且仍有重试额度，可以先重试同一任务。"

    if monitor_status == "over_budget":
        if tool_result_status in ("success", "partial"):
            return "stop_with_summary", "任务已返回结果但超过预算，应该先如实总结并留下继续线索。"
        return "decompose_more", "任务超过预算且结果不可用，应该拆成更小步骤。"

    if tool_result_status == "partial":
        return "stop_with_summary", "已经拿到部分结果，先总结已完成和未完成，避免继续无效消耗。"

    if long_running and retry_count >= max_retries:
        return "replan", "长任务已经没有重试额度，应该请求重新规划替代方案。"

    if tool_name in ("run_tests", "run_build", "run_cargo_test", "run_cargo_check"):
        return "replan", "验证类工具失败，应该根据失败输出重新规划修复路径。"

    return "stop_with_summary", "没有可靠恢复策略，应该停止并如实总结当前进展。"


def _llm_recovery_action(task: dict[str, Any], state: State) -> tuple[RecoveryAction, str]:
    """让 LLM 给出恢复建议。

    中文注释：
    LLM 在这里不是直接执行动作，而是给 Recovery Planner 一个建议。
    最终动作仍然会被本地白名单限制在 RecoveryAction 这些固定选项里。
    """

    response = chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你是 Recovery Planner。"
                    "你要根据执行状态决定下一步恢复动作。"
                    "action 只能是 retry_same、retry_with_new_args、use_alternative_tool、"
                    "replan、decompose_more、ask_human、stop_with_summary。"
                    "不要直接要求执行危险操作。只返回严格 JSON。"
                    '格式：{"action":"...","reason":"一句话原因"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户目标：{state['user_input']}\n\n"
                    f"当前任务：{json.dumps(task, ensure_ascii=False)}\n\n"
                    f"监控状态：{state['execution_monitor_status']}\n"
                    f"监控原因：{state['execution_monitor_reason']}\n"
                    f"执行摘要：{json.dumps(state.get('active_execution', {}), ensure_ascii=False)}\n"
                    f"工具结果状态：{state['tool_result_status']}\n"
                    f"工具输出：{state['tool_result'][:3000]}"
                ),
            },
        ],
        temperature=0,
        max_tokens=260,
    )
    data = json_loads_from_model(response)
    if not isinstance(data, dict):
        raise ValueError("Recovery Planner 返回的不是 JSON object。")
    action = str(data.get("action", "")).strip()
    reason = str(data.get("reason") or "Recovery Planner 未提供原因。")
    allowed = set(RecoveryAction.__args__)  # type: ignore[attr-defined]
    if action not in allowed or action == "none":
        raise ValueError("Recovery action 不合法。")
    return action, reason  # type: ignore[return-value]


def _should_ask_llm(task: dict[str, Any], state: State) -> bool:
    """判断当前恢复策略是否值得请求 LLM。"""

    if state["execution_monitor_status"] in ("blocked", "partial"):
        return False
    if int(task.get("retry_count", 0)) >= int(state["max_task_retries"]):
        return True
    return state["execution_monitor_status"] in ("over_budget", "failed")


def _partial_result(state: State) -> str:
    """提取当前可复用的部分结果。"""

    tool_result = state["tool_result"].strip()
    if not tool_result:
        return ""
    return tool_result[:4000]


def _resume_hint(action: RecoveryAction, state: State) -> str:
    """生成下次继续时的建议入口。"""

    task_id = state["current_task_id"]
    tool_name = state["tool_name"]
    if action in ("retry_same", "retry_with_new_args"):
        return f"可以从任务 {task_id} 继续，优先重新执行或调整工具 {tool_name} 的参数。"
    if action in ("replan", "decompose_more", "use_alternative_tool"):
        return f"可以从任务 {task_id} 继续，先根据失败输出重新拆解或选择替代工具。"
    if action == "ask_human":
        return f"任务 {task_id} 需要人工确认权限或目标边界后再继续。"
    return f"任务 {task_id} 已停止在当前阶段，可根据 partial_result 和 memory 继续。"


def recovery_planner_node(state: State) -> dict[str, Any]:
    """Recovery Planner：决定长任务失败、超预算或部分完成后的恢复动作。

    中文注释：
    这个节点落实你的想法：

    - 长时间没有可靠结果：不要一直等。
    - 能重试就重试。
    - 需要换方案就请求 LLM 给恢复建议。
    - 仍然无法确定就停止并如实总结。

    它不会直接修改文件，也不会绕过 Tool Policy。
    它只把恢复意图写入 State，后面的 Evaluator / Committer 再按规则落地。
    """

    task_id = state["current_task_id"]
    task = dict(state["task_tree"].get(task_id, {}))

    action, reason = _local_recovery_action(task, state)
    if _should_ask_llm(task, state):
        try:
            action, reason = _llm_recovery_action(task, state)
        except (RuntimeError, ValueError, json.JSONDecodeError):
            pass

    partial_result = _partial_result(state)
    resume_hint = _resume_hint(action, state)
    return {
        "recovery_action": action,
        "recovery_reason": reason,
        "partial_result": partial_result,
        "resume_hint": resume_hint,
        "next_action": "evaluate",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"Recovery Planner：恢复动作 {action}。原因：{reason} "
                    f"继续提示：{resume_hint}"
                ),
            }
        ],
    }


def route_after_recovery_planner(state: State) -> RecoveryRoute:
    """Recovery Planner 后进入 Evaluator。"""

    return "evaluate"
