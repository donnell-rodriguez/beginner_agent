from __future__ import annotations

from typing import Literal

from .state import State


ApprovalRoute = Literal["execute", "evaluate"]


def human_approval_node(state: State) -> dict[str, object]:
    """Human Approval：处理 Tool Policy 要求人工确认的工具调用。

    中文注释：
    Tool Policy 只负责判断是否需要审批。
    Human Approval 节点负责真正检查审批结果。

    当前 beginner_agent 没有 UI，所以审批结果来自：

        state["human_approvals"][current_task_id] == True

    如果没有审批，就不会执行工具，而是把任务标记为 blocked，
    交给 Evaluator / Task Committer 处理。

    以后如果接入 CLI / Web UI / LangGraph interrupt，
    应该优先升级这个节点。
    """

    task_tree = dict(state["task_tree"])
    task_id = state["current_task_id"]
    task = dict(task_tree.get(task_id, {}))
    approved = bool(state["human_approvals"].get(task_id, False))

    if approved:
        task["status"] = "approved"
        task_tree[task_id] = task
        return {
            "task_tree": task_tree,
            "pending_approval": {},
            "policy_decision": "allow",
            "policy_reason": "Human Approval：用户已经批准该工具调用。",
            "next_action": "execute",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Human Approval：任务 {task_id} 已批准，进入 Executor。",
                }
            ],
        }

    reason = "Human Approval：缺少人工审批，工具调用被阻断。"
    task["status"] = "blocked"
    task["result"] = reason
    task["tool_result_status"] = "blocked"
    task_tree[task_id] = task
    return {
        "task_tree": task_tree,
        "tool_result": reason,
        "tool_result_status": "blocked",
        "policy_decision": "ask",
        "policy_reason": reason,
        "next_action": "evaluate",
        "messages": [
            {
                "role": "assistant",
                "content": f"Human Approval：任务 {task_id} 未获批准，进入 Evaluator。",
            }
        ],
    }


def route_after_human_approval(state: State) -> ApprovalRoute:
    """Human Approval 后的路由。"""

    if state["next_action"] == "execute":
        return "execute"
    return "evaluate"
