from __future__ import annotations

from typing import Any, Literal

from langgraph.types import interrupt

from .state import State


ApprovalRoute = Literal["execute", "evaluate"]


def _empty_resume_result(reason: str) -> tuple[bool, str, dict[str, Any], str]:
    """构造统一的审批恢复结果。"""

    return False, reason, {}, ""


def _approval_payload(state: State) -> dict[str, Any]:
    """构造要展示给人类的审批信息。"""

    task_id = state["current_task_id"]
    pending = dict(state.get("pending_approval", {}))
    return {
        "approval_id": pending.get("approval_id", ""),
        "task_id": task_id,
        "tool_name": pending.get("tool_name", state["tool_name"]),
        "tool_args": pending.get("tool_args", state["tool_args"]),
        "risk_level": pending.get("risk_level", state["risk_level"]),
        "reason": pending.get("reason", state["policy_reason"]),
        "triggered_rules": pending.get("triggered_rules", []),
        "risk_notes": pending.get("risk_notes", []),
    }


def _resume_is_approved(resume_value: Any, task_id: str) -> tuple[bool, str, dict[str, Any], str]:
    """解析 CLI / UI 恢复 graph 时传回来的审批结果。

    中文注释：
    恢复值不只包含 approved。
    它还可以包含：
    - approver_id：谁审批的。
    - modified_tool_args：审批人修改后的工具参数。

    这样就支持“批准，但把参数改安全一点再执行”。
    """

    if isinstance(resume_value, bool):
        return resume_value, "Approval Interrupt：收到布尔审批结果。", {}, ""
    if isinstance(resume_value, dict):
        approved = bool(resume_value.get("approved", False))
        returned_task_id = str(resume_value.get("task_id", task_id))
        if returned_task_id and returned_task_id != task_id:
            return _empty_resume_result(
                (
                    "Approval Interrupt：审批任务不匹配，"
                    f"期望 {task_id}，实际 {returned_task_id}。"
                ),
            )
        reason = str(
            resume_value.get("reason", "Approval Interrupt：收到结构化审批结果。")
        )
        modified_tool_args = dict(resume_value.get("modified_tool_args") or {})
        approver_id = str(resume_value.get("approver_id", ""))
        return approved, reason, modified_tool_args, approver_id
    return _empty_resume_result(
        f"Approval Interrupt：无法识别审批结果：{resume_value!r}。"
    )


def approval_interrupt_node(state: State) -> dict[str, object]:
    """Approval Interrupt：处理 Tool Policy 要求人工确认的工具调用。

    中文注释：
    Tool Policy 只负责判断是否需要审批。
    Approval Interrupt 节点负责真正暂停图、等待审批、恢复执行。

    当前版本已经接入 LangGraph interrupt：

        approval_interrupt_node
          -> interrupt(payload)
          -> CLI 展示审批请求
          -> Command(resume={"approved": True/False})
          -> approval_interrupt_node 从头恢复执行

    这样图不需要知道 CLI 怎么问用户，
    CLI 也不需要知道审批节点内部怎么更新 State。
    """

    task_tree = dict(state["task_tree"])
    task_id = state["current_task_id"]
    task = dict(task_tree.get(task_id, {}))
    approved = bool(state["human_approvals"].get(task_id, False))
    modified_tool_args: dict[str, Any] = {}
    approver_id = ""

    if not approved:
        resume_value = interrupt(_approval_payload(state))
        approved, approval_reason, modified_tool_args, approver_id = _resume_is_approved(
            resume_value,
            task_id,
        )
    else:
        approval_reason = "Approval Interrupt：用户已经提前批准该工具调用。"

    if approved:
        approvals = dict(state["human_approvals"])
        approvals[task_id] = True
        tool_args = modified_tool_args or dict(state["tool_args"])
        task["status"] = "approved"
        task["args"] = tool_args
        task["approval"] = {
            "approved": True,
            "approver_id": approver_id,
            "reason": approval_reason,
            "modified_tool_args": modified_tool_args,
        }
        task_tree[task_id] = task
        return {
            "task_tree": task_tree,
            "human_approvals": approvals,
            "pending_approval": {},
            "tool_args": tool_args,
            "policy_decision": "allow",
            "policy_reason": approval_reason,
            "next_action": "execute",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Approval Interrupt：任务 {task_id} 已批准，"
                        "进入 Sandbox Runner。"
                    ),
                }
            ],
        }

    reason = approval_reason
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
                "content": (
                    f"Approval Interrupt：任务 {task_id} 未获批准，"
                    "进入 Evaluator。"
                ),
            }
        ],
    }


def route_after_approval_interrupt(state: State) -> ApprovalRoute:
    """Approval Interrupt 后的路由。"""

    if state["next_action"] == "execute":
        return "execute"
    return "evaluate"
