from __future__ import annotations

from typing import Any, Literal

from .state import State
from .tools import validate_tool_request


PlanValidatorRoute = Literal["schedule", "policy", "evaluate"]


# todo 我的问题是，对应生成的子任务生成的合理还是不合理，对还是不对好像没有校对呀，这个应该怎么处理的呢？
def _validate_plan_locally(
    task: dict[str, Any], task_tree: dict[str, dict[str, Any]]
) -> tuple[bool, str]:
    """Plan Validator 的本地质量检查。

    中文注释：
    Planner 可以用 LLM 生成计划，但计划不能直接执行。
    这里先用确定性代码检查结构、安全边界和基本可执行性。

    当前检查的是第一层硬规则：
    - children 是否重复。
    - children 是否真实存在。
    - 子任务是否绑定可用工具。
    - 叶子任务的工具参数是否安全。

    后续 README 里还有 TODO，可以继续升级语义质量检查、成本风险检查等。
    """

    child_ids = list(task.get("children") or [])
    if child_ids:
        if len(child_ids) != len(set(child_ids)):
            return False, "计划里存在重复 child id。"

        missing_children = [child_id for child_id in child_ids if child_id not in task_tree]
        if missing_children:
            return False, f"计划引用了不存在的子任务：{missing_children}。"

        for child_id in child_ids:
            child = task_tree[child_id]
            tool = str(child.get("tool", "none"))
            args = child.get("args", {})
            if not isinstance(args, dict):
                return False, f"子任务 {child_id} 的 args 不是 dict。"
            is_valid, reason = validate_tool_request(tool, args)
            if not is_valid:
                return False, f"子任务 {child_id} 不可执行：{reason}"

        return True, "计划可用：子任务存在、无重复、工具和参数都在允许范围内。"

    tool = str(task.get("tool", "none"))
    args = task.get("args", {})
    if not isinstance(args, dict):
        return False, "计划不可用：当前任务 args 不是 dict。"

    is_valid, reason = validate_tool_request(tool, args)
    if is_valid:
        return True, f"计划可用：当前任务已经足够具体，可以进入工具策略检查。{reason}"
    return False, f"计划不可用：既没有子任务，也没有可执行工具。{reason}"


def plan_validator_node(state: State) -> dict[str, Any]:
    """Plan Validator：评估 Planner 生成的计划质量。"""

    task_tree = dict(state["task_tree"])
    task_id = state["current_task_id"]
    task = dict(task_tree.get(task_id, {}))
    is_valid, reason = _validate_plan_locally(task, task_tree)

    if not is_valid:
        task["status"] = "failed"
        task["result"] = reason
        task_tree[task_id] = task
        return {
            "task_tree": task_tree,
            "plan_validation_status": "invalid",
            "plan_validation_reason": reason,
            "tool_result": reason,
            "tool_result_status": "failed",
            "next_action": "evaluate",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Plan Validator：计划无效，进入 Evaluator。原因：{reason}",
                }
            ],
        }

    # 中文注释：
    # 有 children，说明当前任务已经拆成了多个子任务。
    # 当前任务本身不执行工具，应该回到 Scheduler，让 Scheduler 选择下一个子任务。
    if task.get("children"):
        return {
            "plan_validation_status": "valid",
            "plan_validation_reason": reason,
            "current_task_id": "",
            "tool_name": "none",
            "tool_args": {},
            "tool_result_status": "none",
            "next_action": "schedule",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Plan Validator：拆解计划有效，回到 Scheduler。原因：{reason}",
                }
            ],
        }

    # 中文注释：
    # 没有 children，说明这是叶子任务。
    # Validator 已经确认它绑定了安全工具，下一步进入 Tool Policy 做权限判断。
    return {
        "plan_validation_status": "valid",
        "plan_validation_reason": reason,
        "next_action": "policy",
        "messages": [
            {
                "role": "assistant",
                "content": f"Plan Validator：执行计划有效，进入 Tool Policy。原因：{reason}",
            }
        ],
    }


def route_after_plan_validator(state: State) -> PlanValidatorRoute:
    """Plan Validator 后的路由。"""

    if state["next_action"] == "schedule":
        return "schedule"
    if state["next_action"] == "evaluate":
        return "evaluate"
    return "policy"
