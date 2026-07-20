from __future__ import annotations

from typing import Any, Literal

from .node_utils import goal_progress_snapshot, new_task
from .state import State


# 中文注释：
# 这个文件是 Scheduler / Agenda Manager。
#
# 如果你要读懂 scheduler.py，第一步不是先看 if/for，
# 而是先知道它会读写 State 里的哪些字段。
#
# Scheduler 主要关心这些 State 字段：
#
# 1. user_input
#    用户最开始输入的任务。
#    当 task_tree 还是空的时候，Scheduler 会用它创建 root 任务。
#
# 2. root_task_id
#    根任务 id。
#    默认通常是 "root"。
#    整个任务树都是从 root 开始拆解的。
#
# 3. task_tree
#    任务树。
#    它是一个 dict：
#
#        {
#            "root": {...},
#            "root.1": {...},
#            "root.2": {...},
#        }
#
#    key 是 task_id，value 是任务详情。
#    Scheduler 不直接处理任务内容，只负责从里面找 pending 任务。
#
# 4. agenda
#    待处理任务队列。
#    它不是保存完整任务，而是保存 task_id 列表：
#
#        ["root.1", "root.2"]
#
#    真正任务内容仍然在 task_tree 里。
#
# 5. current_task_id
#    当前这一轮选中的任务 id。
#    Scheduler 会把它写入 State，后面的 Planner 会读取它。
#
# 6. step_count
#    当前复杂 agent loop 已经执行了多少轮。
#    每次进入 Scheduler，都会 +1。
#
# 7. max_steps
#    最大循环次数。
#    防止 agent 因为 LLM 判断错误而无限循环。
#
# 8. done
#    表示复杂 agent 是否应该结束。
#    如果达到 max_steps，或者没有 pending 任务，Scheduler 会设置 done=True。
#
# 9. next_action
#    表示图下一步应该去哪里。
#    Scheduler 正常选中任务后设置为 "plan"。
#    Scheduler 判断结束时设置为 "finish"。
#
# 10. tool_name / tool_args / tool_result_status
#     Scheduler 会把这些字段重置。
#     因为新任务开始前，不应该沿用上一个任务的工具状态。
#
#     tool_result_data 也会重置。
#     它是 Executor 写入的 Pydantic ToolResult dict。
#
# 11. goal_progress
#     当 Scheduler 结束循环时，会计算当前任务整体完成度。
#
# 一句话理解：
#
#   Scheduler 不负责“怎么做任务”。
#   Scheduler 只负责：
#
#     1. 如果没有 root 任务，就创建 root。
#     2. 从 agenda 里找下一个 pending 任务。
#     3. 把 current_task_id 写入 State。
#     4. 判断是否达到 max_steps 或没有任务可做。
#     5. 决定下一步去 Planner 还是 Summarizer。
#
SchedulerRoute = Literal["plan", "finish"]


def _ensure_root_task(state: State) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Memory / Checkpoint 恢复后，如果任务树为空，则初始化 root。"""

    # 中文注释：
    # 从 State 里取出 task_tree 和 agenda。
    # 这里用 dict(...) / list(...) 拷贝一份，避免直接原地修改旧对象。
    task_tree = dict(state["task_tree"])
    agenda = list(state["agenda"])

    # 中文注释：
    # 如果 task_tree 已经有内容，说明 root 任务已经创建过。
    # 可能是本轮之前创建的，也可能是 checkpoint 恢复出来的。
    if task_tree:
        return task_tree, agenda

    # 中文注释：
    # 如果 task_tree 为空，说明这是复杂 agent 第一次进入 Scheduler。
    # 此时需要把用户原始输入包装成 root 任务。
    root_id = state["root_task_id"] or "root"
    task_tree[root_id] = new_task(
        root_id,
        state["user_input"],
        parent_id=None,
        depth=0,
        reason="用户原始任务。",
    )

    # 中文注释：
    # agenda 里只放 task_id。
    # root 是第一个待处理任务，所以 agenda = ["root"]。
    agenda = [root_id]
    return task_tree, agenda


def _select_pending_task_id(task_tree: dict[str, dict[str, Any]], agenda: list[str]) -> str:
    """Scheduler 从 agenda 中选择下一个 pending 任务。"""

    # 中文注释：
    # 按 agenda 的顺序找第一个 status == "pending" 的任务。
    # 找到后返回它的 task_id。
    for task_id in agenda:
        task = task_tree.get(task_id)
        if task and task.get("status") == "pending":
            return task_id

    # 中文注释：
    # 如果没有 pending 任务，返回空字符串。
    # scheduler_node 会据此判断 agent loop 可以结束。
    return ""


def scheduler_node(state: State) -> dict[str, Any]:
    """3. Scheduler / Agenda Manager：决定下一个执行哪个任务。"""

    # 中文注释：
    # 每次进入 Scheduler，都表示 agent loop 往前推进了一轮。
    # 所以 step_count + 1。
    next_step_count = state["step_count"] + 1

    # 中文注释：
    # 确保任务树里至少有 root 任务。
    # 如果是第一次运行，会创建 root。
    # 如果是 checkpoint 恢复，会直接使用已有 task_tree。
    task_tree, agenda = _ensure_root_task(state)

    # 中文注释：
    # 如果超过最大循环次数，强制结束。
    # 这是 agent 工程里非常重要的安全阀，防止无限循环。
    if next_step_count > state["max_steps"]:
        return {
            "task_tree": task_tree,
            "agenda": [],
            "current_task_id": "",
            "tool_name": "none",
            "tool_args": {},
            "tool_result_data": {},
            "tool_result_status": "none",
            "execution_status": "not_started",
            "active_execution": {},
            "done": True,
            "next_action": "finish",
            "step_count": next_step_count,
            "goal_progress": goal_progress_snapshot(state, task_tree),
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Scheduler：达到 max_steps={state['max_steps']}，停止循环。",
                }
            ],
        }

    # 中文注释：
    # 从 agenda 中选出下一个 pending 任务。
    # 后面的 Planner 会根据 current_task_id 找到这个任务并处理它。
    current_task_id = _select_pending_task_id(task_tree, agenda)

    # 中文注释：
    # 如果没有 pending 任务，说明所有可做任务都处理完了。
    # 此时进入 summarize。
    if not current_task_id:
        return {
            "task_tree": task_tree,
            "agenda": [],
            "current_task_id": "",
            "tool_name": "none",
            "tool_args": {},
            "tool_result_data": {},
            "tool_result_status": "none",
            "execution_status": "not_started",
            "active_execution": {},
            "done": True,
            "next_action": "finish",
            "step_count": next_step_count,
            "goal_progress": goal_progress_snapshot(state, task_tree),
            "messages": [
                {"role": "assistant", "content": "Scheduler：没有 pending 任务，进入汇总。"}
            ],
        }

    # 中文注释：
    # 选中的任务要从 agenda 里移除。
    # 否则下一轮 Scheduler 可能重复选择同一个任务。
    agenda = [task_id for task_id in agenda if task_id != current_task_id]

    # 中文注释：
    # 返回一个“局部 State 更新”。
    # LangGraph 会把这里返回的字段合并回总 State。
    #
    # 关键变化：
    #   current_task_id = 本轮选中的任务
    #   next_action = "plan"
    #   done = False
    #   工具相关字段重置
    return {
        "task_tree": task_tree,
        "agenda": agenda,
        "current_task_id": current_task_id,
        "tool_name": "none",
        "tool_args": {},
        "tool_result_data": {},
        "tool_result_status": "none",
        "execution_status": "not_started",
        "active_execution": {},
        "done": False,
        "next_action": "plan",
        "step_count": next_step_count,
        "messages": [
            {
                "role": "assistant",
                "content": f"Scheduler：选择下一个 pending 任务 {current_task_id}。",
            }
        ],
    }


def route_after_scheduler(state: State) -> SchedulerRoute:
    """Scheduler 后的路由。"""

    # 中文注释：
    # 这是 LangGraph 的条件边函数。
    #
    # 如果 Scheduler 判断已经结束：
    #   返回 "finish"，graph.py 会跳到 summarize。
    #
    # 否则：
    #   返回 "plan"，graph.py 会跳到 planner_decomposer。
    if state["done"] or state["next_action"] == "finish":
        return "finish"
    return "plan"
