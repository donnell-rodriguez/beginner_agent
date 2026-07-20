from __future__ import annotations

from typing import Any

from .state import State
from .tools import read_text_for_snapshot, run_tool_model


def executor_node(state: State) -> dict[str, Any]:
    """5. Executor：真正执行工具。"""

    task_tree = dict(state["task_tree"])
    task_id = state["current_task_id"]
    task = dict(task_tree.get(task_id, {}))
    tool_name = state["tool_name"]
    tool_args = dict(state["tool_args"])
    patch_record: dict[str, Any] | None = None

    # 中文注释：
    # rollback 是恢复动作，通常由 Evaluator 根据 patch_history 触发。
    # 如果 Planner 没有提供 content，这里自动取最近一次修改前的内容。
    if tool_name == "rollback" and "content" not in tool_args and state["patch_history"]:
        last_patch = state["patch_history"][-1]
        tool_args = {
            "path": last_patch.get("path", ""),
            "content": last_patch.get("before_content", ""),
        }

    # 中文注释：
    # 写入工具执行前先拍快照。
    # 这样 apply_patch 成功后可以记录 patch_history，后续 rollback 能恢复。
    if tool_name == "apply_patch":
        path = str(tool_args.get("path", ""))
        try:
            before_content = read_text_for_snapshot(path)
        except ValueError:
            before_content = ""
    else:
        before_content = ""

    tool_result_model = run_tool_model(tool_name, tool_args)
    tool_result_data = tool_result_model.model_dump(mode="json")
    tool_result = tool_result_model.output
    result_status = tool_result_model.status
    task["status"] = "executed" if result_status != "blocked" else "failed"
    if tool_name == "apply_patch" and result_status == "success":
        path = str(tool_args.get("path", ""))
        after_content = ""
        if path:
            try:
                after_content = read_text_for_snapshot(path)
            except ValueError:
                after_content = ""
        patch_record = {
            "task_id": task_id,
            "tool_name": tool_name,
            "path": path,
            "before_content": before_content,
            "after_content": after_content,
            "result": tool_result,
        }

    task["result"] = tool_result
    task["tool_result_status"] = result_status
    task["tool_result_data"] = tool_result_data
    task_tree[task_id] = task
    update: dict[str, Any] = {
        "task_tree": task_tree,
        "tool_result": tool_result,
        "tool_result_status": result_status,
        "tool_result_data": tool_result_data,
        "draft": (
            f"Executor 完成任务：{task.get('title', '')}\n"
            f"任务 id：{task_id}\n"
            f"工具名称：{tool_name}\n"
            f"工具参数：{tool_args}\n"
            f"工具状态：{result_status}\n"
            f"耗时：{tool_result_model.duration_ms}ms\n"
            f"可重试：{tool_result_model.retryable}\n"
            f"工具结果：\n{tool_result}"
        ),
        "next_action": "evaluate",
        "messages": [
            {
                "role": "assistant",
                "content": f"Executor：完成任务 {task_id}，进入 Evaluator。",
            }
        ],
    }
    if patch_record:
        update["patch_history"] = [patch_record]
    return update
