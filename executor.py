from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .state import State
from .tools import read_text_for_snapshot, run_tool_model


LONG_RUNNING_TOOLS = {
    "get_diagnostics",
    "lint_typecheck",
    "run_allowed_command",
    "run_build",
    "run_cargo_check",
    "run_cargo_clippy",
    "run_cargo_fmt_check",
    "run_cargo_test",
    "run_impacted_tests",
    "run_mypy",
    "run_package_script",
    "run_pytest",
    "run_ruff",
    "run_targeted_tests",
    "run_tests",
    "run_typecheck",
    "static_check",
}


def _utc_now() -> str:
    """返回 UTC 时间字符串，方便执行记录跨机器对齐。"""

    return datetime.now(timezone.utc).isoformat()


def _execution_status(result_status: str, duration_ms: int, budget_ms: int) -> str:
    """把工具结果状态升级成 Executor 视角的执行状态。

    中文注释：
    tool_result_status 只回答“工具返回结果怎么样”。
    execution_status 额外回答“执行过程怎么样”。

    例如：
    - 工具被参数校验拦截：blocked
    - 工具失败：failed
    - 工具成功但超过预算：completed_over_budget
    - 工具正常完成：completed
    """

    if result_status == "blocked":
        return "blocked"
    if result_status == "failed":
        return "failed"
    if budget_ms > 0 and duration_ms > budget_ms:
        return "completed_over_budget"
    return "completed"


def _is_long_running_tool(tool_name: str) -> bool:
    """判断工具是否属于可能耗时较长的一类。"""

    return tool_name in LONG_RUNNING_TOOLS


def _build_execution_attempt(
    *,
    task_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    result_status: str,
    execution_status: str,
    duration_ms: int,
    budget_ms: int,
    retryable: bool,
    started_at: str,
    finished_at: str,
    tool_result_data: dict[str, Any],
) -> dict[str, Any]:
    """生成一次工具执行尝试记录。

    中文注释：
    生产级 agent 通常不会只保存“工具输出字符串”。
    它会保存一次执行的 run record / attempt record：
    谁执行的、执行什么、用了多久、有没有超预算、能不能重试。

    当前项目还没有接后台队列，所以 execution_mode 先写成：
    - sync：普通同步工具。
    - long_running_sync：可能耗时较长，但当前仍在本进程同步执行。

    后续如果接 Celery / Prefect / Kafka worker，
    这里可以升级成 async_job，并保存 job_id。
    """

    is_long_running = _is_long_running_tool(tool_name)
    return {
        "attempt_id": uuid4().hex,
        "task_id": task_id,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "execution_mode": "long_running_sync" if is_long_running else "sync",
        "execution_status": execution_status,
        "tool_result_status": result_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "budget_ms": budget_ms,
        "over_budget": budget_ms > 0 and duration_ms > budget_ms,
        "long_running_tool": is_long_running,
        "retryable": retryable,
        "future_worker_contract": {
            "job_id": "",
            "poll_after_ms": 0,
            "cancel_supported": False,
            "resume_supported": False,
        },
        "tool_result_data": tool_result_data,
    }


def executor_node(state: State) -> dict[str, Any]:
    """5. Executor：真正执行工具。

    中文注释：
    当前 Executor 已经不是简单的 run_tool(...) 包装。
    它负责把“工具调用”升级成“可审计的执行尝试”：
    - 记录开始/结束时间。
    - 记录耗时和预算。
    - 区分短工具和可能长时间运行的工具。
    - 写入 execution_attempts，方便后续 Evaluator / Memory / Audit 使用。

    重要边界：
    当前项目里的工具函数仍然是同步执行。
    所以这里还不能真正做到“后台执行 + 等待 + 恢复”。
    但 execution_attempts / active_execution 已经给后续接 worker 留好了接口。
    """

    task_tree = dict(state["task_tree"])
    task_id = state["current_task_id"]
    task = dict(task_tree.get(task_id, {}))
    tool_name = state["tool_name"]
    tool_args = dict(state["tool_args"])
    budget_ms = int(state.get("max_tool_duration_ms", 30000))
    started_at = _utc_now()
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
    finished_at = _utc_now()
    tool_result_data = tool_result_model.model_dump(mode="json")
    tool_result = tool_result_model.output
    result_status = tool_result_model.status
    execution_status = _execution_status(
        result_status,
        int(tool_result_model.duration_ms),
        budget_ms,
    )
    execution_attempt = _build_execution_attempt(
        task_id=task_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result_status=result_status,
        execution_status=execution_status,
        duration_ms=int(tool_result_model.duration_ms),
        budget_ms=budget_ms,
        retryable=bool(tool_result_model.retryable),
        started_at=started_at,
        finished_at=finished_at,
        tool_result_data=tool_result_data,
    )
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
    task["execution_status"] = execution_status
    task["execution_attempt_id"] = execution_attempt["attempt_id"]
    task_tree[task_id] = task
    update: dict[str, Any] = {
        "task_tree": task_tree,
        "tool_result": tool_result,
        "tool_result_status": result_status,
        "tool_result_data": tool_result_data,
        "execution_status": execution_status,
        "active_execution": execution_attempt,
        "execution_attempts": [execution_attempt],
        "draft": (
            f"Executor 完成任务：{task.get('title', '')}\n"
            f"任务 id：{task_id}\n"
            f"工具名称：{tool_name}\n"
            f"工具参数：{tool_args}\n"
            f"执行状态：{execution_status}\n"
            f"工具状态：{result_status}\n"
            f"耗时：{tool_result_model.duration_ms}ms\n"
            f"预算：{budget_ms}ms\n"
            f"长任务工具：{execution_attempt['long_running_tool']}\n"
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
