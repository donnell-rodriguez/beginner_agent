from __future__ import annotations

from datetime import datetime, timezone
from difflib import unified_diff
from hashlib import sha256
from typing import Any
from uuid import uuid4

from .state import State
from .tooling.patch_tools import read_patch_plan_metadata
from .tooling.results import ToolResult, ToolValidation, tool_result_json_schema
from .tools import WRITE_TOOLS, read_text_for_snapshot, run_tool_model


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
WRITE_TOOL_SET = set(WRITE_TOOLS)


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


def _text_hash(text: str) -> str:
    """计算文本 hash，用于记录修改前后是否一致。"""

    return sha256(text.encode("utf-8")).hexdigest()


def _diff_text(path: str, before: str, after: str) -> str:
    """生成单文件 unified diff，方便审计和 Evaluator 检查。"""

    diff = unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff)


def _changed_line_count(diff: str) -> int:
    """粗略统计 diff 中真实增删行数，不统计 diff 头部。"""

    count = 0
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            count += 1
    return count


def _target_path_for_write_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """推断写工具最终会修改哪个文件。"""

    if tool_name == "apply_patch_plan":
        patch_plan_id = str(tool_args.get("patch_plan_id", ""))
        return str(read_patch_plan_metadata(patch_plan_id).get("path", ""))
    return str(tool_args.get("path", ""))


def _preflight_write_tool(state: State, tool_name: str, tool_args: dict[str, Any]) -> str:
    """Executor 最后一层写工具保护。

    中文注释：
    Policy 是第一道权限层，但生产级系统还会在真正执行前再检查一次。
    这叫 defense in depth。

    这里主要防几类错误修改：
    - 写工具绕过审批。
    - 直接 apply_patch 绕过 PatchPlan 治理流程。
    - apply_patch_plan 引用未验证或已过期的 PatchPlan。
    - rollback 没有 patch_history。
    """

    if tool_name not in WRITE_TOOL_SET:
        return ""
    task = dict(state["task_tree"].get(state["current_task_id"], {}))
    if state["policy_decision"] != "allow":
        return (
            "Executor preflight 拒绝："
            "写工具必须先通过 Tool Policy / Approval Interrupt。"
        )

    if tool_name == "apply_patch" and not task.get("allow_direct_patch", False):
        return (
            "Executor preflight 拒绝：普通流程禁止直接 apply_patch，"
            "请使用 patch_plan -> validate_patch_plan -> apply_patch_plan。"
        )

    if tool_name == "apply_patch_plan":
        patch_plan_id = str(tool_args.get("patch_plan_id", ""))
        try:
            metadata = read_patch_plan_metadata(patch_plan_id)
            target_path = str(metadata.get("path", ""))
            validated_hash = str(metadata.get("validated_file_hash", ""))
            current = read_text_for_snapshot(target_path)
        except ValueError as exc:
            return f"Executor preflight 拒绝：{exc}"
        if not metadata.get("validated"):
            return "Executor preflight 拒绝：PatchPlan 尚未通过 validate_patch_plan。"
        if not validated_hash or _text_hash(current) != validated_hash:
            return "Executor preflight 拒绝：目标文件在 PatchPlan 验证后发生变化。"

    if tool_name == "rollback" and not state["patch_history"]:
        return "Executor preflight 拒绝：rollback 需要 patch_history。"
    return ""


def _blocked_tool_result(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str,
    started_at: str,
) -> ToolResult:
    """构造一个未真正执行工具的 blocked ToolResult。"""

    return ToolResult(
        status="blocked",
        tool_name=tool_name,
        tool_args=tool_args,
        normalized_args=tool_args,
        output=reason,
        validation=ToolValidation(ok=False, reason=reason),
        metadata={"executor_preflight": True},
        diagnostics={"tool_result_schema": tool_result_json_schema()},
        started_at=started_at,
        duration_ms=0,
        error_type="executor_preflight",
        retryable=False,
    )


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

    preflight_error = _preflight_write_tool(state, tool_name, tool_args)
    write_target_path = ""
    if tool_name in WRITE_TOOL_SET and not preflight_error:
        try:
            write_target_path = _target_path_for_write_tool(tool_name, tool_args)
        except ValueError as exc:
            preflight_error = f"Executor preflight 拒绝：{exc}"

    # 中文注释：
    # 写入工具执行前先拍快照。
    # 这样写成功后可以记录 patch_history，后续 rollback 能恢复。
    if tool_name in {"apply_patch", "apply_patch_plan", "format_apply"} and write_target_path:
        try:
            before_content = read_text_for_snapshot(write_target_path)
        except ValueError:
            before_content = ""
    else:
        before_content = ""

    if preflight_error:
        tool_result_model = _blocked_tool_result(
            tool_name=tool_name,
            tool_args=tool_args,
            reason=preflight_error,
            started_at=started_at,
        )
    else:
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
    if (
        tool_name in {"apply_patch", "apply_patch_plan", "format_apply"}
        and result_status == "success"
    ):
        path = write_target_path or str(tool_args.get("path", ""))
        after_content = ""
        if path:
            try:
                after_content = read_text_for_snapshot(path)
            except ValueError:
                after_content = ""
        diff = _diff_text(path, before_content, after_content)
        patch_record = {
            "task_id": task_id,
            "tool_name": tool_name,
            "path": path,
            "before_content": before_content,
            "after_content": after_content,
            "before_hash": _text_hash(before_content),
            "after_hash": _text_hash(after_content),
            "diff": diff,
            "changed_line_count": _changed_line_count(diff),
            "patch_plan_id": str(tool_args.get("patch_plan_id", "")),
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
        "next_action": "monitor",
        "messages": [
            {
                "role": "assistant",
                "content": f"Executor：完成任务 {task_id}，进入 Execution Monitor。",
            }
        ],
    }
    if patch_record:
        update["patch_history"] = [patch_record]
    return update
