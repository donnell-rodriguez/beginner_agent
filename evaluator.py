from __future__ import annotations

import json
from typing import Any, Literal

from .llm_client import chat_completion
from .node_utils import fallback_subtasks, goal_progress_snapshot, json_loads_from_model, new_task
from .state import State


EvaluatorRoute = Literal["commit"]
TaskCommitterRoute = Literal["memory", "finish"]
SOURCE_WRITE_TOOLS = {"apply_patch", "apply_patch_plan", "format_apply"}


def _fallback_evaluation(task: dict[str, Any], state: State) -> tuple[str, str]:
    """Evaluator 失败时使用的本地判断。"""

    result_status = str(task.get("tool_result_status") or state["tool_result_status"])
    tool_result_data = task.get("tool_result_data") or state.get("tool_result_data", {})
    retryable = bool(tool_result_data.get("retryable")) if isinstance(tool_result_data, dict) else False
    if result_status == "blocked":
        return "fail", "工具被权限层阻断，当前任务不能继续执行。"
    if result_status == "empty":
        return "retry", "工具返回为空，应该重试。"
    if result_status == "failed":
        if retryable and int(task.get("retry_count", 0)) < state["max_task_retries"]:
            return "retry", "ToolResult 标记 retryable=True，且仍有重试额度。"
        return "fail", "工具执行失败，且已经没有重试额度。"
    if result_status == "partial":
        return "complete", "工具返回了部分内容，当前任务可先标记完成，后续可继续补充。"

    result = str(task.get("result", ""))
    if not result.strip():
        return "retry", "结果为空，应该重试。"
    if any(marker in result for marker in ("不存在", "不允许", "拒绝", "安全检查失败", "未知工具")):
        if int(task.get("retry_count", 0)) < state["max_task_retries"]:
            return "retry", "工具结果包含错误信息，且仍有重试额度。"
        return "fail", "工具结果包含错误信息，且已经没有重试额度。"
    return "complete", "工具返回了可用结果。"


def _should_skip_llm_evaluation(task: dict[str, Any], state: State) -> tuple[bool, str, str]:
    """判断是否应该跳过 LLM 评估，直接使用本地规则。

    中文注释：
    生产级 Evaluator 不会把所有判断都交给模型。
    对 blocked / failed / empty 这类明确状态，本地规则比 LLM 更稳定。
    """

    recovery_action = state.get("recovery_action", "none")
    if recovery_action == "retry_same":
        return True, "retry", "Recovery Planner 建议用同一方式重试。"
    if recovery_action == "retry_with_new_args":
        return True, "retry", "Recovery Planner 建议调整参数后重试。"
    if recovery_action in ("replan", "decompose_more", "use_alternative_tool"):
        return True, "expand", f"Recovery Planner 建议 {recovery_action}，应重新拆解或换方案。"
    if recovery_action == "ask_human":
        return True, "fail", "Recovery Planner 判断需要人工确认，当前任务先停止。"
    if recovery_action == "stop_with_summary":
        return True, "fail", "Recovery Planner 判断继续消耗不划算，应停止并总结已完成/未完成。"

    result_status = str(task.get("tool_result_status") or state["tool_result_status"])
    if result_status in ("blocked", "failed", "empty"):
        decision, reason = _fallback_evaluation(task, state)
        return True, decision, f"本地规则优先：{reason}"
    return False, "none", ""


def _evaluate_parent_task(
    task: dict[str, Any], task_tree: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Parent Task Evaluation：根据子任务状态更新父任务。"""

    parent_id = task.get("parent_id")
    if not parent_id or parent_id not in task_tree:
        return {"parent_id": None, "status": "none", "reason": "当前任务没有父任务。"}

    parent = dict(task_tree[parent_id])
    child_ids = list(parent.get("children") or [])
    children = [task_tree[child_id] for child_id in child_ids if child_id in task_tree]
    if not children:
        return {"parent_id": parent_id, "status": "none", "reason": "父任务没有可评估的子任务。"}

    done_count = sum(1 for child in children if child.get("status") == "done")
    failed_count = sum(1 for child in children if child.get("status") in ("failed", "blocked"))
    pending_count = len(children) - done_count - failed_count

    if pending_count > 0:
        parent_status = "partial"
        reason = "父任务还有子任务未完成。"
    elif failed_count > 0 and done_count > 0:
        parent_status = "needs_more"
        reason = "父任务部分完成，但存在失败子任务，可能需要补充任务。"
    elif failed_count > 0:
        parent_status = "failed"
        reason = "父任务的子任务都失败或被阻断。"
    else:
        parent_status = "done"
        reason = "父任务的所有子任务都已完成。"

    parent["status"] = parent_status
    parent["result"] = reason
    task_tree[parent_id] = parent
    return {
        "parent_id": parent_id,
        "status": parent_status,
        "done_children": done_count,
        "failed_children": failed_count,
        "pending_children": pending_count,
        "reason": reason,
    }


def _make_rollback_task(state: State, task: dict[str, Any]) -> dict[str, Any] | None:
    """根据最近一次 patch_history 创建 rollback 任务。

    中文注释：
    真实 code repair agent 的关键闭环是：
    修改代码 -> 运行测试 -> 测试失败 -> 回滚或继续修。

    当前教学版先实现最保守的失败恢复：
    如果测试/lint 失败，并且存在历史 patch，就安排 rollback。
    rollback 仍然是写操作，所以后面还会经过 Tool Policy 的人工审批。
    """

    if not state["patch_history"]:
        return None
    if task.get("tool") not in ("run_tests", "lint_typecheck", "static_check"):
        return None

    last_patch = state["patch_history"][-1]
    path = str(last_patch.get("path", ""))
    before_content = str(last_patch.get("before_content", ""))
    if not path or not before_content:
        return None

    parent_id = str(task.get("parent_id") or task.get("id") or "root")
    rollback_id = f"{parent_id}.rollback.{len(state['patch_history'])}"
    return new_task(
        rollback_id,
        f"回滚最近一次修改：{path}",
        parent_id=task.get("parent_id"),
        depth=int(task.get("depth", 0)),
        tool="rollback",
        args={"path": path, "content": before_content},
        reason="测试或 lint 失败，安排回滚恢复。",
    )


def _changed_files_for_task(task: dict[str, Any], state: State) -> list[str]:
    """从 ToolResult / patch_history 中提取本次写操作影响的文件。"""

    tool_result_data = task.get("tool_result_data") or state.get("tool_result_data", {})
    changed_files = []
    if isinstance(tool_result_data, dict):
        raw = tool_result_data.get("changed_files", [])
        if isinstance(raw, list):
            changed_files.extend(str(path) for path in raw if str(path))
    if not changed_files and state["patch_history"]:
        last_patch = state["patch_history"][-1]
        path = str(last_patch.get("path", ""))
        if path:
            changed_files.append(path)
    return sorted(set(changed_files))


def _verification_tasks_for_write(
    task: dict[str, Any], state: State, changed_files: list[str]
) -> list[dict[str, Any]]:
    """写操作成功后自动生成验证任务。

    中文注释：
    大厂式 code agent 不应该“改完就算完成”。
    写操作成功后，至少要安排几类验证：

    - git_diff_file：确认实际 diff。
    - secret_scan：确认没有引入明显密钥。
    - static_check：确认 Python 代码基本能编译。
    - run_targeted_tests：跑受控测试入口。

    这些任务仍然会经过 Scheduler / Policy / Executor / Evaluator，
    所以不会绕过现有安全链路。
    """

    if not changed_files:
        return []
    if task.get("verification_scheduled"):
        return []
    if len(state["task_tree"]) >= state["max_total_tasks"]:
        return []

    parent_id = str(task.get("id", state["current_task_id"]))
    depth = int(task.get("depth", 0)) + 1
    first_file = changed_files[0]
    specs = [
        ("diff", "检查本次修改 diff 是否只包含预期内容", "git_diff_file", {"path": first_file}),
        ("secret", "扫描本次修改是否引入明显密钥", "secret_scan", {"path": first_file}),
        ("static", "运行静态检查确认代码可解析", "static_check", {}),
        (
            "tests",
            "运行受控测试确认修改没有破坏主流程",
            "run_targeted_tests",
            {"target": "beginner_agent"},
        ),
    ]
    available_slots = max(0, state["max_total_tasks"] - len(state["task_tree"]))
    tasks: list[dict[str, Any]] = []
    for suffix, title, tool, args in specs[:available_slots]:
        tasks.append(
            new_task(
                f"{parent_id}.verify.{suffix}",
                title,
                parent_id=parent_id,
                depth=depth,
                tool=tool,
                args=args,
                reason="写操作完成后自动安排验证，避免错误修改被误判为完成。",
            )
        )
    return tasks


def _llm_evaluate(task: dict[str, Any], state: State) -> tuple[str, str]:
    """让 LLM 检查结果是否完成、是否需要重试或继续拆。"""

    response = chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你是 Evaluator / Verifier。"
                    "请检查当前任务结果是否有用。"
                    "decision 只能是 complete、retry、expand、fail。"
                    "complete 表示结果可用；retry 表示同一任务应重试；"
                    "expand 表示需要补充更多子任务；fail 表示无法完成。"
                    "只返回严格 JSON，不要解释。"
                    '格式：{"decision":"complete|retry|expand|fail","reason":"一句话原因"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户目标：{state['user_input']}\n\n"
                    f"当前任务：{json.dumps(task, ensure_ascii=False)}\n\n"
                    f"已完成任务数量：{len(state['completed_tasks'])}"
                ),
            },
        ],
        temperature=0,
        max_tokens=220,
    )
    data = json_loads_from_model(response)
    if not isinstance(data, dict):
        raise ValueError("Evaluator 返回的不是 JSON object。")
    decision = str(data.get("decision", "")).lower().strip()
    reason = str(data.get("reason") or "Evaluator 未提供原因。")
    if decision not in ("complete", "retry", "expand", "fail"):
        raise ValueError("Evaluator decision 不合法。")
    return decision, reason


def evaluator_verifier_node(state: State) -> dict[str, Any]:
    """6A. Evaluator / Verifier：只判断结果质量。

    中文注释：
    这个节点现在只负责回答一个问题：

        当前任务结果应该 complete / retry / expand / fail？

    它不直接修改 task_tree / agenda / memory。
    真正写回任务状态的动作交给 task_committer_node。
    这样 Evaluator 更像生产级 agent 里的“评估器”，而不是评估、调度、记忆混在一起。
    """

    task_id = state["current_task_id"]
    task = dict(state["task_tree"].get(task_id, {}))

    skip_llm, local_decision, local_reason = _should_skip_llm_evaluation(task, state)
    if skip_llm:
        decision, reason = local_decision, local_reason
    else:
        try:
            decision, reason = _llm_evaluate(task, state)
        except (RuntimeError, ValueError, json.JSONDecodeError):
            decision, reason = _fallback_evaluation(task, state)

    return {
        "evaluation_decision": decision,
        "evaluation_reason": reason,
        "next_action": "commit",
        "messages": [
            {
                "role": "assistant",
                "content": f"Evaluator：任务 {task_id} 的判断是 {decision}。原因：{reason}",
            }
        ],
    }


def task_committer_node(state: State) -> dict[str, Any]:
    """6B. Task Committer：把 Evaluator 的判断写回任务树。

    中文注释：
    Evaluator 只负责“判断”。
    Committer 负责“根据判断更新状态”：

    - retry：增加 retry_count，把任务重新放回 agenda。
    - expand：补充子任务。
    - fail + 测试类工具失败：必要时创建 rollback 任务。
    - complete / fail：写入 completed_tasks，并生成 pending_memory 交给 Memory Writer。

    这种拆法更接近生产级系统：
    判断逻辑和状态提交逻辑分开，后续更容易加审计、事务、回滚和恢复。
    """

    task_tree = dict(state["task_tree"])
    agenda = list(state["agenda"])
    task_id = state["current_task_id"]
    task = dict(task_tree.get(task_id, {}))
    decision = state["evaluation_decision"]
    reason = state["evaluation_reason"]

    if decision == "retry" and int(task.get("retry_count", 0)) < state["max_task_retries"]:
        task["retry_count"] = int(task.get("retry_count", 0)) + 1
        task["status"] = "pending"
        task_tree[task_id] = task
        agenda = [task_id] + agenda
        parent_evaluation = _evaluate_parent_task(task, task_tree)
        return {
            "task_tree": task_tree,
            "agenda": agenda,
            "current_task_id": "",
            "evaluation_decision": "retry",
            "evaluation_reason": reason,
            "parent_evaluation": parent_evaluation,
            "goal_progress": goal_progress_snapshot(state, task_tree),
            "next_action": "memory",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator：任务 {task_id} 需要重试。原因：{reason}",
                }
            ],
        }

    if decision == "expand" and task.get("depth", 0) < state["max_depth"]:
        subtasks = fallback_subtasks(task, {**state, "task_tree": task_tree})
        if subtasks:
            child_ids = [subtask["id"] for subtask in subtasks]
            task["status"] = "expanded"
            task["children"] = child_ids
            task_tree[task_id] = task
            for subtask in subtasks:
                task_tree[subtask["id"]] = subtask
            agenda = child_ids + agenda
            parent_evaluation = _evaluate_parent_task(task, task_tree)
            return {
                "task_tree": task_tree,
                "agenda": agenda,
                "current_task_id": "",
                "evaluation_decision": "expand",
                "evaluation_reason": reason,
                "parent_evaluation": parent_evaluation,
                "goal_progress": goal_progress_snapshot(state, task_tree),
                "next_action": "memory",
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"Evaluator：任务 {task_id} 需要继续拆解。原因：{reason}",
                    }
                ],
            }

    if decision in ("retry", "fail") and task.get("tool") in (
        "run_tests",
        "lint_typecheck",
        "static_check",
    ):
        rollback_task = _make_rollback_task(state, task)
        if rollback_task:
            task["status"] = "failed"
            task_tree[task_id] = task
            task_tree[rollback_task["id"]] = rollback_task
            agenda = [rollback_task["id"]] + agenda
            parent_evaluation = _evaluate_parent_task(task, task_tree)
            return {
                "task_tree": task_tree,
                "agenda": agenda,
                "current_task_id": "",
                "evaluation_decision": "fail",
                "evaluation_reason": f"{reason} 已安排 rollback 任务。",
                "parent_evaluation": parent_evaluation,
                "goal_progress": goal_progress_snapshot(state, task_tree),
                "next_action": "memory",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            f"Evaluator：检测到验证失败，安排 {rollback_task['id']} "
                            f"回滚最近一次修改。"
                        ),
                    }
                ],
            }

    final_status = "done" if decision == "complete" else "failed"
    task["status"] = final_status
    task["tool_result_status"] = task.get("tool_result_status") or state["tool_result_status"]
    verification_tasks: list[dict[str, Any]] = []
    if (
        final_status == "done"
        and task.get("tool") in SOURCE_WRITE_TOOLS
        and task.get("tool_result_status") == "success"
    ):
        changed_files = _changed_files_for_task(task, state)
        verification_tasks = _verification_tasks_for_write(task, state, changed_files)
        if verification_tasks:
            task["verification_scheduled"] = True
            task["verification_task_ids"] = [item["id"] for item in verification_tasks]
    if verification_tasks:
        final_status = "pending_verification"
        task["status"] = final_status
        task["children"] = [item["id"] for item in verification_tasks]
    task_tree[task_id] = task
    if verification_tasks:
        for verification_task in verification_tasks:
            task_tree[verification_task["id"]] = verification_task
        agenda = [item["id"] for item in verification_tasks] + agenda
    parent_evaluation = _evaluate_parent_task(task, task_tree)
    goal_progress = goal_progress_snapshot(state, task_tree)
    completed_records = [] if verification_tasks else [{**task, "status": final_status}]
    memory_note = {
        "task_id": task_id,
        "title": task.get("title", ""),
        "decision": decision,
        "reason": reason,
        "tool_result_status": task.get("tool_result_status", "none"),
        "tool_result_data": task.get("tool_result_data") or state.get("tool_result_data", {}),
        "execution_monitor_status": state.get("execution_monitor_status", "ok"),
        "execution_monitor_reason": state.get("execution_monitor_reason", ""),
        "recovery_action": state.get("recovery_action", "none"),
        "recovery_reason": state.get("recovery_reason", ""),
        "partial_result": state.get("partial_result", ""),
        "resume_hint": state.get("resume_hint", ""),
        "parent_evaluation": parent_evaluation,
        "goal_progress": goal_progress,
    }
    return {
        "task_tree": task_tree,
        "agenda": agenda,
        "current_task_id": "",
        "tool_name": "none",
        "tool_args": {},
        "completed_tasks": completed_records,
        "pending_memory": memory_note,
        "evaluation_decision": decision,
        "evaluation_reason": reason,
        "parent_evaluation": parent_evaluation,
        "goal_progress": goal_progress,
        "next_action": "memory",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"Evaluator：任务 {task_id} 标记为 {final_status}。"
                    f"原因：{reason} 父任务评估：{parent_evaluation.get('status')}。"
                    f"目标进度：{goal_progress.get('status')} "
                    f"{goal_progress.get('completion_ratio')}。"
                ),
            }
        ],
    }


def route_after_evaluator(state: State) -> EvaluatorRoute:
    """Evaluator 后的路由。"""

    return "commit"


def route_after_task_committer(state: State) -> TaskCommitterRoute:
    """Task Committer 后的路由。"""

    if state["done"] or state["next_action"] == "finish":
        return "finish"
    return "memory"
