from __future__ import annotations

import json
from typing import Any

from .llm_client import chat_completion
from .state import State


def search_node(state: State) -> dict[str, Any]:
    """简单 search 分支。"""

    try:
        draft = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你是资料整理节点。"
                        "当前没有联网工具，请基于已有知识做结构化整理。"
                    ),
                },
                {"role": "user", "content": state["user_input"]},
            ],
            temperature=0.2,
            max_tokens=700,
        )
    except RuntimeError as exc:
        draft = f"搜索节点调用本地模型失败：{exc}"
    return {"draft": draft, "messages": [{"role": "assistant", "content": f"Search：{draft}"}]}


def write_node(state: State) -> dict[str, Any]:
    """简单 write 分支。"""

    try:
        draft = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你是中文写作助手。"
                        "不要输出思考过程，只输出正文。"
                    ),
                },
                {"role": "user", "content": state["user_input"]},
            ],
            temperature=0.5,
            max_tokens=900,
        )
    except RuntimeError as exc:
        draft = f"写作节点调用本地模型失败：{exc}"
    return {"draft": draft, "messages": [{"role": "assistant", "content": f"Write：{draft}"}]}


def chat_node(state: State) -> dict[str, Any]:
    """简单 chat 分支。"""

    try:
        draft = chat_completion(
            [
                {"role": "system", "content": "你是耐心的中文技术学习助手。"},
                {"role": "user", "content": state["user_input"]},
            ],
            temperature=0.3,
            max_tokens=900,
        )
    except RuntimeError as exc:
        draft = f"问答节点调用本地模型失败：{exc}"
    return {"draft": draft, "messages": [{"role": "assistant", "content": f"Chat：{draft}"}]}


def simple_summarize_node(state: State) -> dict[str, Any]:
    """简单任务汇总节点。

    中文注释：
    search / write / chat 是简单分支。
    它们不需要展示 task_tree、patch_history、验证任务这些 code-agent 细节。

    所以这里只把 draft 整理成更适合用户直接阅读的最终答案。
    """

    draft = state["draft"]
    try:
        final_answer = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你是最终答案整理节点。"
                        "请把简单任务结果整理成清楚、"
                        "适合直接给用户看的中文回答。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"用户原始输入：{state['user_input']}\n\n草稿：{draft}",
                },
            ],
            temperature=0.2,
            max_tokens=900,
        )
    except RuntimeError:
        final_answer = draft

    return {
        "final_answer": final_answer,
        "messages": [{"role": "assistant", "content": f"Simple Summarize：{final_answer}"}],
    }


def _task_title(task: dict[str, Any]) -> str:
    """安全读取任务标题。"""

    return str(task.get("title") or task.get("id") or "未命名任务")


def _completed_lines(state: State) -> list[str]:
    """提取“完成了什么”。

    中文注释：
    completed_tasks 是 Task Committer 记录的完成任务。
    如果它为空，就从 task_tree 里找 done / completed / pending_verification 等状态。
    """

    completed_tasks = list(state.get("completed_tasks", []))
    if completed_tasks:
        return [
            f"- {task.get('id', '')}：{_task_title(task)}"
            for task in completed_tasks
            if isinstance(task, dict)
        ]

    task_tree = dict(state.get("task_tree", {}))
    done_statuses = {"done", "completed", "complete", "pending_verification"}
    lines = []
    for task_id, task in task_tree.items():
        if str(task.get("status", "")) in done_statuses:
            lines.append(f"- {task_id}：{_task_title(task)}（{task.get('status')}）")
    return lines or ["- 暂无明确完成记录。"]


def _unfinished_lines(state: State) -> list[str]:
    """提取“未完成什么”。"""

    task_tree = dict(state.get("task_tree", {}))
    unfinished_statuses = {
        "pending",
        "planned",
        "waiting_approval",
        "blocked",
        "failed",
        "pending_verification",
    }
    lines = []
    for task_id, task in task_tree.items():
        status = str(task.get("status", ""))
        if status in unfinished_statuses:
            reason = str(task.get("result") or task.get("evaluation_reason") or "")
            suffix = f"，原因：{reason}" if reason else ""
            lines.append(f"- {task_id}：{_task_title(task)}（{status}{suffix}）")
    return lines or ["- 暂无明确未完成项。"]


def _changed_file_lines(state: State) -> list[str]:
    """提取“修改了哪些文件”。"""

    files: list[str] = []
    for patch in state.get("patch_history", []):
        if isinstance(patch, dict):
            path = str(patch.get("path") or patch.get("file_path") or "")
            if path and path not in files:
                files.append(path)

    task_tree = dict(state.get("task_tree", {}))
    for task in task_tree.values():
        if not isinstance(task, dict):
            continue
        data = task.get("tool_result_data", {})
        if isinstance(data, dict):
            path = str(data.get("path") or data.get("file_path") or "")
            changed_files = data.get("changed_files", [])
            if path and path not in files:
                files.append(path)
            if isinstance(changed_files, list):
                for item in changed_files:
                    item_path = str(item)
                    if item_path and item_path not in files:
                        files.append(item_path)

    return [f"- {path}" for path in files] or ["- 本次没有记录到文件修改。"]


def _verification_lines(state: State) -> list[str]:
    """提取“验证结果”。"""

    task_tree = dict(state.get("task_tree", {}))
    verification_tools = {
        "run_tests",
        "run_targeted_tests",
        "run_impacted_tests",
        "static_check",
        "lint_typecheck",
        "run_typecheck",
        "run_build",
        "git_diff",
        "git_diff_file",
        "secret_scan",
    }
    lines = []
    for task_id, task in task_tree.items():
        tool = str(task.get("tool", ""))
        if tool in verification_tools or "验证" in _task_title(task):
            status = str(task.get("tool_result_status") or task.get("status") or "unknown")
            result = str(task.get("result", "")).strip()
            preview = result[:180].replace("\n", " ")
            suffix = f"：{preview}" if preview else ""
            lines.append(f"- {task_id} / {tool}：{status}{suffix}")

    if not lines and state.get("tool_result_status") != "none":
        lines.append(
            f"- 最近一次工具结果："
            f"{state.get('tool_name')} -> {state.get('tool_result_status')}"
        )
    return lines or ["- 暂无验证任务记录。"]


def _recovery_lines(state: State) -> list[str]:
    """提取“恢复建议”。"""

    lines = []
    recovery_action = str(state.get("recovery_action", "none"))
    recovery_reason = str(state.get("recovery_reason", ""))
    resume_hint = str(state.get("resume_hint", ""))
    partial_result = str(state.get("partial_result", ""))

    if recovery_action != "none":
        lines.append(f"- 恢复动作：{recovery_action}。{recovery_reason}")
    if resume_hint:
        lines.append(f"- 下次继续建议：{resume_hint}")
    if partial_result:
        lines.append(f"- 当前可复用的部分结果：{partial_result[:240]}")
    return lines or ["- 暂无恢复建议。"]


def _risk_lines(state: State) -> list[str]:
    """提取“风险提示”。"""

    lines = [
        f"- Router 风险等级：{state.get('risk_level', 'unknown')}",
        (
            f"- Tool Policy 决策：{state.get('policy_decision', 'unknown')}。"
            f"{state.get('policy_reason', '')}"
        ),
    ]
    pending_approval = state.get("pending_approval", {})
    if isinstance(pending_approval, dict) and pending_approval:
        lines.append(
            f"- 仍有待审批操作：{pending_approval.get('tool_name')} "
            f"({pending_approval.get('approval_id')})"
        )
    if state.get("patch_history"):
        lines.append(
            "- 本次涉及代码修改记录；"
            "如验证失败，应优先查看 patch_history 和 git_diff。"
        )
    return lines


def code_agent_summarize_node(state: State) -> dict[str, Any]:
    """复杂 code-agent 汇总节点。

    中文注释：
    复杂 agent 的最终输出不能只是一段“好像完成了”的自然语言。
    它必须明确告诉用户：
    - 完成了什么。
    - 未完成什么。
    - 修改了哪些文件。
    - 验证结果。
    - 恢复建议。
    - 风险提示。

    这更接近生产级 code agent 的交付报告。
    """

    sections = [
        "# Code Agent Summary",
        "",
        f"用户目标：{state['user_input']}",
        "",
        "## 完成了什么",
        *_completed_lines(state),
        "",
        "## 未完成什么",
        *_unfinished_lines(state),
        "",
        "## 修改了哪些文件",
        *_changed_file_lines(state),
        "",
        "## 验证结果",
        *_verification_lines(state),
        "",
        "## 恢复建议",
        *_recovery_lines(state),
        "",
        "## 风险提示",
        *_risk_lines(state),
        "",
        "## 结构化状态摘要",
        "```json",
        json.dumps(
            {
                "task_type": state["task_type"],
                "risk_level": state["risk_level"],
                "done": state["done"],
                "step_count": state["step_count"],
                "goal_progress": state["goal_progress"],
                "task_count": len(state["task_tree"]),
                "completed_count": len(state["completed_tasks"]),
                "tool_result_status": state["tool_result_status"],
                "execution_monitor_status": state["execution_monitor_status"],
                "recovery_action": state["recovery_action"],
                "checkpoint_report": state.get("checkpoint_report", {}),
                "sandbox_report": state.get("sandbox_report", {}),
                "async_job_report": state.get("async_job_report", {}),
                "artifact_report": state.get("artifact_report", {}),
                "observability_report": state.get("observability_report", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    final_answer = "\n".join(sections)
    return {
        "final_answer": final_answer,
        "messages": [{"role": "assistant", "content": f"Code Agent Summarize：{final_answer}"}],
    }
