from __future__ import annotations

import json
from typing import Any

from .state import State
from .tools import ALL_TOOLS, FAILED_TOOL_RESULT_PREFIXES, project_text_files


PROJECT_TEXT_FILES = project_text_files()
PROJECT_TOOLS = ALL_TOOLS

def json_loads_from_model(text: str) -> Any:
    """解析模型返回的 JSON，兼容 Markdown 代码块。"""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def new_task(
    task_id: str,
    title: str,
    *,
    parent_id: str | None,
    depth: int,
    tool: str = "none",
    args: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """创建标准任务节点。

    中文注释：
    每个任务都放在 task_tree 里，Scheduler 只在 agenda 里保存任务 id。
    """

    return {
        "id": task_id,
        "title": title,
        "status": "pending",
        "parent_id": parent_id,
        "children": [],
        "depth": depth,
        "tool": tool,
        "args": args or {},
        "reason": reason,
        "retry_count": 0,
        "result": "",
        "tool_result_status": "none",
    }


def tool_result_status(result: str, *, blocked: bool = False) -> str:
    """把工具结果归类成 success / failed / blocked / empty / partial。

    中文注释：
    工具调用不能只看有没有返回字符串。
    真实 agent 会把结果分成更细的状态，方便 Evaluator 决定重试、失败或继续。
    """

    if blocked:
        return "blocked"
    if not result.strip():
        return "empty"
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        status = str(parsed.get("status", "")).lower()
        if status in ("failed", "timeout", "blocked", "invalid"):
            return "failed"
        if status in ("skipped", "partial"):
            return "partial"
        if status == "success":
            return "success"
    if result.startswith(FAILED_TOOL_RESULT_PREFIXES):
        return "failed"
    if "内容过长，已截断" in result:
        return "partial"
    return "success"


def goal_progress_snapshot(state: State, task_tree: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """估算当前执行结果距离用户目标还有多远。

    中文注释：
    这里先用规则做教学版 goal progress。
    大厂系统里这一步可能会再调用 LLM，判断“还缺哪些证据/步骤”。
    """

    tasks = list(task_tree.values())
    leaf_tasks = [task for task in tasks if not task.get("children")]
    done_tasks = [task for task in leaf_tasks if task.get("status") == "done"]
    failed_tasks = [
        task for task in leaf_tasks if task.get("status") in ("failed", "blocked")
    ]
    pending_tasks = [
        task for task in tasks if task.get("status") in ("pending", "planned", "approved")
    ]
    total_leaf_count = len(leaf_tasks) or 1
    completion_ratio = round(len(done_tasks) / total_leaf_count, 2)

    if pending_tasks:
        status = "in_progress"
        missing = [str(task.get("title", task.get("id"))) for task in pending_tasks[:5]]
    elif failed_tasks:
        status = "partial"
        missing = ["存在失败或阻断任务，需要人工检查。"]
    else:
        status = "complete"
        missing = []

    return {
        "status": status,
        "completion_ratio": completion_ratio,
        "done_leaf_tasks": len(done_tasks),
        "failed_leaf_tasks": len(failed_tasks),
        "pending_tasks": len(pending_tasks),
        "missing": missing,
    }


def completed_paths(state: State) -> set[str]:
    """提取已经读取过的文件路径。"""

    return {
        task.get("args", {}).get("path", "")
        for task in state["completed_tasks"]
        if task.get("tool") == "read_file"
    }


def fallback_subtasks(task: dict[str, Any], state: State) -> list[dict[str, Any]]:
    """Planner 或 Evaluator 失败时使用的保底子任务。"""

    parent_id = str(task["id"])
    depth = int(task.get("depth", 0)) + 1
    read_files = completed_paths(state)
    children: list[dict[str, Any]] = []

    if not state["completed_tasks"]:
        children.append(
            new_task(
                f"{parent_id}.1",
                "列出 beginner_agent 文件",
                parent_id=parent_id,
                depth=depth,
                tool="list_files",
                args={"path": "."},
                reason="先获得项目文件列表。",
            )
        )
        children.append(
            new_task(
                f"{parent_id}.2",
                "运行只读静态检查",
                parent_id=parent_id,
                depth=depth,
                tool="static_check",
                args={},
                reason="先确认当前 Python 文件是否存在语法错误。",
            )
        )
        children.append(
            new_task(
                f"{parent_id}.3",
                "运行真实验证测试",
                parent_id=parent_id,
                depth=depth,
                tool="run_tests",
                args={},
                reason="建立真实验证基线，后续如果修改代码可以再次验证。",
            )
        )

    preferred_files = (
        "README.md",
        "state.py",
        "graph.py",
        "router.py",
        "scheduler.py",
        "planner.py",
        "plan_validator.py",
        "policy.py",
        "executor.py",
        "evaluator.py",
        "tools.py",
    )
    for file_name in preferred_files:
        if file_name in read_files:
            continue
        index = len(children) + 1
        children.append(
            new_task(
                f"{parent_id}.{index}",
                f"读取 {file_name}",
                parent_id=parent_id,
                depth=depth,
                tool="read_file",
                args={"path": file_name},
                reason=f"通过 {file_name} 理解 agent 的模块职责。",
            )
        )
        if len(children) >= 4:
            break

    return children
