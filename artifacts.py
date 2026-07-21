from __future__ import annotations

from typing import Any

from .artifact_store import write_artifact_manifest
from .state import State


def _unique(items: list[str]) -> list[str]:
    """保持顺序去重。"""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _changed_files(state: State) -> list[str]:
    """从 patch_history 和 ToolResult 中提取改动文件。"""

    files: list[str] = []
    for patch in state.get("patch_history", []):
        if isinstance(patch, dict):
            files.append(str(patch.get("path") or patch.get("file_path") or ""))

    for task in state.get("task_tree", {}).values():
        if not isinstance(task, dict):
            continue
        data = task.get("tool_result_data", {})
        if not isinstance(data, dict):
            continue
        files.append(str(data.get("path") or data.get("file_path") or ""))
        changed = data.get("changed_files", [])
        if isinstance(changed, list):
            files.extend(str(item) for item in changed)
    return _unique(files)


def artifact_collector_node(state: State) -> dict[str, Any]:
    """Artifact Collector：收集本轮 agent 产生的交付物。

    中文注释：
    大厂 code agent 不只输出一段文字。
    它还要知道自己产生了哪些 artifact：
    - 修改过哪些文件。
    - 产生了哪些 patch。
    - 有哪些验证记录。
    - 有哪些可审计的执行尝试。

    这个节点不做文件写入，只从 State 中汇总 artifact 索引。
    """

    changed_files = _changed_files(state)
    verification_task_ids = [
        task_id
        for task_id, task in state.get("task_tree", {}).items()
        if isinstance(task, dict)
        and str(task.get("tool", ""))
        in {
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
    ]
    report: dict[str, Any] = {
        "changed_files": changed_files,
        "patch_count": len(state.get("patch_history", [])),
        "execution_attempt_count": len(state.get("execution_attempts", [])),
        "verification_task_ids": verification_task_ids,
        "completed_task_count": len(state.get("completed_tasks", [])),
        "memory_note_count": len(state.get("memory_notes", [])),
    }
    storage = write_artifact_manifest(
        run_id=state["run_id"],
        report=report,
        state_snapshot={
            "task_type": state["task_type"],
            "risk_level": state["risk_level"],
            "done": state["done"],
            "step_count": state["step_count"],
            "current_task_id": state["current_task_id"],
            "tool_name": state["tool_name"],
            "tool_result_status": state["tool_result_status"],
            "execution_monitor_status": state["execution_monitor_status"],
            "recovery_action": state["recovery_action"],
        },
    )
    report["storage"] = storage
    return {
        "artifact_report": report,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Artifact Collector：收集到 "
                    f"{len(changed_files)} 个改动文件、"
                    f"{len(verification_task_ids)} 个验证任务。"
                ),
            }
        ],
    }
