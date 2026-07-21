from __future__ import annotations

import os
from typing import Any, Literal

from .config import load_project_env
from .state import State
from .tooling.core import active_project_root
from .tools import WRITE_TOOLS


load_project_env()

SandboxRoute = Literal["execute", "evaluate"]


def sandbox_mode() -> str:
    """读取 sandbox backend。

    中文注释：
    local_controlled 是当前可运行模式。
    docker / firecracker / remote_worker 是后续生产级隔离模式，
    当前先作为明确 contract 暴露出来，不假装已经实现。
    """

    return os.getenv("BEGINNER_AGENT_SANDBOX_MODE", "local_controlled").strip()


def sandbox_runner_node(state: State) -> dict[str, Any]:
    """Sandbox Runner：在 Executor 前准备受控运行边界。

    中文注释：
    当前项目还没有真正的容器 sandbox 或远程 worker。
    但在大厂 code agent 里，Executor 前通常会有一层 sandbox decision：

        工具调用
          -> 审批 / 权限
          -> Sandbox Runner
          -> Executor

    这个节点先把“运行边界”显式记录到 State：
    - 当前 active project。
    - 是否写工具。
    - 是否已经通过 policy。
    - 使用本地受控工具层，而不是任意 shell。

    后续可以把这里替换成 Docker sandbox、Firecracker、远程 runner。
    """

    tool_name = state["tool_name"]
    is_write_tool = tool_name in set(WRITE_TOOLS)
    mode = sandbox_mode()
    supported = mode == "local_controlled"
    report = {
        "mode": mode,
        "status": "ready" if supported else "not_configured",
        "active_project_root": active_project_root().as_posix(),
        "run_id": state["run_id"],
        "task_id": state["current_task_id"],
        "tool_name": tool_name,
        "is_write_tool": is_write_tool,
        "policy_decision": state["policy_decision"],
        "approved": state["policy_decision"] == "allow",
        "isolation": [
            "ToolSpec registry",
            "Pydantic args schema",
            "safe path resolver",
            "allowlisted command profiles",
            "patch plan governance for writes",
        ],
        "future_upgrade": (
            "查看 sandbox_backends/TODO.md，接入 Docker / Firecracker / remote sandbox。"
        ),
    }
    if not supported:
        report["warning"] = (
            f"Sandbox backend {mode} 尚未实现，当前仍不会直接执行任意 shell。"
        )
        return {
            "sandbox_report": report,
            "tool_result": report["warning"],
            "tool_result_status": "blocked",
            "execution_status": "blocked",
            "next_action": "evaluate",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Sandbox Runner：{report['warning']}",
                }
            ],
        }
    return {
        "sandbox_report": report,
        "next_action": "execute",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"Sandbox Runner：任务 {state['current_task_id']} "
                    "使用本地受控 sandbox。"
                ),
            }
        ],
    }


def route_after_sandbox_runner(state: State) -> SandboxRoute:
    """Sandbox Runner 后的路由。"""

    if state["next_action"] == "execute":
        return "execute"
    return "evaluate"
