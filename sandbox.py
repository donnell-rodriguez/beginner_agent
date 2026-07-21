from __future__ import annotations

from typing import Any

from .state import State
from .tooling.core import active_project_root
from .tools import WRITE_TOOLS


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
    report = {
        "mode": "local_controlled",
        "active_project_root": active_project_root().as_posix(),
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
        "future_upgrade": "container_or_remote_sandbox",
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
