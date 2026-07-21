from __future__ import annotations

from typing import Any

from .checkpointing import checkpoint_backend_name
from .state import State


def postgres_checkpoint_node(state: State) -> dict[str, Any]:
    """Postgres Checkpoint：记录当前 LangGraph checkpoint 后端。

    中文注释：
    真正的 checkpoint 不是在这个节点里手动保存的。
    真正保存发生在 graph.py 的：

        builder.compile(checkpointer=build_checkpointer())

    这个节点的作用是把 checkpoint 后端显式放进 State，
    让后续 Observability Reporter / Summary 能看到：
    当前运行是 memory checkpoint 还是 postgres checkpoint。
    """

    backend = checkpoint_backend_name()
    report = {
        "backend": backend,
        "persistent": backend == "postgres",
        "runtime_owned": True,
        "run_id": state["run_id"],
        "state_keys_tracked_by_graph": True,
        "recovery_contract": {
            "requires_thread_id": True,
            "configured_in": "checkpointing.py",
            "runtime_entry": "builder.compile(checkpointer=build_checkpointer())",
        },
        "reason": (
            "Postgres checkpoint 可支持长任务恢复。"
            if backend == "postgres"
            else "Memory checkpoint 只适合本地单进程实验。"
        ),
    }
    return {
        "checkpoint_report": report,
        "messages": [
            {
                "role": "assistant",
                "content": f"Postgres Checkpoint：当前 checkpoint backend={backend}。",
            }
        ],
    }
