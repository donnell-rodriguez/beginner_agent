from __future__ import annotations

import uuid
from typing import Any


def make_thread_id(prefix: str = "beginner-agent-run") -> str:
    """生成新的 LangGraph checkpoint thread_id。

    中文注释：
    thread_id 是 LangGraph runtime 用来区分“哪一次会话/哪一条任务线”的 id。
    只把 thread_id 放进 State 不够，真正恢复时还必须传给 graph：

        config = {"configurable": {"thread_id": "..."}}

    所以这里把 thread_id 的生成集中起来，避免 main.py / cli.py 各写各的。
    """

    return f"{prefix}-{uuid.uuid4()}"


def resolve_thread_id(
    explicit_thread_id: str | None,
    *,
    fallback_prefix: str = "beginner-agent-run",
) -> str:
    """解析本次运行应该使用哪个 thread_id。

    中文注释：
    - 用户传了 --thread-id：说明他想恢复或继续同一条任务线，优先使用。
    - 用户没传：生成一个新的 thread_id。
    """

    if explicit_thread_id and explicit_thread_id.strip():
        return explicit_thread_id.strip()
    return make_thread_id(fallback_prefix)


def langgraph_runtime_config(thread_id: str) -> dict[str, Any]:
    """生成 LangGraph invoke/stream/get_state 使用的 runtime config。

    中文注释：
    这是 checkpoint 恢复闭环最关键的对象。

    同一个任务从开始、interrupt、approval resume、get_state 到后续恢复，
    都必须使用同一个：

        {"configurable": {"thread_id": thread_id}}

    否则 State 里的 thread_id 看起来对了，
    但 LangGraph runtime 实际会写入/读取另一条 checkpoint 线程。
    """

    resolved = thread_id.strip()
    if not resolved:
        raise ValueError("LangGraph runtime config 需要非空 thread_id。")
    return {"configurable": {"thread_id": resolved}}
