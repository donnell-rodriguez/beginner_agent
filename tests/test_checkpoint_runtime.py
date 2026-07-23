from __future__ import annotations

import pytest

from beginner_agent.checkpoint_runtime import langgraph_runtime_config, resolve_thread_id
from beginner_agent.state_factory import create_initial_state


def test_create_initial_state_uses_explicit_thread_id() -> None:
    """State.thread_id 要和 LangGraph runtime config 使用同一个值。"""

    state = create_initial_state("hello", thread_id="thread-123")
    config = langgraph_runtime_config(state["thread_id"])

    assert state["thread_id"] == "thread-123"
    assert config == {"configurable": {"thread_id": "thread-123"}}


def test_resolve_thread_id_reuses_user_supplied_value() -> None:
    """用户传入 thread_id 时，用它恢复同一条 checkpoint 任务线。"""

    assert resolve_thread_id("  existing-thread  ") == "existing-thread"


def test_runtime_config_rejects_empty_thread_id() -> None:
    """空 thread_id 不能进入 LangGraph runtime config。"""

    with pytest.raises(ValueError):
        langgraph_runtime_config("")
