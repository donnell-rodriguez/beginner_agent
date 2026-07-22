from __future__ import annotations

import pytest

from beginner_agent import router
from beginner_agent.state_factory import create_initial_state


def test_router_parses_string_false_as_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 返回字符串 false 时，不能被 Python bool("false") 误判成 True。"""

    monkeypatch.setattr(
        router,
        "chat_completion",
        lambda *args, **kwargs: (
            '{"task_type":"chat","risk_level":"low",'
            '"needs_tool":"false","reason":"普通问答","confidence":0.9}'
        ),
    )

    result = router.router_classifier_node(create_initial_state("LangGraph 是什么？"))

    assert result["task_type"] == "chat"
    assert result["needs_tool"] is False
    assert result["risk_level"] == "low"


def test_router_fallback_keeps_high_risk_for_code_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM 不可用时，本地 fallback 仍然要保留高风险判断。"""

    def fail_chat_completion(*args, **kwargs):
        raise RuntimeError("local model unavailable")

    monkeypatch.setattr(router, "chat_completion", fail_chat_completion)

    result = router.router_classifier_node(
        create_initial_state("帮我修改代码并 apply_patch 修复测试")
    )

    assert result["task_type"] == "agent"
    assert result["needs_tool"] is True
    assert result["risk_level"] == "high"
    assert result["next_action"] == "schedule"


def test_router_rejects_extra_model_fields_and_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型输出多余字段时走 fallback，避免未治理字段混进 Router 决策。"""

    monkeypatch.setattr(
        router,
        "chat_completion",
        lambda *args, **kwargs: (
            '{"task_type":"agent","risk_level":"low","needs_tool":true,'
            '"reason":"ok","unexpected":"bad"}'
        ),
    )

    result = router.router_classifier_node(create_initial_state("帮我读取 graph.py 源码"))

    assert result["task_type"] == "agent"
    assert result["risk_level"] == "low"
    assert "兜底规则" in result["route_reason"]


def test_route_by_task_guards_invalid_state() -> None:
    """即使 State 被外部写脏，条件路由也不要返回 LangGraph 未注册分支。"""

    state = create_initial_state("hello")
    state["task_type"] = "invalid"  # type: ignore[typeddict-item]

    assert router.route_by_task(state) == "chat"
