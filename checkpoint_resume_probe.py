from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from .checkpoint_runtime import langgraph_runtime_config
from .checkpointing import build_checkpointer, checkpoint_backend_name


# 中文注释：
# checkpoint_resume_probe.py 专门验证 checkpoint 的最终价值：
#
#   第一次运行
#     -> 图执行到 interrupt
#     -> LangGraph 把中间状态写入 checkpoint
#
#   第二次运行
#     -> 重新 build graph
#     -> 用同一个 thread_id
#     -> Command(resume=...)
#     -> 从 checkpoint 继续执行剩余节点
#
# 这个 probe 不依赖业务 Router / Planner / Tool Policy。
# 这样可以把“checkpoint 能不能恢复”这个基础设施能力单独测清楚。


class CheckpointResumeProbeState(TypedDict):
    """最小 LangGraph 状态。

    中文注释：
    这里故意不用 beginner_agent 的完整 State，
    因为完整业务图会牵涉 LLM、工具、审批策略。
    checkpoint 恢复测试只需要证明：
    interrupt 前写入的 state，resume 后还能继续用。
    """

    marker: str
    approved: bool
    final: str


class CheckpointResumeProbeResult(BaseModel):
    """Checkpoint resume 闭环测试结果。"""

    model_config = ConfigDict(extra="forbid")

    backend: str
    thread_id: str
    interrupted: bool
    interrupt_payload: dict[str, Any] = Field(default_factory=dict)
    state_before_resume: dict[str, Any] = Field(default_factory=dict)
    state_after_resume: dict[str, Any] = Field(default_factory=dict)
    resumed: bool


def run_checkpoint_resume_probe(thread_id: str | None = None) -> CheckpointResumeProbeResult:
    """运行一次 checkpoint interrupt/resume 闭环测试。

    中文注释：
    这个函数会创建两个 graph 实例：

    - graph_before：第一次运行，触发 interrupt。
    - graph_after：模拟进程重启后重新 build graph，再用同一个 thread_id 恢复。

    如果第二个 graph 能继续执行到 final="resumed"，说明 checkpoint 恢复闭环成立。
    """

    resolved_thread_id = thread_id or f"checkpoint-resume-probe-{uuid.uuid4()}"
    config = langgraph_runtime_config(resolved_thread_id)

    graph_before = _build_resume_probe_graph()
    first_chunks = list(
        graph_before.stream(
            {
                "marker": "",
                "approved": False,
                "final": "",
            },
            config=config,
        )
    )
    interrupt_payload = _extract_interrupt_payload(first_chunks)
    state_before_resume = dict(graph_before.get_state(config).values)

    graph_after = _build_resume_probe_graph()
    list(
        graph_after.stream(
            Command(resume={"approved": True, "reason": "resume probe approved"}),
            config=config,
        )
    )
    state_after_resume = dict(graph_after.get_state(config).values)

    return CheckpointResumeProbeResult(
        backend=checkpoint_backend_name(),
        thread_id=resolved_thread_id,
        interrupted=bool(interrupt_payload),
        interrupt_payload=interrupt_payload,
        state_before_resume=state_before_resume,
        state_after_resume=state_after_resume,
        resumed=state_after_resume.get("final") == "resumed",
    )


def _build_resume_probe_graph():
    """构造最小 interrupt/resume probe graph。"""

    builder = StateGraph(CheckpointResumeProbeState)
    builder.add_node("start", _probe_start_node)
    builder.add_node("approval", _probe_interrupt_node)
    builder.add_node("finish", _probe_finish_node)
    builder.add_edge(START, "start")
    builder.add_edge("start", "approval")
    builder.add_edge("approval", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=build_checkpointer())


def _probe_start_node(state: CheckpointResumeProbeState) -> dict[str, str]:
    """写入 interrupt 前的状态，用来确认 resume 后没有丢。"""

    return {"marker": "started-before-interrupt"}


def _probe_interrupt_node(state: CheckpointResumeProbeState) -> dict[str, Any]:
    """触发 LangGraph interrupt，并在 resume 后读取恢复值。"""

    resume_value = interrupt(
        {
            "probe": "checkpoint_resume",
            "marker": state["marker"],
            "question": "Approve checkpoint resume probe?",
        }
    )
    approved = bool(resume_value.get("approved")) if isinstance(resume_value, dict) else bool(resume_value)
    return {"approved": approved}


def _probe_finish_node(state: CheckpointResumeProbeState) -> dict[str, str]:
    """resume 后执行的节点。"""

    return {"final": "resumed" if state["approved"] else "denied"}


def _extract_interrupt_payload(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    for chunk in chunks:
        interrupts = chunk.get("__interrupt__")
        if not interrupts:
            continue
        interrupt_obj = interrupts[0]
        value = getattr(interrupt_obj, "value", interrupt_obj)
        return value if isinstance(value, dict) else {"value": value}
    return {}
