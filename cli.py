from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command  # noqa: E402

from beginner_agent.graph import build_graph  # noqa: E402
from beginner_agent.serialization import serialize_for_json  # noqa: E402
from beginner_agent.state_factory import create_initial_state  # noqa: E402


def _extract_interrupt(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """从 LangGraph stream chunk 中取出 interrupt payload。

    中文注释：
    当 human_approval_node 调用 interrupt(...) 时，
    graph.stream(...) 会吐出类似：

        {"__interrupt__": (Interrupt(value={...}),)}

    CLI 不应该关心图内部节点怎么实现审批，
    它只关心有没有 "__interrupt__"。
    """

    interrupts = chunk.get("__interrupt__")
    if not interrupts:
        return None
    interrupt_obj = interrupts[0]
    value = getattr(interrupt_obj, "value", interrupt_obj)
    return value if isinstance(value, dict) else {"message": str(value)}


def _print_approval_request(payload: dict[str, Any]) -> None:
    """把审批请求打印给用户看。"""

    print("\n需要人工审批")
    print("-" * 40)
    print(f"审批 id：{payload.get('approval_id', '')}")
    print(f"任务 id：{payload.get('task_id', '')}")
    print(f"工具名：{payload.get('tool_name', '')}")
    print(f"风险等级：{payload.get('risk_level', '')}")
    print(f"原因：{payload.get('reason', '')}")
    print("\n工具参数：")
    print(json.dumps(payload.get("tool_args", {}), ensure_ascii=False, indent=2))


def _ask_human(payload: dict[str, Any]) -> dict[str, Any]:
    """CLI 人工审批入口。

    中文注释：
    当前是最小可用 CLI：
    - y / yes / approve：批准，图恢复后继续执行工具。
    - 其他输入：拒绝，图恢复后进入 Evaluator，任务会被标记 blocked。

    后续可以把这里替换成：
    - Web UI。
    - Slack / 飞书审批。
    - 更细粒度的“修改参数后批准”。
    """

    _print_approval_request(payload)
    answer = input("\n是否批准执行这个工具？输入 y 批准，其他任意输入拒绝：").strip().lower()
    approved = answer in {"y", "yes", "approve", "approved"}
    return {
        "approved": approved,
        "approval_id": payload.get("approval_id", ""),
        "task_id": payload.get("task_id", ""),
        "reason": "CLI 用户批准。" if approved else "CLI 用户拒绝。",
    }


def _stream_until_interrupt_or_done(graph: Any, graph_input: Any, config: dict[str, Any]) -> Any:
    """运行 graph，遇到 interrupt 就返回审批 payload，跑完就返回完整 State。"""

    for chunk in graph.stream(graph_input, config=config):
        payload = _extract_interrupt(chunk)
        if payload is not None:
            return {"interrupted": True, "payload": payload}

    # 中文注释：
    # graph.stream(...) 每次吐出来的是“某个节点刚返回的局部更新”，
    # 不一定是完整 State。
    #
    # graph.get_state(config).values 才是 checkpoint 里当前 thread 的完整 State。
    return {"interrupted": False, "result": graph.get_state(config).values}


def run_cli(user_input: str, *, thread_id: str | None = None) -> Any:
    """运行支持 Human Approval interrupt 的 CLI。"""

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or f"beginner-agent-cli-{uuid.uuid4()}"}}
    graph_input: Any = create_initial_state(user_input)

    while True:
        outcome = _stream_until_interrupt_or_done(graph, graph_input, config)
        if not outcome["interrupted"]:
            return outcome["result"]

        approval_result = _ask_human(outcome["payload"])
        graph_input = Command(resume=approval_result)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Run beginner_agent with CLI human approval.")
    parser.add_argument("user_input", nargs="*", help="用户任务。为空时进入交互输入。")
    parser.add_argument("--thread-id", default=None, help="LangGraph checkpoint thread_id。")
    args = parser.parse_args()

    user_input = " ".join(args.user_input).strip()
    if not user_input:
        user_input = input("请输入任务：").strip()

    result = run_cli(user_input, thread_id=args.thread_id)
    print("\n最终输出：")
    print(json.dumps(serialize_for_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
