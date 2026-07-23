from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command  # noqa: E402

from beginner_agent.approval_server import run_approval_server  # noqa: E402
from beginner_agent.approval_store import (  # noqa: E402
    ApprovalStore,
    default_approver_id,
    default_timeout_seconds,
)
from beginner_agent.checkpoint_runtime import (  # noqa: E402
    langgraph_runtime_config,
    resolve_thread_id,
)
from beginner_agent.graph import build_graph  # noqa: E402
from beginner_agent.serialization import serialize_for_json  # noqa: E402
from beginner_agent.state_factory import create_initial_state  # noqa: E402


def _extract_interrupt(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """从 LangGraph stream chunk 中取出 interrupt payload。

    中文注释：
    当 approval_interrupt_node 调用 interrupt(...) 时，
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


def _read_modified_tool_args(payload: dict[str, Any]) -> dict[str, Any]:
    """让审批人可选地修改工具参数。

    中文注释：
    生产级审批不是只有 approve / deny。
    有时工具方向是对的，但参数需要收窄，比如：

        {"path": "."} 改成 {"path": "graph.py"}

    这里用 JSON 输入，保持 CLI 简单但能力完整。
    """

    answer = (
        input("是否修改工具参数？输入 e 编辑，直接回车表示不修改：")
        .strip()
        .lower()
    )
    if answer != "e":
        return {}

    current_args = payload.get("tool_args", {})
    print("\n当前工具参数：")
    print(json.dumps(current_args, ensure_ascii=False, indent=2))
    raw = input("\n请输入新的 JSON 工具参数：").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("新的工具参数必须是 JSON object。")
    return parsed


def _ask_human(payload: dict[str, Any], *, approver_id: str) -> dict[str, Any]:
    """CLI 人工审批入口。

    中文注释：
    当前是本地可用 CLI：
    - y / yes / approve：批准，图恢复后继续执行工具。
    - 其他输入：拒绝，图恢复后进入 Evaluator，任务会被标记 blocked。
    - 批准时可以修改 tool_args，然后再恢复 graph。
    """

    _print_approval_request(payload)
    answer = (
        input("\n是否批准执行这个工具？输入 y 批准，其他任意输入拒绝：")
        .strip()
        .lower()
    )
    approved = answer in {"y", "yes", "approve", "approved"}
    modified_tool_args = _read_modified_tool_args(payload) if approved else {}
    return {
        "approved": approved,
        "approval_id": payload.get("approval_id", ""),
        "task_id": payload.get("task_id", ""),
        "approver_id": approver_id,
        "modified_tool_args": modified_tool_args,
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


def _approval_resume_value(
    payload: dict[str, Any],
    *,
    store: ApprovalStore,
    thread_id: str,
    approval_mode: str,
    approver_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """保存审批请求，并返回 Command(resume=...) 所需的审批结果。"""

    record = store.create_or_update_request(
        payload,
        thread_id=thread_id,
        timeout_seconds=timeout_seconds,
    )
    payload = {**payload, "approval_id": record.approval_id}

    if approval_mode == "api":
        print("\n审批请求已经保存，等待 API / Web UI 写入审批结果。")
        print(f"审批 id：{record.approval_id}")
        print(f"超时时间：{timeout_seconds} 秒")
        decided = store.wait_for_decision(
            record.approval_id,
            timeout_seconds=timeout_seconds,
        )
        return decided.to_resume_value()

    decision = _ask_human(payload, approver_id=approver_id)
    decided = store.decide(
        record.approval_id,
        approved=bool(decision["approved"]),
        approver_id=approver_id,
        reason=str(decision["reason"]),
        modified_tool_args=dict(decision.get("modified_tool_args") or {}),
    )
    return decided.to_resume_value()


def run_cli(
    user_input: str,
    *,
    thread_id: str | None = None,
    approval_mode: str = "cli",
    approver_id: str | None = None,
    timeout_seconds: int | None = None,
) -> Any:
    """运行支持 Approval Interrupt 的 CLI。"""

    graph = build_graph()
    resolved_thread_id = resolve_thread_id(thread_id, fallback_prefix="beginner-agent-cli")
    config = langgraph_runtime_config(resolved_thread_id)
    graph_input: Any = create_initial_state(user_input, thread_id=resolved_thread_id)
    store = ApprovalStore()
    resolved_approver = approver_id or default_approver_id()
    resolved_timeout = timeout_seconds or default_timeout_seconds()

    while True:
        outcome = _stream_until_interrupt_or_done(graph, graph_input, config)
        if not outcome["interrupted"]:
            return outcome["result"]

        approval_result = _approval_resume_value(
            outcome["payload"],
            store=store,
            thread_id=resolved_thread_id,
            approval_mode=approval_mode,
            approver_id=resolved_approver,
            timeout_seconds=resolved_timeout,
        )
        graph_input = Command(resume=approval_result)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """添加运行 agent 所需参数。"""

    parser.add_argument(
        "user_input",
        nargs="*",
        help="用户任务。为空时进入交互输入。",
    )
    parser.add_argument("--thread-id", default=None, help="LangGraph checkpoint thread_id。")
    parser.add_argument(
        "--approval-mode",
        choices=["cli", "api"],
        default="cli",
        help="cli 表示本地询问；api 表示等待审批 API 写入结果。",
    )
    parser.add_argument("--approver-id", default=None, help="审批人身份。")
    parser.add_argument(
        "--approval-timeout-seconds",
        type=int,
        default=None,
        help="审批超时时间，超时后自动拒绝。",
    )


def _print_approval_records(records: list[Any]) -> None:
    """打印审批列表。"""

    print(
        json.dumps(
            [
                {
                    "approval_id": record.approval_id,
                    "task_id": record.task_id,
                    "status": record.status,
                    "approver_id": record.approver_id,
                    "expires_at": record.expires_at,
                    "reason": record.reason,
                }
                for record in records
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Run beginner_agent.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="运行 agent。")
    _add_run_args(run_parser)

    serve_parser = subparsers.add_parser("serve-approvals", help="启动本地审批 API。")
    serve_parser.add_argument("--host", default=None, help="审批 API host。")
    serve_parser.add_argument("--port", type=int, default=None, help="审批 API port。")

    list_parser = subparsers.add_parser("list-approvals", help="列出审批请求。")
    list_parser.add_argument(
        "--status",
        default=None,
        help="过滤 pending/approved/denied/expired。",
    )
    list_parser.add_argument("--limit", type=int, default=20, help="最多返回多少条。")

    # 中文注释：
    # 为了兼容旧用法：
    #
    #     python main.py "帮我看 graph.py"
    #
    # 如果第一个参数不是已知子命令，就当成 run 的 user_input。
    known_commands = {"run", "serve-approvals", "list-approvals"}
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands:
        args = parser.parse_args(["run", *sys.argv[1:]])
    else:
        args = parser.parse_args()

    if args.command == "serve-approvals":
        run_approval_server(host=args.host, port=args.port)
        return

    if args.command == "list-approvals":
        _print_approval_records(ApprovalStore().list(status=args.status, limit=args.limit))
        return

    if args.command is None:
        parser.print_help()
        return

    user_input = " ".join(args.user_input).strip()
    if not user_input:
        user_input = input("请输入任务：").strip()

    result = run_cli(
        user_input,
        thread_id=args.thread_id,
        approval_mode=args.approval_mode,
        approver_id=args.approver_id,
        timeout_seconds=args.approval_timeout_seconds,
    )
    print("\n最终输出：")
    print(json.dumps(serialize_for_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
