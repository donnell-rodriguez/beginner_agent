from __future__ import annotations

import argparse
import json
from typing import Any

from beginner_agent.router import router_classifier_node
from beginner_agent.routering.eval_runner import (
    append_router_eval_trend,
    append_router_feedback,
    load_router_eval_dataset,
    make_feedback_record,
    read_router_eval_trends,
    run_router_eval,
)
from beginner_agent.routering.models import RouterDecision
from beginner_agent.state_factory import create_initial_state


# 中文注释：
# 这是 Router eval 的 CLI。
#
# 常用命令：
#
#   PYTHONPATH=.. uv run python scripts/router_eval.py replay
#   PYTHONPATH=.. uv run python scripts/router_eval.py replay --dataset router_eval.json
#   PYTHONPATH=.. uv run python scripts/router_eval.py feedback --user-input "帮我修复测试" \
#       --task-type agent --risk-level high --needs-tool true --reason "代码修改必须进 agent"
#   PYTHONPATH=.. uv run python scripts/router_eval.py trends --limit 10
#
# 它不放在 router.py 里，是为了让业务节点保持干净。


def _predict_with_router(user_input: str) -> RouterDecision:
    result = router_classifier_node(create_initial_state(user_input))
    return RouterDecision(
        task_type=result["task_type"],
        risk_level=result["risk_level"],
        needs_tool=bool(result["needs_tool"]),
        reason=str(result.get("route_reason", "")),
        confidence=float(
            result.get("router_report", {}).get("decision", {}).get("confidence", 0.7)
        ),
    )


def _cmd_replay(args: argparse.Namespace) -> int:
    dataset = load_router_eval_dataset(args.dataset)
    run = run_router_eval(
        dataset,
        _predict_with_router,
        router_version=args.router_version,
        max_failures=args.max_failures,
    )
    append_router_eval_trend(run)
    print(json.dumps(run.as_dict(), ensure_ascii=False, indent=2))
    return 0 if run.failed == 0 else 1


def _cmd_feedback(args: argparse.Namespace) -> int:
    record = make_feedback_record(
        user_input=args.user_input,
        expected_task_type=args.task_type,
        expected_risk_level=args.risk_level,
        expected_needs_tool=_parse_bool(args.needs_tool),
        reason=args.reason,
        source=args.source,
    )
    case = append_router_feedback(record)
    print(json.dumps(case.as_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_trends(args: argparse.Namespace) -> int:
    records = read_router_eval_trends(args.limit)
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Router eval replay / feedback / trends CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="批量回放 Router eval dataset。")
    replay.add_argument("--dataset", default=None, help="JSON/JSONL dataset 路径。")
    replay.add_argument("--router-version", default=None, help="本次 Router 版本标识。")
    replay.add_argument("--max-failures", type=int, default=50, help="最多保留多少失败样本。")
    replay.set_defaults(func=_cmd_replay)

    feedback = subparsers.add_parser("feedback", help="把线上纠错反馈写成 eval case。")
    feedback.add_argument("--user-input", required=True)
    feedback.add_argument(
        "--task-type",
        required=True,
        choices=["search", "write", "chat", "agent"],
    )
    feedback.add_argument("--risk-level", required=True, choices=["low", "medium", "high"])
    feedback.add_argument("--needs-tool", required=True)
    feedback.add_argument("--reason", required=True)
    feedback.add_argument("--source", default="manual_feedback")
    feedback.set_defaults(func=_cmd_feedback)

    trends = subparsers.add_parser("trends", help="查看 Router eval 趋势。")
    trends.add_argument("--limit", type=int, default=None)
    trends.set_defaults(func=_cmd_trends)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
