from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from beginner_agent.routering.nodes import router_classifier_node
from beginner_agent.routering.eval_runner import (
    append_router_eval_trend,
    append_router_feedback,
    load_router_eval_dataset,
    make_feedback_record,
    read_router_eval_trends,
    run_router_eval,
)
from beginner_agent.routering.feedback import read_router_feedback, record_router_correction
from beginner_agent.routering.models import RouterDecision
from beginner_agent.routering.regression_gate import evaluate_router_regression_gate
from beginner_agent.routering.regression_gate import evaluate_router_release_gate
from beginner_agent.routering.regression_gate import load_router_eval_baseline
from beginner_agent.routering.regression_gate import write_router_eval_baseline
from beginner_agent.state_factory import create_initial_state


# 中文注释：
# 这是 Router eval 的 CLI。
#
# 常用命令：
#
#   PYTHONPATH=.. uv run python scripts/router_eval.py replay
#   PYTHONPATH=.. uv run python scripts/router_eval.py replay --dataset router_eval.json
#   PYTHONPATH=.. uv run python scripts/router_eval.py gate --dataset router_eval.json
#   PYTHONPATH=.. uv run python scripts/router_eval.py write-baseline --dataset router_eval.json
#   PYTHONPATH=.. uv run python scripts/router_eval.py feedback --user-input "帮我修复测试" \
#       --task-type agent --risk-level high --needs-tool true --reason "代码修改必须进 agent"
#   PYTHONPATH=.. uv run python scripts/router_eval.py feedback --router-report router_report.json \
#       --task-type agent --risk-level high --needs-tool true --reason "这次应该进入 code agent"
#   PYTHONPATH=.. uv run python scripts/router_eval.py trends --limit 10
#
# 它不放在 routering/nodes.py 里，是为了让业务节点保持干净。


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
    gate = evaluate_router_regression_gate(run)
    payload = run.as_dict()
    payload["regression_gate"] = gate.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if gate.passed else 1


def _cmd_gate(args: argparse.Namespace) -> int:
    """运行 Router 上线门禁。

    中文注释：
    这个命令适合放到 CI 或 pre-push hook 里：

        改了 prompt / rules / security
          -> replay eval dataset
          -> 对比绝对阈值和 baseline 下降幅度
          -> 不通过就返回退出码 1
    """

    dataset = load_router_eval_dataset(args.dataset)
    run = run_router_eval(
        dataset,
        _predict_with_router,
        router_version=args.router_version,
        max_failures=args.max_failures,
    )
    baseline = load_router_eval_baseline(args.baseline)
    gate = evaluate_router_release_gate(run, baseline=baseline)
    if args.record_trend:
        append_router_eval_trend(run)
    payload = run.as_dict()
    payload["release_gate"] = gate.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if gate.passed else 1


def _cmd_write_baseline(args: argparse.Namespace) -> int:
    """把当前 Router eval run 写成后续上线门禁 baseline。"""

    dataset = load_router_eval_dataset(args.dataset)
    run = run_router_eval(
        dataset,
        _predict_with_router,
        router_version=args.router_version,
        max_failures=args.max_failures,
    )
    gate = evaluate_router_release_gate(run)
    if not gate.passed and not args.force:
        payload = run.as_dict()
        payload["release_gate"] = gate.as_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    path = write_router_eval_baseline(run, path=args.output)
    payload = {
        "baseline_path": str(path),
        "run": run.as_dict(),
        "release_gate": gate.as_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_feedback(args: argparse.Namespace) -> int:
    # 中文注释：
    # 如果传入 router_report 或 decision_id，就走新的反馈闭环：
    #
    #   真实 RouterEvent / router_report
    #     -> RouterFeedbackEvent
    #     -> RouterEvalCase
    #
    # 如果只传 user_input，则保留旧的手动 eval case 录入方式。
    if args.router_report or args.decision_id:
        result = record_router_correction(
            router_report=_load_router_report(args.router_report),
            decision_id=args.decision_id or "",
            user_input=args.user_input or "",
            expected_task_type=args.task_type,
            expected_risk_level=args.risk_level,
            expected_needs_tool=_parse_bool(args.needs_tool),
            correction_reason=args.reason,
            source=args.source,
            actor_id=args.actor_id,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 0

    if not args.user_input:
        raise SystemExit("--user-input is required when --router-report/--decision-id is not provided")
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


def _cmd_feedback_list(args: argparse.Namespace) -> int:
    records = read_router_feedback(args.limit)
    print(json.dumps(records, ensure_ascii=False, indent=2))
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


def _load_router_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--router-report must point to a JSON object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Router eval replay / feedback / trends CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="批量回放 Router eval dataset。")
    replay.add_argument("--dataset", default=None, help="JSON/JSONL dataset 路径。")
    replay.add_argument("--router-version", default=None, help="本次 Router 版本标识。")
    replay.add_argument("--max-failures", type=int, default=50, help="最多保留多少失败样本。")
    replay.set_defaults(func=_cmd_replay)

    gate = subparsers.add_parser("gate", help="运行 Router release gate，可用于 CI/pre-push。")
    gate.add_argument("--dataset", default=None, help="JSON/JSONL dataset 路径。")
    gate.add_argument("--router-version", default=None, help="本次 Router 版本标识。")
    gate.add_argument("--baseline", default=None, help="baseline JSON 路径。")
    gate.add_argument("--max-failures", type=int, default=50, help="最多保留多少失败样本。")
    gate.add_argument(
        "--record-trend",
        action="store_true",
        help="通过/失败都把本次 run 写入趋势文件。",
    )
    gate.set_defaults(func=_cmd_gate)

    baseline = subparsers.add_parser("write-baseline", help="写入 Router gate baseline。")
    baseline.add_argument("--dataset", default=None, help="JSON/JSONL dataset 路径。")
    baseline.add_argument("--router-version", default=None, help="本次 Router 版本标识。")
    baseline.add_argument("--output", default=None, help="baseline JSON 输出路径。")
    baseline.add_argument("--max-failures", type=int, default=50, help="最多保留多少失败样本。")
    baseline.add_argument(
        "--force",
        action="store_true",
        help="即使当前 release gate 不通过，也强制写 baseline。",
    )
    baseline.set_defaults(func=_cmd_write_baseline)

    feedback = subparsers.add_parser("feedback", help="把线上纠错反馈写成 eval case。")
    feedback.add_argument("--user-input", default="")
    feedback.add_argument("--router-report", default=None, help="router_report JSON 文件路径。")
    feedback.add_argument("--decision-id", default="", help="根据历史 RouterEvent decision_id 纠错。")
    feedback.add_argument(
        "--task-type",
        required=True,
        choices=["search", "write", "chat", "agent"],
    )
    feedback.add_argument("--risk-level", required=True, choices=["low", "medium", "high"])
    feedback.add_argument("--needs-tool", required=True)
    feedback.add_argument("--reason", required=True)
    feedback.add_argument("--source", default="manual_feedback")
    feedback.add_argument(
        "--actor-id",
        default=os.getenv("BEGINNER_AGENT_ROUTER_FEEDBACK_ACTOR_ID", "local-user"),
    )
    feedback.set_defaults(func=_cmd_feedback)

    feedback_list = subparsers.add_parser("feedback-list", help="查看 Router 人工纠错事件。")
    feedback_list.add_argument("--limit", type=int, default=None)
    feedback_list.set_defaults(func=_cmd_feedback_list)

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
