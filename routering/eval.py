from __future__ import annotations

from typing import Any

from .models import RouterDecision


def evaluate_router_prediction(
    case: dict[str, Any],
    decision: RouterDecision,
) -> dict[str, Any]:
    """评估一次 Router 预测是否命中 eval case。

    中文注释：
    这就是 Router eval 的最小闭环：
    - case 是历史沉淀的“输入 -> 期望路由”。
    - decision 是当前 Router 的预测。
    - 返回哪些字段命中、哪些字段不命中。

    后续可以把它扩展成批量回放、准确率趋势和失败样本分析。
    """

    checks = {
        "task_type": decision.task_type == case.get("expected_task_type"),
        "risk_level": decision.risk_level == case.get("expected_risk_level"),
        "needs_tool": decision.needs_tool == case.get("expected_needs_tool"),
    }
    mismatches = [name for name, ok in checks.items() if not ok]
    return {
        "passed": not mismatches,
        "checks": checks,
        "mismatches": mismatches,
        "expected": {
            "task_type": case.get("expected_task_type"),
            "risk_level": case.get("expected_risk_level"),
            "needs_tool": case.get("expected_needs_tool"),
        },
        "actual": decision.model_dump(mode="json"),
    }


def summarize_router_eval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 Router eval 结果。"""

    total = len(results)
    passed = sum(1 for item in results if item.get("passed") is True)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
    }
