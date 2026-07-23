from __future__ import annotations

from typing import Any

from .models import RouterDecision
from .eval_categories import normalize_router_eval_category


def classify_router_eval_failure(mismatches: list[str]) -> str:
    """给 Router eval 失败做粗粒度归因。

    中文注释：
    大厂 eval 不会只说 failed。
    它至少要知道失败主要发生在哪一层：
    - intent_mismatch：任务类型错。
    - risk_mismatch：风险等级错，通常最影响安全。
    - tool_need_mismatch：是否需要工具判断错。
    - multi_field_mismatch：多个字段同时错。
    """

    if not mismatches:
        return "none"
    if len(mismatches) > 1:
        return "multi_field_mismatch"
    if mismatches[0] == "task_type":
        return "intent_mismatch"
    if mismatches[0] == "risk_level":
        return "risk_mismatch"
    if mismatches[0] == "needs_tool":
        return "tool_need_mismatch"
    return "unknown_mismatch"


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
    failure_category = classify_router_eval_failure(mismatches)
    return {
        "passed": not mismatches,
        "category": normalize_router_eval_category(case.get("category", "general")),
        "tags": list(case.get("tags", [])) if isinstance(case.get("tags", []), list) else [],
        "checks": checks,
        "mismatches": mismatches,
        "failure_category": failure_category,
        "expected": {
            "task_type": case.get("expected_task_type"),
            "risk_level": case.get("expected_risk_level"),
            "needs_tool": case.get("expected_needs_tool"),
        },
        "actual": decision.model_dump(mode="json"),
        "user_input": case.get("user_input", ""),
        "case_reason": case.get("reason", ""),
    }


def summarize_router_eval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 Router eval 结果。"""

    total = len(results)
    passed = sum(1 for item in results if item.get("passed") is True)
    task_type_passed = sum(1 for item in results if item.get("checks", {}).get("task_type") is True)
    risk_passed = sum(1 for item in results if item.get("checks", {}).get("risk_level") is True)
    tool_passed = sum(1 for item in results if item.get("checks", {}).get("needs_tool") is True)
    failure_categories: dict[str, int] = {}
    for item in results:
        category = str(item.get("failure_category", "unknown_mismatch"))
        if category == "none":
            continue
        failure_categories[category] = failure_categories.get(category, 0) + 1
    category_metrics = _summarize_by_category(results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "task_type_accuracy": task_type_passed / total if total else 0.0,
        "risk_level_accuracy": risk_passed / total if total else 0.0,
        "needs_tool_accuracy": tool_passed / total if total else 0.0,
        "failure_categories": failure_categories,
        "category_metrics": category_metrics,
    }


def _summarize_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 eval category 汇总指标。

    中文注释：
    总体 pass_rate 可能很好看，但某个关键类别可能已经坏了。
    例如 normal_chat_cases 100%，prompt_injection_cases 0%，总体也可能不低。
    所以生产级 Router eval 必须按类别拆开看。
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        category = normalize_router_eval_category(item.get("category", "general"))
        grouped.setdefault(category, []).append(item)

    metrics: dict[str, dict[str, Any]] = {}
    for category, items in grouped.items():
        total = len(items)
        passed = sum(1 for item in items if item.get("passed") is True)
        task_type_passed = sum(
            1 for item in items if item.get("checks", {}).get("task_type") is True
        )
        risk_passed = sum(
            1 for item in items if item.get("checks", {}).get("risk_level") is True
        )
        tool_passed = sum(
            1 for item in items if item.get("checks", {}).get("needs_tool") is True
        )
        failure_categories: dict[str, int] = {}
        for item in items:
            failure_category = str(item.get("failure_category", "unknown_mismatch"))
            if failure_category == "none":
                continue
            failure_categories[failure_category] = failure_categories.get(failure_category, 0) + 1
        metrics[category] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0.0,
            "task_type_accuracy": task_type_passed / total if total else 0.0,
            "risk_level_accuracy": risk_passed / total if total else 0.0,
            "needs_tool_accuracy": tool_passed / total if total else 0.0,
            "failure_categories": failure_categories,
        }
    return metrics
