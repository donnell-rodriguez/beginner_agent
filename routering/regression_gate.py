from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..config import load_project_env
from .eval_models import RouterEvalRun


# 中文注释：
# regression_gate.py 是 Router eval 的“上线门禁”。
#
# 它不会运行 eval；它接收 eval run 的结果，并根据阈值判断：
# - 是否允许启用当前 Router 配置。
# - 哪个指标没有达标。


@dataclass(frozen=True)
class RouterRegressionGateResult:
    passed: bool
    reasons: tuple[str, ...]
    thresholds: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "thresholds": self.thresholds,
        }


def evaluate_router_regression_gate(run: RouterEvalRun) -> RouterRegressionGateResult:
    """根据 eval run 判断 Router 是否通过回归门禁。"""

    load_project_env()
    thresholds = {
        "pass_rate": _float_env("BEGINNER_AGENT_ROUTER_GATE_MIN_PASS_RATE", 0.90),
        "task_type_accuracy": _float_env("BEGINNER_AGENT_ROUTER_GATE_MIN_TASK_TYPE_ACCURACY", 0.90),
        "risk_level_accuracy": _float_env("BEGINNER_AGENT_ROUTER_GATE_MIN_RISK_LEVEL_ACCURACY", 0.90),
        "needs_tool_accuracy": _float_env("BEGINNER_AGENT_ROUTER_GATE_MIN_NEEDS_TOOL_ACCURACY", 0.90),
    }
    reasons: list[str] = []
    if run.pass_rate < thresholds["pass_rate"]:
        reasons.append(f"pass_rate {run.pass_rate:.3f} < {thresholds['pass_rate']:.3f}")
    if run.task_type_accuracy < thresholds["task_type_accuracy"]:
        reasons.append(
            f"task_type_accuracy {run.task_type_accuracy:.3f} < {thresholds['task_type_accuracy']:.3f}"
        )
    if run.risk_level_accuracy < thresholds["risk_level_accuracy"]:
        reasons.append(
            f"risk_level_accuracy {run.risk_level_accuracy:.3f} < {thresholds['risk_level_accuracy']:.3f}"
        )
    if run.needs_tool_accuracy < thresholds["needs_tool_accuracy"]:
        reasons.append(
            f"needs_tool_accuracy {run.needs_tool_accuracy:.3f} < {thresholds['needs_tool_accuracy']:.3f}"
        )
    return RouterRegressionGateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        thresholds=thresholds,
    )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
