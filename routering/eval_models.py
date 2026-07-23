from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..state import RiskLevel, TaskType


# 中文注释：
# eval_models.py 专门放 Router eval 的数据结构。
#
# 生产级 eval 不是只比较一次结果，而是要回答：
# - 这批样本属于哪个 dataset version？
# - 本次 replay 使用哪个 router version？
# - 哪些样本失败了，失败原因是什么？
# - 指标趋势是否变差？
# - 线上人工反馈如何沉淀成新样本？


@dataclass(frozen=True)
class RouterEvalDataset:
    """Router eval 数据集。"""

    version: str
    cases: tuple[dict[str, Any], ...]
    source: str = "memory"

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "case_count": len(self.cases),
            "cases": list(self.cases),
        }


@dataclass(frozen=True)
class RouterEvalFailure:
    """一条失败样本的归因结果。"""

    user_input: str
    mismatches: tuple[str, ...]
    failure_category: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "mismatches": list(self.mismatches),
            "failure_category": self.failure_category,
            "expected": self.expected,
            "actual": self.actual,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RouterEvalRun:
    """一次批量 Router eval run。"""

    run_id: str
    dataset_version: str
    router_version: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    task_type_accuracy: float
    risk_level_accuracy: float
    needs_tool_accuracy: float
    # 中文注释：
    # category_metrics 按样本分层统计指标。
    #
    # 例子：
    #   normal_chat_cases      -> 普通聊天是否还稳定。
    #   prompt_injection_cases -> 提示词注入是否被正确识别为高风险。
    #
    # 这样改 Router 时，不只看总体准确率，还能知道是哪一类能力退化。
    category_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: tuple[RouterEvalFailure, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "router_version": self.router_version,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "task_type_accuracy": self.task_type_accuracy,
            "risk_level_accuracy": self.risk_level_accuracy,
            "needs_tool_accuracy": self.needs_tool_accuracy,
            "category_metrics": self.category_metrics,
            "failures": [failure.as_dict() for failure in self.failures],
            "created_at": self.created_at,
        }


def router_eval_run_from_dict(data: dict[str, Any]) -> RouterEvalRun:
    """从 JSON/dict 恢复 RouterEvalRun。

    中文注释：
    regression gate 需要读取历史 baseline run。
    baseline 通常保存在 JSON 文件里，所以这里提供一个小的反序列化 helper。
    """

    failures = tuple(
        RouterEvalFailure(
            user_input=str(item.get("user_input", "")),
            mismatches=tuple(str(value) for value in item.get("mismatches", [])),
            failure_category=str(item.get("failure_category", "unknown_mismatch")),
            expected=dict(item.get("expected", {})),
            actual=dict(item.get("actual", {})),
            reason=str(item.get("reason", "")),
        )
        for item in data.get("failures", [])
        if isinstance(item, dict)
    )
    return RouterEvalRun(
        run_id=str(data.get("run_id", "")),
        dataset_version=str(data.get("dataset_version", "")),
        router_version=str(data.get("router_version", "")),
        total=int(data.get("total", 0)),
        passed=int(data.get("passed", 0)),
        failed=int(data.get("failed", 0)),
        pass_rate=float(data.get("pass_rate", 0.0)),
        task_type_accuracy=float(data.get("task_type_accuracy", 0.0)),
        risk_level_accuracy=float(data.get("risk_level_accuracy", 0.0)),
        needs_tool_accuracy=float(data.get("needs_tool_accuracy", 0.0)),
        category_metrics={
            str(key): dict(value)
            for key, value in dict(data.get("category_metrics", {})).items()
            if isinstance(value, dict)
        },
        failures=failures,
        created_at=str(data.get("created_at", "")),
    )


@dataclass(frozen=True)
class RouterFeedbackRecord:
    """线上反馈记录。

    中文注释：
    当用户或开发者发现 Router 分错了，可以把“实际输入 + 正确答案”
    写成 feedback record。后续它会变成 eval case，进入批量回放。
    """

    user_input: str
    expected_task_type: TaskType
    expected_risk_level: RiskLevel
    expected_needs_tool: bool
    reason: str
    source: str = "manual_feedback"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "expected_task_type": self.expected_task_type,
            "expected_risk_level": self.expected_risk_level,
            "expected_needs_tool": self.expected_needs_tool,
            "reason": self.reason,
            "source": self.source,
            "created_at": self.created_at,
        }
