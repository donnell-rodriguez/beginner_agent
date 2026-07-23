from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from ..config import load_project_env
from ..state import RiskLevel, TaskType
from ..tooling.core import ensure_state_dirs
from .eval import evaluate_router_prediction, summarize_router_eval_results
from .eval_models import RouterEvalDataset, RouterEvalFailure, RouterEvalRun, RouterFeedbackRecord
from .models import RouterDecision, RouterEvalCase
from .observability import append_router_eval_case
from .regression_gate import evaluate_router_regression_gate
from .sinks import ROUTER_DIR


# 中文注释：
# eval_runner.py 负责 Router eval 的“批量运行和沉淀”。
#
# eval.py 只做单条预测评估；
# eval_runner.py 做生产系统更关心的事情：
# - 加载 dataset version。
# - 批量 replay。
# - 生成 run_id。
# - 记录趋势。
# - 归因失败样本。
# - 把线上反馈写成 eval case。


ROUTER_EVAL_TRENDS_FILE = ROUTER_DIR / "router_eval_trends.jsonl"
DEFAULT_ROUTER_EVAL_DATASET_VERSION = "router-eval-local-v1"
DEFAULT_ROUTER_VERSION = "router-local"


RouterPredictFn = Callable[[str], RouterDecision]


def load_router_eval_dataset(path: str | Path | None = None) -> RouterEvalDataset:
    """加载 Router eval 数据集。

    中文注释：
    支持两种格式：

    1. JSON：

        {"version": "v1", "cases": [{...}, {...}]}

    2. JSONL：

        每一行一个 eval case。

    如果 path 为空，则从 observability sink 读取已经沉淀的 eval cases。
    """

    load_project_env()
    resolved_path = path or os.getenv("BEGINNER_AGENT_ROUTER_EVAL_DATASET_PATH", "").strip()
    if not resolved_path:
        from .observability import read_router_eval_cases

        cases = tuple(read_router_eval_cases())
        return RouterEvalDataset(
            version=os.getenv(
                "BEGINNER_AGENT_ROUTER_EVAL_DATASET_VERSION",
                DEFAULT_ROUTER_EVAL_DATASET_VERSION,
            ),
            cases=cases,
            source="observability",
        )

    dataset_path = _resolve_path(resolved_path)
    if dataset_path.suffix == ".jsonl":
        cases = tuple(_read_jsonl(dataset_path))
        return RouterEvalDataset(
            version=os.getenv(
                "BEGINNER_AGENT_ROUTER_EVAL_DATASET_VERSION",
                dataset_path.stem,
            ),
            cases=cases,
            source=str(dataset_path),
        )

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Router eval dataset must be a JSON object: {dataset_path}")
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError(f"Router eval dataset cases must be a list: {dataset_path}")
    cases = tuple(item for item in raw_cases if isinstance(item, dict))
    return RouterEvalDataset(
        version=str(data.get("version", dataset_path.stem)).strip() or dataset_path.stem,
        cases=cases,
        source=str(dataset_path),
    )


def run_router_eval(
    dataset: RouterEvalDataset,
    predict: RouterPredictFn,
    *,
    router_version: str | None = None,
    max_failures: int = 50,
) -> RouterEvalRun:
    """批量回放 Router eval dataset。"""

    results: list[dict[str, Any]] = []
    failures: list[RouterEvalFailure] = []
    selected_router_version = router_version or os.getenv(
        "BEGINNER_AGENT_ROUTER_VERSION",
        DEFAULT_ROUTER_VERSION,
    )

    for case in dataset.cases:
        user_input = str(case.get("user_input", ""))
        if not user_input:
            continue
        decision = predict(user_input)
        result = evaluate_router_prediction(case, decision)
        results.append(result)
        if result["passed"] is not True and len(failures) < max_failures:
            failures.append(
                RouterEvalFailure(
                    user_input=user_input,
                    mismatches=tuple(str(item) for item in result.get("mismatches", [])),
                    failure_category=str(result.get("failure_category", "unknown_mismatch")),
                    expected=dict(result.get("expected", {})),
                    actual=dict(result.get("actual", {})),
                    reason=str(case.get("reason", "")),
                )
            )

    summary = summarize_router_eval_results(results)
    run_id = _router_eval_run_id(dataset.version, selected_router_version, summary)
    return RouterEvalRun(
        run_id=run_id,
        dataset_version=dataset.version,
        router_version=selected_router_version,
        total=int(summary["total"]),
        passed=int(summary["passed"]),
        failed=int(summary["failed"]),
        pass_rate=float(summary["pass_rate"]),
        task_type_accuracy=float(summary["task_type_accuracy"]),
        risk_level_accuracy=float(summary["risk_level_accuracy"]),
        needs_tool_accuracy=float(summary["needs_tool_accuracy"]),
        failures=tuple(failures),
    )


def append_router_eval_trend(run: RouterEvalRun) -> None:
    """把一次 eval run 写入趋势文件。"""

    records = _read_jsonl(ROUTER_EVAL_TRENDS_FILE)
    payload = run.as_dict()
    payload["regression_gate"] = evaluate_router_regression_gate(run).as_dict()
    records.append(payload)
    limit = _positive_int_env("BEGINNER_AGENT_MAX_ROUTER_EVAL_TRENDS", 500)
    _write_jsonl(ROUTER_EVAL_TRENDS_FILE, records[-limit:])


def read_router_eval_trends(limit: int | None = None) -> list[dict[str, Any]]:
    """读取最近的 Router eval 趋势。"""

    records = _read_jsonl(ROUTER_EVAL_TRENDS_FILE)
    return records[-limit:] if limit else records


def append_router_feedback(record: RouterFeedbackRecord) -> RouterEvalCase:
    """把线上反馈沉淀成 eval case。

    中文注释：
    这就是“反馈回流”：

        用户/开发者发现 Router 分错
          -> 写入 RouterFeedbackRecord
          -> 追加 RouterEvalCase
          -> 下一次批量 replay 会覆盖这个场景
    """

    case = RouterEvalCase(
        user_input=record.user_input,
        expected_task_type=record.expected_task_type,
        expected_risk_level=record.expected_risk_level,
        expected_needs_tool=record.expected_needs_tool,
        reason=f"{record.source}: {record.reason}",
        created_at=record.created_at,
    )
    append_router_eval_case(case)
    return case


def make_feedback_record(
    *,
    user_input: str,
    expected_task_type: str,
    expected_risk_level: str,
    expected_needs_tool: bool,
    reason: str,
    source: str = "manual_feedback",
) -> RouterFeedbackRecord:
    """构造并校验一条 Router feedback。"""

    if expected_task_type not in {"search", "write", "chat", "agent"}:
        raise ValueError(f"Invalid expected_task_type: {expected_task_type}")
    if expected_risk_level not in {"low", "medium", "high"}:
        raise ValueError(f"Invalid expected_risk_level: {expected_risk_level}")
    return RouterFeedbackRecord(
        user_input=user_input,
        expected_task_type=cast(TaskType, expected_task_type),
        expected_risk_level=cast(RiskLevel, expected_risk_level),
        expected_needs_tool=expected_needs_tool,
        reason=reason,
        source=source,
    )


def _router_eval_run_id(
    dataset_version: str,
    router_version: str,
    summary: dict[str, Any],
) -> str:
    raw = json.dumps(
        {
            "dataset_version": dataset_version,
            "router_version": router_version,
            "summary": summary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_state_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default
