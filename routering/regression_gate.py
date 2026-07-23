from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_project_env
from .eval_models import RouterEvalRun, router_eval_run_from_dict


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


@dataclass(frozen=True)
class RouterConfigFingerprint:
    """Router 配置指纹。

    中文注释：
    上线门禁不仅要知道“当前 eval 分数是多少”，
    还要知道这次分数对应哪一版 prompt / rules / security 配置。
    这里会把相关配置文件内容和关键 env 值做 hash。
    """

    fingerprint: str
    files: tuple[dict[str, Any], ...]
    env: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "files": list(self.files),
            "env": self.env,
        }


@dataclass(frozen=True)
class RouterEvalBaseline:
    """Router gate 使用的历史 baseline。"""

    run: RouterEvalRun
    config_fingerprint: RouterConfigFingerprint | None = None
    source: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.as_dict(),
            "config_fingerprint": (
                self.config_fingerprint.as_dict() if self.config_fingerprint else None
            ),
            "source": self.source,
        }


@dataclass(frozen=True)
class RouterReleaseGateResult:
    """Router 上线门禁结果。

    中文注释：
    regression gate 只看绝对阈值；
    release gate 在它的基础上增加 baseline 对比和配置指纹。
    CLI 可以根据 passed 返回 0/1，从而阻止提交或阻止启用新配置。
    """

    passed: bool
    reasons: tuple[str, ...]
    regression_gate: RouterRegressionGateResult
    baseline: dict[str, Any] | None
    config_fingerprint: RouterConfigFingerprint
    config_changed_from_baseline: bool
    thresholds: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "regression_gate": self.regression_gate.as_dict(),
            "baseline": self.baseline,
            "config_fingerprint": self.config_fingerprint.as_dict(),
            "config_changed_from_baseline": self.config_changed_from_baseline,
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


def evaluate_router_release_gate(
    run: RouterEvalRun,
    *,
    baseline: RouterEvalBaseline | None = None,
    config_fingerprint: RouterConfigFingerprint | None = None,
) -> RouterReleaseGateResult:
    """评估 Router 是否允许发布/启用。

    中文注释：
    生产级门禁通常同时看两类指标：

    1. 绝对阈值：
       当前准确率不能低于最低要求。

    2. baseline 下降幅度：
       改了 prompt/rules/security 后，即使仍然高于 0.90，
       如果比上一版明显退化，也应该阻止上线。
    """

    load_project_env()
    regression = evaluate_router_regression_gate(run)
    selected_fingerprint = config_fingerprint or current_router_config_fingerprint()
    drop_thresholds = _drop_thresholds()
    reasons = list(regression.reasons)
    baseline_payload: dict[str, Any] | None = None

    if baseline is not None:
        baseline_payload = baseline.as_dict()
        for metric, allowed_drop in drop_thresholds.items():
            current_value = _metric_value(run, metric)
            baseline_value = _metric_value(baseline.run, metric)
            drop = baseline_value - current_value
            if drop > allowed_drop:
                reasons.append(
                    f"{metric} drop {drop:.3f} > {allowed_drop:.3f} "
                    f"(baseline={baseline_value:.3f}, current={current_value:.3f})"
                )
    config_changed = (
        baseline.config_fingerprint.fingerprint != selected_fingerprint.fingerprint
        if baseline and baseline.config_fingerprint
        else False
    )

    return RouterReleaseGateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        regression_gate=regression,
        baseline=baseline_payload,
        config_fingerprint=selected_fingerprint,
        config_changed_from_baseline=config_changed,
        thresholds={
            **{f"min_{key}": value for key, value in regression.thresholds.items()},
            **{f"max_{key}_drop": value for key, value in drop_thresholds.items()},
        },
    )


def current_router_config_fingerprint() -> RouterConfigFingerprint:
    """计算当前 Router 配置指纹。"""

    load_project_env()
    file_env_names = (
        "BEGINNER_AGENT_ROUTER_PROMPT_PATH",
        "BEGINNER_AGENT_ROUTER_PROMPT_ROLLBACK_PATH",
        "BEGINNER_AGENT_ROUTER_RULES_PATH",
        "BEGINNER_AGENT_ROUTER_RULES_ROLLBACK_PATH",
        "BEGINNER_AGENT_ROUTER_SECURITY_POLICY_PATH",
        "BEGINNER_AGENT_ROUTER_ABUSE_PATTERNS_PATH",
    )
    value_env_names = (
        "BEGINNER_AGENT_ROUTER_VERSION",
        "BEGINNER_AGENT_ROUTER_PROMPT_VERSION",
        "BEGINNER_AGENT_ROUTER_PROMPT_EXPERIMENT_GROUP",
        "BEGINNER_AGENT_ROUTER_MIN_CONFIDENCE",
        "BEGINNER_AGENT_ROUTER_FAILURE_LOW_CONFIDENCE_POLICY",
        "BEGINNER_AGENT_ROUTER_FAILURE_RISK_POLICY",
        "BEGINNER_AGENT_ROUTER_FAILURE_SECURITY_POLICY",
    )
    files = tuple(_file_fingerprint(name) for name in file_env_names)
    env = {name: os.getenv(name, "") for name in value_env_names}
    raw = json.dumps(
        {
            "files": files,
            "env": env,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return RouterConfigFingerprint(
        fingerprint=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        files=files,
        env=env,
    )


def load_router_eval_baseline(path: str | Path | None = None) -> RouterEvalBaseline | None:
    """加载 Router baseline 文件。"""

    load_project_env()
    selected_path = path or os.getenv("BEGINNER_AGENT_ROUTER_EVAL_BASELINE_PATH", "").strip()
    if not selected_path:
        return None
    baseline_path = _resolve_path(selected_path)
    if not baseline_path.exists():
        return None
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Router baseline must be a JSON object: {baseline_path}")
    run_payload = data.get("run", data)
    if not isinstance(run_payload, dict):
        raise ValueError(f"Router baseline run must be a JSON object: {baseline_path}")
    fingerprint_payload = data.get("config_fingerprint")
    fingerprint = (
        _config_fingerprint_from_dict(fingerprint_payload)
        if isinstance(fingerprint_payload, dict)
        else None
    )
    return RouterEvalBaseline(
        run=router_eval_run_from_dict(run_payload),
        config_fingerprint=fingerprint,
        source=str(baseline_path),
    )


def write_router_eval_baseline(
    run: RouterEvalRun,
    *,
    path: str | Path | None = None,
    config_fingerprint: RouterConfigFingerprint | None = None,
) -> Path:
    """把当前通过门禁的 run 写成后续对比 baseline。"""

    load_project_env()
    selected_path = path or os.getenv(
        "BEGINNER_AGENT_ROUTER_EVAL_BASELINE_PATH",
        ".agent_state/router/router_eval_baseline.json",
    )
    baseline_path = _resolve_path(selected_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = RouterEvalBaseline(
        run=run,
        config_fingerprint=config_fingerprint or current_router_config_fingerprint(),
        source=str(baseline_path),
    ).as_dict()
    baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return baseline_path


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _drop_thresholds() -> dict[str, float]:
    return {
        "pass_rate": _float_env("BEGINNER_AGENT_ROUTER_GATE_MAX_PASS_RATE_DROP", 0.02),
        "task_type_accuracy": _float_env(
            "BEGINNER_AGENT_ROUTER_GATE_MAX_TASK_TYPE_ACCURACY_DROP", 0.02
        ),
        "risk_level_accuracy": _float_env(
            "BEGINNER_AGENT_ROUTER_GATE_MAX_RISK_LEVEL_ACCURACY_DROP", 0.02
        ),
        "needs_tool_accuracy": _float_env(
            "BEGINNER_AGENT_ROUTER_GATE_MAX_NEEDS_TOOL_ACCURACY_DROP", 0.02
        ),
    }


def _metric_value(run: RouterEvalRun, metric: str) -> float:
    return float(getattr(run, metric))


def _file_fingerprint(env_name: str) -> dict[str, Any]:
    path_value = os.getenv(env_name, "").strip()
    payload: dict[str, Any] = {
        "env": env_name,
        "path": path_value,
        "exists": False,
        "sha256": "",
    }
    if not path_value:
        return payload
    path = _resolve_path(path_value)
    payload["path"] = str(path)
    payload["exists"] = path.exists()
    if path.exists() and path.is_file():
        payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _config_fingerprint_from_dict(data: dict[str, Any]) -> RouterConfigFingerprint:
    return RouterConfigFingerprint(
        fingerprint=str(data.get("fingerprint", "")),
        files=tuple(item for item in data.get("files", []) if isinstance(item, dict)),
        env={str(key): str(value) for key, value in dict(data.get("env", {})).items()},
    )
