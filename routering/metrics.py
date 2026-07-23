from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tooling.core import ensure_state_dirs
from .models import RouterEvent
from .sinks import ROUTER_DIR


# 中文注释：
# metrics.py 是 Router 的本地在线指标聚合。
#
# 生产环境可以接 Prometheus / OpenTelemetry。
# 本地项目先写 JSON 文件，保证不依赖外部服务也能看趋势。


ROUTER_METRICS_FILE = ROUTER_DIR / "router_metrics.json"


@dataclass
class RouterMetricsSnapshot:
    request_total: int = 0
    fallback_total: int = 0
    repair_total: int = 0
    security_override_total: int = 0
    low_confidence_total: int = 0
    human_review_total: int = 0
    conflict_total: int = 0
    latency_total_ms: int = 0
    latency_max_ms: int = 0
    task_type_distribution: dict[str, int] = field(default_factory=dict)
    risk_level_distribution: dict[str, int] = field(default_factory=dict)
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        avg = self.latency_total_ms / self.request_total if self.request_total else 0
        return {
            "request_total": self.request_total,
            "fallback_total": self.fallback_total,
            "repair_total": self.repair_total,
            "security_override_total": self.security_override_total,
            "low_confidence_total": self.low_confidence_total,
            "human_review_total": self.human_review_total,
            "conflict_total": self.conflict_total,
            "latency_total_ms": self.latency_total_ms,
            "latency_avg_ms": avg,
            "latency_max_ms": self.latency_max_ms,
            "task_type_distribution": self.task_type_distribution,
            "risk_level_distribution": self.risk_level_distribution,
            "updated_at": self.updated_at or datetime.now(timezone.utc).isoformat(),
        }


def update_router_metrics(
    event: RouterEvent,
    *,
    conflict_count: int,
    human_review_required: bool,
) -> RouterMetricsSnapshot:
    """更新 Router 本地指标快照。"""

    snapshot = _read_metrics(ROUTER_METRICS_FILE)
    snapshot.request_total += 1
    snapshot.fallback_total += int(event.source == "fallback")
    snapshot.security_override_total += int(event.source == "security_override")
    snapshot.repair_total += sum(
        1 for item in event.failure_audit if int(item.get("repair_attempt_count", 0)) > 0
    )
    snapshot.low_confidence_total += int(event.decision.confidence < 0.5)
    snapshot.human_review_total += int(human_review_required)
    snapshot.conflict_total += conflict_count
    snapshot.latency_total_ms += event.latency_ms
    snapshot.latency_max_ms = max(snapshot.latency_max_ms, event.latency_ms)
    _inc(snapshot.task_type_distribution, event.decision.task_type)
    _inc(snapshot.risk_level_distribution, event.decision.risk_level)
    snapshot.updated_at = datetime.now(timezone.utc).isoformat()
    _write_metrics(ROUTER_METRICS_FILE, snapshot)
    return snapshot


def read_router_metrics() -> RouterMetricsSnapshot:
    """读取 Router 指标快照。"""

    return _read_metrics(ROUTER_METRICS_FILE)


def _read_metrics(path: Path) -> RouterMetricsSnapshot:
    if not path.exists():
        return RouterMetricsSnapshot()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RouterMetricsSnapshot()
    if not isinstance(data, dict):
        return RouterMetricsSnapshot()
    return RouterMetricsSnapshot(
        request_total=int(data.get("request_total", 0)),
        fallback_total=int(data.get("fallback_total", 0)),
        repair_total=int(data.get("repair_total", 0)),
        security_override_total=int(data.get("security_override_total", 0)),
        low_confidence_total=int(data.get("low_confidence_total", 0)),
        human_review_total=int(data.get("human_review_total", 0)),
        conflict_total=int(data.get("conflict_total", 0)),
        latency_total_ms=int(data.get("latency_total_ms", 0)),
        latency_max_ms=int(data.get("latency_max_ms", 0)),
        task_type_distribution=dict(data.get("task_type_distribution", {})),
        risk_level_distribution=dict(data.get("risk_level_distribution", {})),
        updated_at=str(data.get("updated_at", "")),
    )


def _write_metrics(path: Path, snapshot: RouterMetricsSnapshot) -> None:
    ensure_state_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _inc(bucket: dict[str, int], key: str) -> None:
    bucket[key] = int(bucket.get(key, 0)) + 1
