from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import load_project_env
from ..tooling.core import ensure_state_dirs
from .conflicts import RouterConflict
from .models import RouterEvent
from .sinks import ROUTER_DIR


# 中文注释：
# review.py 是 Router 人工复核队列。
#
# 高风险、低置信度、security override、repair 后通过、LLM/规则冲突，
# 都可以进入队列，后续由 CLI/API/UI 做人工复核。


ROUTER_REVIEW_QUEUE_FILE = ROUTER_DIR / "router_review_queue.jsonl"


@dataclass(frozen=True)
class RouterReviewItem:
    decision_id: str
    run_id: str
    user_input: str
    reasons: tuple[str, ...]
    priority: str
    status: str = "pending"
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": True,
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "user_input": self.user_input,
            "reasons": list(self.reasons),
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }


def build_router_review_item(
    event: RouterEvent,
    *,
    conflicts: tuple[RouterConflict, ...],
    max_total_latency_ms: int,
) -> RouterReviewItem | None:
    """判断是否需要人工复核，并构造队列项。"""

    load_project_env()
    if os.getenv("BEGINNER_AGENT_ROUTER_REVIEW_QUEUE_ENABLED", "true").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    reasons: list[str] = []
    if event.decision.risk_level == "high":
        reasons.append("risk_level=high")
    if event.source == "security_override":
        reasons.append("security_override")
    if conflicts:
        reasons.append("router_conflict")
    if event.latency_ms > max_total_latency_ms:
        reasons.append(f"latency_ms>{max_total_latency_ms}")
    if any(int(item.get("repair_attempt_count", 0)) > 0 for item in event.failure_audit):
        reasons.append("repair_used")
    if event.decision.confidence < _float_env("BEGINNER_AGENT_ROUTER_REVIEW_CONFIDENCE", 0.5):
        reasons.append("low_confidence")

    if not reasons:
        return None

    priority = "high" if event.decision.risk_level == "high" or event.source == "security_override" else "medium"
    return RouterReviewItem(
        decision_id=event.decision_id,
        run_id=event.run_id,
        user_input=event.user_input,
        reasons=tuple(dict.fromkeys(reasons)),
        priority=priority,
    )


def append_router_review_item(item: RouterReviewItem) -> None:
    """写入 Router 人工复核队列。"""

    ensure_state_dirs()
    ROUTER_REVIEW_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ROUTER_REVIEW_QUEUE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")


def read_router_review_queue(limit: int | None = None) -> list[dict[str, Any]]:
    """读取待复核队列。"""

    return _read_jsonl(ROUTER_REVIEW_QUEUE_FILE)[-limit:] if limit else _read_jsonl(ROUTER_REVIEW_QUEUE_FILE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            result.append(data)
    return result


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
