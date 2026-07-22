from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from .settings import (
    MAX_MEMORY_RERANK_TELEMETRY_EVENTS,
    MEMORY_DIR,
    MEMORY_RERANK_TELEMETRY_FILE,
)
from ..tooling.core import ensure_state_dirs


def rerank_ab_bucket(run_id: str, user_input: str) -> str:
    """给 reranker 分配稳定 A/B bucket。

    中文注释：
    大厂不会只凭感觉改 reranker。
    通常会把流量分成 A/B bucket，观察命中率、误召回、用户反馈。
    本地项目用稳定 hash 模拟这个机制。
    """

    configured = os.getenv("BEGINNER_AGENT_MEMORY_RERANK_BUCKET", "").strip()
    if configured:
        return configured
    raw = f"{run_id}:{user_input}".encode("utf-8")
    value = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    return "candidate" if value % 10 >= 5 else "control"


def _ensure_telemetry_file() -> None:
    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_RERANK_TELEMETRY_FILE.exists():
        MEMORY_RERANK_TELEMETRY_FILE.write_text("", encoding="utf-8")


def append_rerank_telemetry(event: dict[str, Any]) -> None:
    """记录一次 rerank telemetry。

    中文注释：
    这不是业务 memory，而是观测数据。
    后续可以用它分析：
    - 哪些 memory 被召回但丢弃。
    - 哪些 memory 进入 prompt。
    - rerank 分数分布。
    - A/B bucket 的命中情况。
    """

    _ensure_telemetry_file()
    events = read_rerank_telemetry(MAX_MEMORY_RERANK_TELEMETRY_EVENTS)
    events.append(
        {
            **event,
            "created_at": event.get("created_at")
            or datetime.now(timezone.utc).isoformat(),
        }
    )
    trimmed = events[-MAX_MEMORY_RERANK_TELEMETRY_EVENTS:]
    MEMORY_RERANK_TELEMETRY_FILE.write_text(
        "".join(
            f"{json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            for item in trimmed
        ),
        encoding="utf-8",
    )


def read_rerank_telemetry(limit: int = MAX_MEMORY_RERANK_TELEMETRY_EVENTS) -> list[dict[str, Any]]:
    """读取 rerank telemetry。"""

    _ensure_telemetry_file()
    events: list[dict[str, Any]] = []
    for line in MEMORY_RERANK_TELEMETRY_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events[-limit:]


def summarize_rerank_telemetry(limit: int = 200) -> dict[str, Any]:
    """生成命中率/误召回分析需要的轻量统计。"""

    events = read_rerank_telemetry(limit)
    by_bucket: dict[str, dict[str, int]] = {}
    for event in events:
        bucket = str(event.get("ab_bucket", "unknown"))
        stats = by_bucket.setdefault(
            bucket,
            {"runs": 0, "candidates": 0, "included": 0, "dropped": 0},
        )
        stats["runs"] += 1
        stats["candidates"] += int(event.get("candidate_count", 0) or 0)
        stats["included"] += int(event.get("included_count", 0) or 0)
        stats["dropped"] += int(event.get("dropped_count", 0) or 0)
    return {
        "event_count": len(events),
        "by_bucket": by_bucket,
    }
