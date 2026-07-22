from __future__ import annotations

import json
import os
from typing import Any, Protocol

from .settings import MEMORY_DIR
from ..tooling.core import ensure_state_dirs


# 中文注释：
# observability_sinks.py 让 memory 观测成为可插拔旁路。
#
# 当前支持：
# - jsonl：本地直接写文件。
# - kafka_spool：写本地 spool，后续由 producer/sidecar 发 Kafka。
# - otel_spool：写本地 OpenTelemetry 风格事件，后续由 collector 采集。
# - null：测试或关闭观测。
#
# 关键原则：
# observability 失败不能影响 Memory Retriever / Writer 主路径。


MEMORY_OBSERVABILITY_FILE = MEMORY_DIR / "memory_observability.jsonl"
MEMORY_KAFKA_SPOOL_FILE = MEMORY_DIR / "memory_kafka_spool.jsonl"
MEMORY_OTEL_SPOOL_FILE = MEMORY_DIR / "memory_otel_spool.jsonl"
MAX_MEMORY_OBSERVABILITY_EVENTS = 3000


class MemoryObservabilitySink(Protocol):
    def append(self, event: dict[str, Any]) -> None:
        """写入 memory 观测事件。"""


class NullMemoryObservabilitySink:
    def append(self, event: dict[str, Any]) -> None:
        return None


class JsonlMemoryObservabilitySink:
    def append(self, event: dict[str, Any]) -> None:
        _append_jsonl(MEMORY_OBSERVABILITY_FILE, event)


class KafkaSpoolMemoryObservabilitySink:
    def append(self, event: dict[str, Any]) -> None:
        _append_jsonl(
            MEMORY_KAFKA_SPOOL_FILE,
            {
                **event,
                "_sink": "kafka_spool",
                "_topic": os.getenv(
                    "BEGINNER_AGENT_MEMORY_KAFKA_TOPIC",
                    "beginner_agent.memory_events",
                ),
            },
        )


class OTelSpoolMemoryObservabilitySink:
    def append(self, event: dict[str, Any]) -> None:
        _append_jsonl(
            MEMORY_OTEL_SPOOL_FILE,
            {
                "name": f"beginner_agent.memory.{event.get('event_type', 'event')}",
                "attributes": event,
                "_sink": "otel_spool",
                "_exporter": os.getenv(
                    "BEGINNER_AGENT_MEMORY_OTEL_EXPORTER",
                    "local-spool",
                ),
            },
        )


def append_memory_observability_event(event: dict[str, Any]) -> None:
    """安全追加 memory 观测事件。

    中文注释：
    这里会吞掉 sink 错误，因为 memory 主流程不能被观测系统拖垮。
    """

    try:
        resolve_memory_observability_sink().append(event)
    except Exception:
        return None


def read_memory_observability_events(limit: int = MAX_MEMORY_OBSERVABILITY_EVENTS) -> list[dict[str, Any]]:
    """读取本地 memory observability events。

    中文注释：
    这里同时读取三类本地文件：
    - memory_observability.jsonl：普通 JSONL sink。
    - memory_kafka_spool.jsonl：准备发给 Kafka 的本地 spool。
    - memory_otel_spool.jsonl：准备交给 OTel collector 的本地 spool。

    真实生产环境会直接查日志平台/指标平台；
    当前项目先保留本地可观测证据，方便测试和学习。
    """

    events = (
        _read_jsonl(MEMORY_OBSERVABILITY_FILE, limit)
        + _read_jsonl(MEMORY_KAFKA_SPOOL_FILE, limit)
        + _read_jsonl(MEMORY_OTEL_SPOOL_FILE, limit)
    )
    events.sort(key=lambda item: str(item.get("created_at", "")))
    return events[-limit:]


def resolve_memory_observability_sink() -> MemoryObservabilitySink:
    sink = os.getenv("BEGINNER_AGENT_MEMORY_OBSERVABILITY_SINK", "jsonl").strip().lower()
    enabled = os.getenv("BEGINNER_AGENT_MEMORY_OBSERVABILITY_ENABLED", "true").strip().lower()
    if enabled not in {"1", "true", "yes", "on"} or sink == "null":
        return NullMemoryObservabilitySink()
    if sink == "kafka_spool":
        return KafkaSpoolMemoryObservabilitySink()
    if sink == "otel_spool":
        return OTelSpoolMemoryObservabilitySink()
    return JsonlMemoryObservabilitySink()


def _append_jsonl(path: Any, event: dict[str, Any]) -> None:
    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    events = _read_jsonl(path, MAX_MEMORY_OBSERVABILITY_EVENTS)
    events.append(event)
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in events[-MAX_MEMORY_OBSERVABILITY_EVENTS:]
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Any, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events[-limit:]
