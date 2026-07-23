from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from .config import load_project_env
from .tooling.core import STATE_DIR, ensure_state_dirs


# 中文注释：
# checkpoint_observability.py 专门负责 checkpoint 事件“写到哪里”。
#
# checkpoint_node.py 只负责产生 checkpoint health event；
# 这个文件负责把事件写入 JSONL / Kafka spool / Null sink。
#
# 这样后续要接 Kafka、OpenTelemetry、Prometheus、告警系统时，
# 不需要改 checkpoint_node.py 的主逻辑。


CHECKPOINT_DIR = STATE_DIR / "checkpoint"
CHECKPOINT_EVENTS_FILE = CHECKPOINT_DIR / "checkpoint_events.jsonl"
CHECKPOINT_KAFKA_SPOOL_FILE = CHECKPOINT_DIR / "checkpoint_kafka_spool.jsonl"

DEFAULT_MAX_CHECKPOINT_EVENTS = 2000


class CheckpointObservabilitySink(Protocol):
    """Checkpoint 观测 sink 接口。

    中文注释：
    Protocol 类似接口。
    只要一个类实现 append_event / read_events，
    就可以作为 checkpoint observability sink。
    """

    def append_event(self, event: dict[str, Any]) -> None:
        """写入 checkpoint 观测事件。"""

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """读取 checkpoint 观测事件。"""


def checkpoint_observability_enabled() -> bool:
    """读取 checkpoint observability 总开关。"""

    load_project_env()
    raw = os.getenv("BEGINNER_AGENT_CHECKPOINT_OBSERVABILITY_ENABLED", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def checkpoint_observability_sink_name() -> str:
    """读取 checkpoint observability sink 类型。

    中文注释：
    当前支持：
    - jsonl：默认写入 .agent_state/checkpoint/checkpoint_events.jsonl。
    - null：不落盘，适合测试或隐私场景。
    - kafka_spool：先写本地 spool 文件，后续由 Kafka producer/sidecar 转发。
    """

    load_project_env()
    return os.getenv("BEGINNER_AGENT_CHECKPOINT_OBSERVABILITY_SINK", "jsonl").strip().lower()


def append_checkpoint_event(event: dict[str, Any]) -> None:
    """写入 checkpoint event。

    中文注释：
    这里刻意吞掉 OSError：
    checkpoint observability 是旁路能力，
    写日志失败不应该拖垮 agent 主流程。
    """

    try:
        resolve_checkpoint_observability_sink().append_event(event)
    except OSError:
        return None


def read_checkpoint_events(limit: int | None = None) -> list[dict[str, Any]]:
    """读取 checkpoint event。"""

    return resolve_checkpoint_observability_sink().read_events(limit)


class NullCheckpointObservabilitySink:
    """不写任何 checkpoint 事件的 sink。"""

    def append_event(self, event: dict[str, Any]) -> None:
        return None

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return []


class JsonlCheckpointObservabilitySink:
    """把 checkpoint 事件写入本地 JSONL 文件。"""

    def append_event(self, event: dict[str, Any]) -> None:
        records = _read_jsonl(CHECKPOINT_EVENTS_FILE, _max_checkpoint_events())
        records.append(_normalize_checkpoint_event(event, sink="jsonl"))
        _write_jsonl(CHECKPOINT_EVENTS_FILE, records[-_max_checkpoint_events():])

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return _read_jsonl(CHECKPOINT_EVENTS_FILE, limit or _max_checkpoint_events())


class KafkaSpoolCheckpointObservabilitySink:
    """Kafka 接入前的本地 spool sink。

    中文注释：
    生产级系统常见链路是：

        checkpoint event -> Kafka topic -> consumer -> metrics/dashboard/alert

    当前项目先不引入 Kafka 依赖。
    这个 sink 会把事件写到 spool 文件，后续可以由独立 producer 发送到 Kafka。
    """

    def append_event(self, event: dict[str, Any]) -> None:
        records = _read_jsonl(CHECKPOINT_KAFKA_SPOOL_FILE, _max_checkpoint_events())
        payload = _normalize_checkpoint_event(event, sink="kafka_spool")
        payload["_topic"] = os.getenv(
            "BEGINNER_AGENT_CHECKPOINT_KAFKA_TOPIC",
            "beginner_agent.checkpoint_events",
        )
        records.append(payload)
        _write_jsonl(CHECKPOINT_KAFKA_SPOOL_FILE, records[-_max_checkpoint_events():])

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return JsonlCheckpointObservabilitySink().read_events(limit)


def resolve_checkpoint_observability_sink() -> CheckpointObservabilitySink:
    """根据 env 选择 checkpoint observability sink。"""

    if not checkpoint_observability_enabled():
        return NullCheckpointObservabilitySink()

    sink_name = checkpoint_observability_sink_name()
    if sink_name == "null":
        return NullCheckpointObservabilitySink()
    if sink_name == "kafka_spool":
        return KafkaSpoolCheckpointObservabilitySink()
    return JsonlCheckpointObservabilitySink()


def _normalize_checkpoint_event(event: dict[str, Any], *, sink: str) -> dict[str, Any]:
    """补齐 checkpoint event 的统一字段。"""

    payload = dict(event)
    payload["_sink"] = sink
    payload.setdefault("event_type", "checkpoint_health")
    payload.setdefault("alerts", _checkpoint_alerts(payload))
    return payload


def _checkpoint_alerts(event: dict[str, Any]) -> list[dict[str, str]]:
    """根据 checkpoint event 生成轻量告警信号。

    中文注释：
    这不是完整告警系统，而是把“哪些事件值得报警”结构化出来。
    后续 Kafka/OTel/Prometheus 消费者可以直接读取 alerts 字段。
    """

    alerts: list[dict[str, str]] = []
    status = str(event.get("status", ""))
    if status in {"blocked", "degraded"}:
        alerts.append(
            {
                "severity": "critical" if status == "blocked" else "warning",
                "code": f"checkpoint_{status}",
                "message": f"checkpoint health status is {status}",
            }
        )
    if event.get("requested_backend") == "postgres" and event.get("backend") == "memory":
        alerts.append(
            {
                "severity": "warning",
                "code": "checkpoint_fallback_to_memory",
                "message": "requested postgres checkpoint but effective backend is memory",
            }
        )
    elif event.get("persistent") is False:
        alerts.append(
            {
                "severity": "warning",
                "code": "checkpoint_non_persistent",
                "message": "checkpoint backend is not durable across process restarts",
            }
        )
    if event.get("setup_status") == "missing_tables":
        alerts.append(
            {
                "severity": "warning",
                "code": "checkpoint_setup_missing",
                "message": "postgres checkpoint tables are missing",
            }
        )
    if event.get("resume_supported") is False and event.get("persistent") is True:
        alerts.append(
            {
                "severity": "warning",
                "code": "checkpoint_resume_not_supported",
                "message": "persistent checkpoint backend cannot currently resume this run",
            }
        )
    return alerts


def _max_checkpoint_events() -> int:
    raw = os.getenv("BEGINNER_AGENT_MAX_CHECKPOINT_EVENTS", str(DEFAULT_MAX_CHECKPOINT_EVENTS))
    try:
        value = int(raw.strip())
    except ValueError:
        return DEFAULT_MAX_CHECKPOINT_EVENTS
    return value if value > 0 else DEFAULT_MAX_CHECKPOINT_EVENTS


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
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
    return records[-limit:]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_state_dirs()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")
