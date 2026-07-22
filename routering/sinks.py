from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from ..config import load_project_env
from ..tooling.core import STATE_DIR, ensure_state_dirs
from .models import RouterEvalCase, RouterEvent


# 中文注释：
# 这个文件专门负责 Router observability 的“写到哪里”。
#
# 过去 observability.py 直接写 JSONL 文件，后续如果要换成 Kafka、
# OpenTelemetry、HTTP Collector，就会改到业务入口。
#
# 现在改成 sink 结构：
#
#     routering/nodes.py
#       -> observability.append_router_event(...)
#       -> resolve_router_observability_sink()
#       -> Jsonl / Null / KafkaSpool sink
#
# 这样 Router 节点只关心“产生事件”，不关心“事件发到哪里”。


ROUTER_DIR = STATE_DIR / "router"
ROUTER_EVENTS_FILE = ROUTER_DIR / "router_events.jsonl"
ROUTER_EVAL_CASES_FILE = ROUTER_DIR / "router_eval_cases.jsonl"
ROUTER_FEEDBACK_FILE = ROUTER_DIR / "router_feedback.jsonl"
ROUTER_KAFKA_SPOOL_FILE = ROUTER_DIR / "router_kafka_spool.jsonl"

DEFAULT_MAX_ROUTER_EVENTS = 2000
DEFAULT_MAX_ROUTER_EVAL_CASES = 1000


class RouterObservabilitySink(Protocol):
    """Router 观测 sink 的接口。

    中文注释：
    Protocol 类似“约定一个对象必须有哪些方法”。
    只要一个类实现了下面三个方法，就可以被当作 Router observability sink。

    生产级系统常见做法就是这样：
    - 业务层只依赖接口。
    - JSONL / Kafka / OTel / HTTP 只是不同实现。
    - 后续替换后端时，不需要改 Router 节点。
    """

    def append_event(self, event: RouterEvent) -> None:
        """写入一次 Router 决策事件。"""

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """读取 Router 决策事件，用于定位某次错误路由。"""

    def append_eval_case(self, case: RouterEvalCase) -> None:
        """写入一条 Router 离线评估 case。"""

    def read_eval_cases(self, limit: int | None = None) -> list[dict[str, Any]]:
        """读取 Router eval case，用于离线回放。"""

    def append_feedback_event(self, event: dict[str, Any]) -> None:
        """写入一条人工纠错反馈事件。"""

    def read_feedback_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """读取人工纠错反馈事件。"""


def router_observability_enabled() -> bool:
    """读取 Router observability 总开关。"""

    load_project_env()
    raw = os.getenv("BEGINNER_AGENT_ROUTER_OBSERVABILITY_ENABLED", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def router_observability_sink_name() -> str:
    """读取当前 sink 类型。

    中文注释：
    当前支持：
    - jsonl：默认，写入本地 .agent_state/router/*.jsonl。
    - null：完全不落盘，适合测试或临时关闭观测。
    - kafka_spool：先写本地 spool 文件，后续由 Kafka producer/sidecar 转发。

    为什么不是直接连 Kafka？
    因为当前项目没有引入 Kafka client，也没有部署 Kafka。
    先做 spool sink 可以让事件格式和接口稳定下来，后续接真实 Kafka 时只新增实现。
    """

    load_project_env()
    return os.getenv("BEGINNER_AGENT_ROUTER_OBSERVABILITY_SINK", "jsonl").strip().lower()


def _max_router_events() -> int:
    return _positive_int_env("BEGINNER_AGENT_MAX_ROUTER_EVENTS", DEFAULT_MAX_ROUTER_EVENTS)


def _max_router_eval_cases() -> int:
    return _positive_int_env(
        "BEGINNER_AGENT_MAX_ROUTER_EVAL_CASES",
        DEFAULT_MAX_ROUTER_EVAL_CASES,
    )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


class NullRouterObservabilitySink:
    """不写任何观测事件的 sink。

    中文注释：
    它不是错误处理，而是一种明确配置：
    当本地测试或隐私场景不希望落盘时，可以设置：

        BEGINNER_AGENT_ROUTER_OBSERVABILITY_SINK=null
    """

    def append_event(self, event: RouterEvent) -> None:
        return None

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return []

    def append_eval_case(self, case: RouterEvalCase) -> None:
        return None

    def read_eval_cases(self, limit: int | None = None) -> list[dict[str, Any]]:
        return []

    def append_feedback_event(self, event: dict[str, Any]) -> None:
        return None

    def read_feedback_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return []


class JsonlRouterObservabilitySink:
    """把 Router 观测事件写入本地 JSONL 文件。"""

    def append_event(self, event: RouterEvent) -> None:
        records = _read_jsonl(ROUTER_EVENTS_FILE, _max_router_events())
        records.append(event.as_dict())
        _write_jsonl(ROUTER_EVENTS_FILE, records[-_max_router_events():])

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return _read_jsonl(ROUTER_EVENTS_FILE, limit or _max_router_events())

    def append_eval_case(self, case: RouterEvalCase) -> None:
        records = _read_jsonl(ROUTER_EVAL_CASES_FILE, _max_router_eval_cases())
        records.append(case.as_dict())
        _write_jsonl(ROUTER_EVAL_CASES_FILE, records[-_max_router_eval_cases():])

    def read_eval_cases(self, limit: int | None = None) -> list[dict[str, Any]]:
        return _read_jsonl(ROUTER_EVAL_CASES_FILE, limit or _max_router_eval_cases())

    def append_feedback_event(self, event: dict[str, Any]) -> None:
        records = _read_jsonl(ROUTER_FEEDBACK_FILE, _max_router_eval_cases())
        records.append(event)
        _write_jsonl(ROUTER_FEEDBACK_FILE, records[-_max_router_eval_cases():])

    def read_feedback_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return _read_jsonl(ROUTER_FEEDBACK_FILE, limit or _max_router_eval_cases())


class KafkaSpoolRouterObservabilitySink:
    """Kafka 接入前的本地 spool sink。

    中文注释：
    大厂通常不会让业务进程直接散乱地写日志文件，而是发到 Kafka /
    PubSub / EventBus，再由消费者写入数据仓库、监控系统、审计系统。

    当前项目先不引入 Kafka 依赖，避免为了一个教学环境增加部署复杂度。
    这个 sink 会把“应该发往 Kafka 的事件”写入 spool 文件。
    后续可以增加独立 producer/sidecar：

        router_kafka_spool.jsonl -> Kafka topic -> 消费者

    这就是工程上常见的渐进式落地：接口先稳定，传输层后替换。
    """

    def append_event(self, event: RouterEvent) -> None:
        records = _read_jsonl(ROUTER_KAFKA_SPOOL_FILE, _max_router_events())
        payload = event.as_dict()
        payload["_sink"] = "kafka_spool"
        payload["_topic"] = os.getenv(
            "BEGINNER_AGENT_ROUTER_KAFKA_TOPIC",
            "beginner_agent.router_events",
        )
        records.append(payload)
        _write_jsonl(ROUTER_KAFKA_SPOOL_FILE, records[-_max_router_events():])

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return JsonlRouterObservabilitySink().read_events(limit)

    def append_eval_case(self, case: RouterEvalCase) -> None:
        JsonlRouterObservabilitySink().append_eval_case(case)

    def read_eval_cases(self, limit: int | None = None) -> list[dict[str, Any]]:
        return JsonlRouterObservabilitySink().read_eval_cases(limit)

    def append_feedback_event(self, event: dict[str, Any]) -> None:
        JsonlRouterObservabilitySink().append_feedback_event(event)

    def read_feedback_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return JsonlRouterObservabilitySink().read_feedback_events(limit)


def resolve_router_observability_sink() -> RouterObservabilitySink:
    """根据 env 选择 Router observability sink。"""

    if not router_observability_enabled():
        return NullRouterObservabilitySink()

    sink_name = router_observability_sink_name()
    if sink_name == "null":
        return NullRouterObservabilitySink()
    if sink_name == "kafka_spool":
        return KafkaSpoolRouterObservabilitySink()
    return JsonlRouterObservabilitySink()
