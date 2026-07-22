from __future__ import annotations

from typing import Any

from .models import RouterEvalCase, RouterEvent
from .sinks import (
    ROUTER_DIR,
    ROUTER_EVAL_CASES_FILE,
    ROUTER_EVENTS_FILE,
    ROUTER_FEEDBACK_FILE,
    ROUTER_KAFKA_SPOOL_FILE,
    resolve_router_observability_sink,
)


# 中文注释：
# observability.py 现在只保留 Router 观测的稳定入口。
#
# 你可以把它理解成“业务 API 层”：
#
#     router.py 调用 append_router_event(...)
#       -> observability.py 选择当前 sink
#       -> sinks.py 负责具体写入 JSONL / Null / Kafka spool
#
# 这样以后要接 Kafka、OpenTelemetry、HTTP Collector，
# 不需要修改 router.py，只需要新增一个 sink 实现。


def append_router_event(event: RouterEvent) -> None:
    """追加 Router 观测事件。"""

    resolve_router_observability_sink().append_event(event)


def read_router_events(limit: int | None = None) -> list[dict[str, Any]]:
    """读取 Router 观测事件。

    中文注释：
    人工反馈闭环需要先找到“哪一次 Router 决策错了”。
    所以这里提供稳定读取入口，而不是让 CLI/API 直接读 JSONL 文件。
    """

    return resolve_router_observability_sink().read_events(limit)


def append_router_eval_case(case: RouterEvalCase) -> None:
    """追加 Router eval case。"""

    resolve_router_observability_sink().append_eval_case(case)


def read_router_eval_cases(limit: int | None = None) -> list[dict[str, Any]]:
    """读取 Router eval case。"""

    return resolve_router_observability_sink().read_eval_cases(limit)


def append_router_feedback_event(event: dict[str, Any]) -> None:
    """追加 Router 人工纠错反馈事件。"""

    resolve_router_observability_sink().append_feedback_event(event)


def read_router_feedback_events(limit: int | None = None) -> list[dict[str, Any]]:
    """读取 Router 人工纠错反馈事件。"""

    return resolve_router_observability_sink().read_feedback_events(limit)


__all__ = [
    "ROUTER_DIR",
    "ROUTER_EVAL_CASES_FILE",
    "ROUTER_EVENTS_FILE",
    "ROUTER_FEEDBACK_FILE",
    "ROUTER_KAFKA_SPOOL_FILE",
    "append_router_eval_case",
    "append_router_event",
    "append_router_feedback_event",
    "read_router_events",
    "read_router_eval_cases",
    "read_router_feedback_events",
]
