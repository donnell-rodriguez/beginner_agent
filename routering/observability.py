from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)
_last_router_event_error: str = ""


def append_router_event(event: RouterEvent) -> None:
    """追加 Router 观测事件。

    中文注释：
    这是 Router 主路径的旁路写入。

    生产级系统里，Router 做出的业务决策不能因为日志/Kafka/JSONL
    写入失败而失败。否则会出现很糟糕的问题：

        模型分类成功
          -> 只是观测系统写失败
          -> 整个用户请求失败

    所以这里会吞掉 observability sink 的异常，并记录 warning。
    这样“可观测性失败”不会拖垮 Router 主流程。
    """

    global _last_router_event_error
    try:
        resolve_router_observability_sink().append_event(event)
        _last_router_event_error = ""
    except Exception as exc:  # noqa: BLE001
        _last_router_event_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Router observability append_event failed; router main path continues.",
            exc_info=True,
        )


def last_router_event_error() -> str:
    """返回最近一次 Router event 旁路写入错误。

    中文注释：
    append_router_event(...) 不会抛错，
    所以如果你想排查 observability 是否失败，
    可以读取这个轻量状态。

    这不是长期监控系统，只是本地项目里的最小健康信号。
    后续接 OpenTelemetry/Kafka 时，可以把这个状态上报成 metric。
    """

    return _last_router_event_error


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
    "last_router_event_error",
    "read_router_events",
    "read_router_eval_cases",
    "read_router_feedback_events",
]
