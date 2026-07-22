from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..config import load_project_env
from ..tooling.core import STATE_DIR, ensure_state_dirs
from .models import RouterEvalCase, RouterEvent


ROUTER_DIR = STATE_DIR / "router"
ROUTER_EVENTS_FILE = ROUTER_DIR / "router_events.jsonl"
ROUTER_EVAL_CASES_FILE = ROUTER_DIR / "router_eval_cases.jsonl"
DEFAULT_MAX_ROUTER_EVENTS = 2000
DEFAULT_MAX_ROUTER_EVAL_CASES = 1000


def _router_observability_enabled() -> bool:
    load_project_env()
    raw = os.getenv("BEGINNER_AGENT_ROUTER_OBSERVABILITY_ENABLED", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def append_router_event(event: RouterEvent) -> None:
    """追加 Router 观测事件。"""

    if not _router_observability_enabled():
        return
    records = _read_jsonl(ROUTER_EVENTS_FILE, _max_router_events())
    records.append(event.as_dict())
    _write_jsonl(ROUTER_EVENTS_FILE, records[-_max_router_events():])


def append_router_eval_case(case: RouterEvalCase) -> None:
    """追加 Router eval case。"""

    records = _read_jsonl(ROUTER_EVAL_CASES_FILE, _max_router_eval_cases())
    records.append(case.as_dict())
    _write_jsonl(ROUTER_EVAL_CASES_FILE, records[-_max_router_eval_cases():])


def read_router_eval_cases(limit: int | None = None) -> list[dict[str, Any]]:
    return _read_jsonl(ROUTER_EVAL_CASES_FILE, limit or _max_router_eval_cases())
