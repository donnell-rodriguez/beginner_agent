from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .eval_cases import evaluate_retrieval_case, read_memory_eval_cases
from .settings import MEMORY_DIR
from ..state import State
from ..tooling.core import ensure_state_dirs


# 中文注释：
# online_eval.py 负责 retrieval online eval。
#
# 离线 eval 是“拿固定样本回放”。
# 在线 eval 是“每次真实检索后，把结果和已知 eval case 对照，持续记录质量”。
#
# 这让你后续能回答：
# - 最近检索命中率有没有下降？
# - 哪些 query 总是漏召回？
# - reranker candidate bucket 是否更好？


MEMORY_ONLINE_EVAL_FILE = MEMORY_DIR / "memory_online_eval.jsonl"
MAX_MEMORY_ONLINE_EVAL_EVENTS = 3000


@dataclass(frozen=True)
class MemoryOnlineEvalEvent:
    """一次 Memory Retriever 的在线评估事件。"""

    run_id: str
    task_id: str
    query: str
    returned_ids: list[str]
    eval_matches: list[dict[str, Any]]
    backend: str
    backend_error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        passed = sum(1 for item in self.eval_matches if item.get("passed") is True)
        total = len(self.eval_matches)
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "query": self.query,
            "returned_ids": self.returned_ids,
            "eval_matches": self.eval_matches,
            "matched_case_count": total,
            "passed_case_count": passed,
            "pass_rate": passed / total if total else 0.0,
            "backend": self.backend,
            "backend_error": self.backend_error,
            "created_at": self.created_at,
        }


def append_online_eval_event(event: MemoryOnlineEvalEvent) -> None:
    """追加 online eval event。"""

    _ensure_online_eval_file()
    events = read_online_eval_events(MAX_MEMORY_ONLINE_EVAL_EVENTS)
    events.append(event.as_dict())
    MEMORY_ONLINE_EVAL_FILE.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in events[-MAX_MEMORY_ONLINE_EVAL_EVENTS:]
        ),
        encoding="utf-8",
    )


def record_retrieval_online_eval(
    state: State,
    returned: list[dict[str, Any]],
    *,
    backend: str,
    backend_error: str,
) -> None:
    """根据当前真实检索结果记录 online eval。"""

    query = str(state.get("user_input", ""))
    returned_ids = [str(record.get("id", "")) for record in returned if record.get("id")]
    eval_matches = [
        evaluate_retrieval_case(case, returned)
        for case in read_memory_eval_cases(50)
        if str(case.get("query", "")).strip()
        and str(case.get("query", "")).lower() in query.lower()
    ]
    append_online_eval_event(
        MemoryOnlineEvalEvent(
            run_id=str(state.get("run_id", "")),
            task_id=str(state.get("current_task_id", "")),
            query=query,
            returned_ids=returned_ids,
            eval_matches=eval_matches[:10],
            backend=backend,
            backend_error=backend_error,
        )
    )


def read_online_eval_events(limit: int = MAX_MEMORY_ONLINE_EVAL_EVENTS) -> list[dict[str, Any]]:
    """读取 online eval events。"""

    _ensure_online_eval_file()
    events: list[dict[str, Any]] = []
    for line in MEMORY_ONLINE_EVAL_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events[-limit:]


def summarize_online_eval(limit: int = 500) -> dict[str, Any]:
    """汇总最近 online eval 质量。"""

    events = read_online_eval_events(limit)
    matched = [event for event in events if int(event.get("matched_case_count", 0) or 0) > 0]
    total_cases = sum(int(event.get("matched_case_count", 0) or 0) for event in matched)
    passed_cases = sum(int(event.get("passed_case_count", 0) or 0) for event in matched)
    return {
        "event_count": len(events),
        "matched_event_count": len(matched),
        "matched_case_count": total_cases,
        "passed_case_count": passed_cases,
        "pass_rate": passed_cases / total_cases if total_cases else 0.0,
        "recent_events": events[-20:],
    }


def _ensure_online_eval_file() -> None:
    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_ONLINE_EVAL_FILE.exists():
        MEMORY_ONLINE_EVAL_FILE.write_text("", encoding="utf-8")
