from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .settings import (
    MAX_MEMORY_EVAL_CASES,
    MEMORY_DIR,
    MEMORY_EVAL_CASES_FILE,
)
from ..tooling.core import ensure_state_dirs


@dataclass(frozen=True)
class MemoryEvalCase:
    """离线 eval case。

    中文注释：
    大厂通常会把线上真实问题沉淀成离线评测集。
    这样每次改 reranker / memory policy 后，可以回放这些 case，
    看命中率、误召回、漏召回有没有变差。
    """

    query: str
    expected_memory_ids: list[str]
    negative_memory_ids: list[str]
    source_run_id: str = ""
    reason: str = ""
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expected_memory_ids": self.expected_memory_ids,
            "negative_memory_ids": self.negative_memory_ids,
            "source_run_id": self.source_run_id,
            "reason": self.reason,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }


def _ensure_eval_file() -> None:
    ensure_state_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_EVAL_CASES_FILE.exists():
        MEMORY_EVAL_CASES_FILE.write_text("", encoding="utf-8")


def append_memory_eval_case(case: MemoryEvalCase) -> None:
    """追加一条离线 eval case。"""

    _ensure_eval_file()
    cases = read_memory_eval_cases(MAX_MEMORY_EVAL_CASES)
    cases.append(case.as_dict())
    trimmed = cases[-MAX_MEMORY_EVAL_CASES:]
    MEMORY_EVAL_CASES_FILE.write_text(
        "".join(
            f"{json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            for item in trimmed
        ),
        encoding="utf-8",
    )


def read_memory_eval_cases(limit: int = MAX_MEMORY_EVAL_CASES) -> list[dict[str, Any]]:
    """读取离线 eval cases。"""

    _ensure_eval_file()
    cases: list[dict[str, Any]] = []
    for line in MEMORY_EVAL_CASES_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            cases.append(data)
    return cases[-limit:]


def evaluate_retrieval_case(
    case: dict[str, Any],
    retrieved_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """计算单个 retrieval eval case 的命中情况。"""

    retrieved_ids = [str(record.get("id", "")) for record in retrieved_records]
    expected = [str(item) for item in case.get("expected_memory_ids", [])]
    negative = [str(item) for item in case.get("negative_memory_ids", [])]
    hit_ids = [item for item in expected if item in retrieved_ids]
    false_positive_ids = [item for item in negative if item in retrieved_ids]
    recall = len(hit_ids) / max(1, len(expected))
    precision_guard = 1.0 - (len(false_positive_ids) / max(1, len(retrieved_ids)))
    passed = bool(expected and len(hit_ids) == len(expected) and not false_positive_ids)
    return {
        "query": case.get("query", ""),
        "retrieved_ids": retrieved_ids,
        "expected_ids": expected,
        "negative_ids": negative,
        "hit_ids": hit_ids,
        "false_positive_ids": false_positive_ids,
        "recall": round(recall, 4),
        "precision_guard": round(max(0.0, precision_guard), 4),
        "passed": passed,
    }
