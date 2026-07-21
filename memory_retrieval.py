from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .failure_memory import failure_rerank_signal
from .memory_audit import _build_audit_event
from .memory_jsonl_store import JsonlMemoryStore
from .memory_models import MemoryRecord
from .memory_policy import (
    _dedupe_contradiction_records,
    _memory_access_context,
    _preference_memory_records_for_context,
    _preference_records_for_state,
    _record_access_control,
    _record_allowed_in_prompt,
    _record_created_at,
    _record_is_active,
    _record_visible_to_context,
    _scope_matches_state,
)
from .memory_settings import MAX_MEMORY_RECORDS, MAX_RERANK_CANDIDATES, MAX_RETRIEVED_RECORDS, MIN_RERANK_SCORE
from .memory_store import _configured_store, _upsert_memory_audit_event, _upsert_memory_record
from .preference_memory import default_preference_payloads, preference_rerank_signal
from .privacy_governance import memory_prompt_allowed_by_privacy
from .state import State

def _query_text_for_state(state: State) -> str:
    """构造 Memory Retriever 的向量查询文本。"""

    current_task = state["task_tree"].get(state["current_task_id"], {})
    return "\n".join(
        [
            f"user_goal: {state['user_input']}",
            f"current_task: {current_task.get('title', '')}",
            f"tool_name: {state.get('tool_name', 'none')}",
            f"tool_result_status: {state.get('tool_result_status', 'none')}",
        ]
    )

def _seed_default_preference_memories(state: State) -> dict[str, Any]:
    """把默认偏好写入长期 memory store。

    中文注释：
    这里使用稳定 id。
    只有缺失或内容变化时才 upsert，避免每次运行都重写 Postgres / pgvector。
    """

    records = _preference_memory_records_for_context(state)
    existing_records, list_backend, list_error = _list_memory_records()
    existing_by_id = {str(record.get("id", "")): record for record in existing_records}
    written = 0
    skipped = 0
    backend = list_backend
    errors: list[str] = []
    if list_error:
        errors.append(list_error)
    for record in records:
        existing = existing_by_id.get(record.id)
        existing_metadata = existing.get("metadata", {}) if existing else {}
        existing_preference = (
            existing_metadata.get("preference_memory", {})
            if isinstance(existing_metadata, dict)
            else {}
        )
        current_preference = record.metadata.get("preference_memory", {})
        if existing and existing_preference == current_preference:
            skipped += 1
            continue
        backend, error, _ = _upsert_memory_record(record)
        written += 1
        if error:
            errors.append(error)
    return {
        "seeded": written,
        "skipped": skipped,
        "backend": backend,
        "errors": errors[:3],
    }

def _audit_sensitive_memory_access(
    state: State,
    records: list[dict[str, Any]],
    *,
    backend: str,
) -> None:
    """审计敏感 memory 的检索访问。

    中文注释：
    大厂级隐私治理不只关心“有没有脱敏”，还要知道：
    - 谁访问过敏感记忆。
    - 是哪个 project/user/thread 触发的访问。
    - 这条 memory 有没有被允许进入 prompt。

    这里不会记录原始敏感内容，只记录 memory id 和访问控制结果。
    """

    context = _memory_access_context(state)
    for record in records:
        access_control = record.get("access_control", {})
        sensitivity = str(record.get("sensitivity_level", "internal"))
        prompt_allowed = bool(access_control.get("prompt_allowed", False))
        if sensitivity in {"public", "internal"} and prompt_allowed:
            continue
        _upsert_memory_audit_event(
            _build_audit_event(
                action="sensitive_access",
                memory_id=str(record.get("id", "")),
                reason=(
                    "Memory Retriever 命中敏感或 prompt 禁用记忆，"
                    "记录访问审计。"
                ),
                backend=backend,
                metadata={
                    "run_id": state["run_id"],
                    "access_context": context,
                    "access_control": _safe_memory_value(access_control),
                    "sensitivity_level": sensitivity,
                    "prompt_allowed": prompt_allowed,
                    "retrieval_source": record.get("retrieval_source", ""),
                    "task_id": state.get("current_task_id", ""),
                },
            )
        )

def _list_memory_records() -> tuple[list[dict[str, Any]], str, str]:
    """读取 memory records，并返回 backend 信息。

    中文注释：
    如果配置了 Postgres 但连接失败，不让整个 agent 崩掉。
    它会回退到 JSONL，并把错误原因写进 memory_context。
    """

    try:
        store = _configured_store()
        records = [
            record
            for record in store.list_records(MAX_MEMORY_RECORDS)
            if _record_is_active(record)
        ]
        records = _dedupe_contradiction_records(records)
        return records, store.backend_name, ""
    except Exception as exc:
        fallback = JsonlMemoryStore()
        records = [
            record
            for record in fallback.list_records(MAX_MEMORY_RECORDS)
            if _record_is_active(record)
        ]
        records = _dedupe_contradiction_records(records)
        return records, fallback.backend_name, str(exc)


def _search_vector_records(query_text: str) -> tuple[list[dict[str, Any]], str, str]:
    """执行向量检索，并返回 backend 信息。"""

    try:
        store = _configured_store()
        records = store.search_similar_records(query_text, MAX_RETRIEVED_RECORDS)
        return records, store.backend_name, ""
    except Exception as exc:
        return [], "jsonl-fallback", str(exc)

def _score_record(record: dict[str, Any], state: State) -> int:
    """给一条历史记忆打分，分数越高越相关。

    中文注释：
    这是 hybrid retrieval 里的“规则打分”部分。
    它和 pgvector 语义检索一起工作：
    - 规则分数适合处理路径、工具名、状态、关键词。
    - 向量检索适合处理“意思相近但词不一样”的经验。
    - 后续 TODO 里的 reranker 可以在两者之后做最终排序。
    """

    query = state["user_input"].lower()
    current_task = state["task_tree"].get(state["current_task_id"], {})
    task_text = str(current_task.get("title", "")).lower()
    score = 0
    haystack = " ".join(
        [
            str(record.get("title", "")),
            str(record.get("summary", "")),
            str(record.get("tool_name", "")),
            " ".join(str(tag) for tag in record.get("tags", [])),
            " ".join(str(path) for path in record.get("paths", [])),
        ]
    ).lower()
    for token in set(query.replace("/", " ").replace("_", " ").split()):
        if len(token) >= 2 and token in haystack:
            score += 2
    for token in set(task_text.replace("/", " ").replace("_", " ").split()):
        if len(token) >= 2 and token in haystack:
            score += 3
    if record.get("kind") == "failure":
        score += 1
    if record.get("tool_result_status") == "success":
        score += 1
    if record.get("pinned"):
        score += 5
    score += int(float(record.get("importance", 0.5)) * 4)
    score += int(float(record.get("confidence", 0.7)) * 2)
    return score


def _float_record_value(record: dict[str, Any], key: str, default: float) -> float:
    """安全读取 record 里的数字字段。"""

    try:
        return float(record.get(key, default))
    except (TypeError, ValueError):
        return default


def _tokenize_for_rerank(text: str) -> set[str]:
    """把文本切成适合轻量 reranker 使用的 token 集合。"""

    normalized = re.sub(r"[^0-9A-Za-z_\-/.\u4e00-\u9fff]+", " ", text.lower())
    return {token for token in normalized.split() if len(token) >= 2}


def _rerank_query_text(state: State) -> str:
    """构造 reranker 使用的任务文本。"""

    current_task = state["task_tree"].get(state["current_task_id"], {})
    task_title = str(current_task.get("title", "")) if isinstance(current_task, dict) else ""
    return "\n".join(
        [
            str(state.get("user_input", "")),
            task_title,
            str(state.get("tool_name", "none")),
            str(state.get("tool_result_status", "none")),
        ]
    )


def _record_rerank_text(record: dict[str, Any]) -> str:
    """构造 reranker 使用的记忆文本。"""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    failure_profile = metadata.get("failure_memory")
    failure_profile = failure_profile if isinstance(failure_profile, dict) else {}
    preference = metadata.get("preference_memory")
    preference = preference if isinstance(preference, dict) else {}
    return "\n".join(
        [
            str(record.get("title", "")),
            str(record.get("summary", "")),
            str(record.get("tool_name", "")),
            str(record.get("tool_result_status", "")),
            str(failure_profile.get("category", "")),
            str(failure_profile.get("owner", "")),
            str(failure_profile.get("retry_class", "")),
            str(failure_profile.get("stack_signature", "")),
            str(failure_profile.get("recommendation", "")),
            str(preference.get("key", "")),
            str(preference.get("value", "")),
            str(preference.get("category", "")),
            " ".join(str(tag) for tag in record.get("tags", [])),
            " ".join(str(path) for path in record.get("paths", [])),
        ]
    )


def _recency_score(record: dict[str, Any]) -> float:
    """越新的记忆分数越高，pinned/long_term 不被强烈惩罚。"""

    if record.get("pinned") or record.get("retention_policy") in {"pinned", "long_term"}:
        return 1.0
    created_at = _record_created_at(record)
    age_days = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 86400)
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.75
    if age_days <= 90:
        return 0.45
    return 0.2


def _path_overlap_score(record: dict[str, Any], state: State) -> float:
    """判断记忆路径和当前任务路径是否重合。"""

    current_task = state["task_tree"].get(state["current_task_id"], {})
    args = current_task.get("args", {}) if isinstance(current_task, dict) else {}
    current_path = str(args.get("path", "")) if isinstance(args, dict) else ""
    if not current_path:
        return 0.0
    paths = {str(path) for path in record.get("paths", [])}
    if current_path in paths:
        return 1.0
    current_parts = set(Path(current_path).parts)
    for path in paths:
        if current_parts.intersection(Path(path).parts):
            return 0.5
    return 0.0


def _reliability_score(record: dict[str, Any]) -> float:
    """评估这条记忆是否可信。"""

    confidence = _float_record_value(record, "confidence", 0.7)
    importance = _float_record_value(record, "importance", 0.5)
    quality = _float_record_value(record, "quality_score", 0.5)
    trust = _float_record_value(record, "trust_score", 0.5)
    decay = _float_record_value(record, "decay_score", 0.0)
    status_bonus = 0.2 if record.get("tool_result_status") == "success" else 0.0
    pinned_bonus = 0.25 if record.get("pinned") else 0.0
    failure_penalty = 0.15 if record.get("kind") == "failure" else 0.0
    score = (
        (confidence * 0.25)
        + (importance * 0.20)
        + (quality * 0.20)
        + (trust * 0.20)
        + status_bonus
        + pinned_bonus
        - failure_penalty
        - (decay * 0.15)
    )
    return max(0.0, min(1.0, score))


def _misleading_risk_score(record: dict[str, Any], state: State) -> float:
    """估计记忆误导当前任务的风险，分数越高风险越大。"""

    if record.get("pinned"):
        return 0.0
    risk = 0.0
    if record.get("kind") == "failure" and state.get("tool_result_status") == "success":
        risk += 0.25
    if record.get("validity_status", "active") != "active":
        risk += 1.0
    if _recency_score(record) <= 0.2:
        risk += 0.2
    if (
        record.get("tool_name") not in {state.get("tool_name"), "none", ""}
        and _path_overlap_score(record, state) == 0
    ):
        risk += 0.15
    return min(1.0, risk)


def _semantic_score(record: dict[str, Any]) -> float:
    """把 pgvector distance 转成 0..1 的相似度分数。"""

    if "vector_distance" not in record:
        return 0.0
    distance = _float_record_value(record, "vector_distance", 1.0)
    return max(0.0, min(1.0, 1.0 - distance))


def _rerank_memory_candidates(
    records: list[dict[str, Any]],
    state: State,
) -> list[dict[str, Any]]:
    """对召回后的记忆做任务感知重排。

    中文注释：
    这一步对应生产系统里的 MemoryReranker。
    它不负责“能不能访问”，访问边界仍由 scope / validity / TTL 硬规则控制。
    它负责回答：
    - 这条记忆对当前任务是否真的有帮助？
    - 这条记忆可靠性高不高？
    - 这条记忆会不会误导当前任务？
    - 是否应该进入 memory_context？

    当前实现是稳定的本地特征 reranker。
    后续可以把它替换成 cross-encoder / LLM judge。
    """

    query_tokens = _tokenize_for_rerank(_rerank_query_text(state))
    reranked: list[dict[str, Any]] = []
    for record in records[:MAX_RERANK_CANDIDATES]:
        record_tokens = _tokenize_for_rerank(_record_rerank_text(record))
        token_overlap = (
            len(query_tokens.intersection(record_tokens)) / max(1, len(query_tokens))
        )
        semantic = _semantic_score(record)
        rule_score = _float_record_value(record, "rule_score", 0.0)
        normalized_rule = min(1.0, rule_score / 12.0)
        reliability = _reliability_score(record)
        recency = _recency_score(record)
        path_overlap = _path_overlap_score(record, state)
        risk = _misleading_risk_score(record, state)
        failure_signal = failure_rerank_signal(record)
        failure_weight = float(failure_signal.get("failure_weight", 0.0))
        preference_signal = preference_rerank_signal(record)
        preference_weight = float(preference_signal.get("preference_weight", 0.0))
        pinned = 1.0 if record.get("pinned") else 0.0

        score = (
            semantic * 0.28
            + normalized_rule * 0.20
            + token_overlap * 0.18
            + reliability * 0.16
            + recency * 0.08
            + path_overlap * 0.07
            + failure_weight * 0.08
            + preference_weight * 0.07
            + pinned * 0.08
            - risk * 0.20
        )
        score = max(0.0, min(1.0, score))
        decision = "include" if score >= MIN_RERANK_SCORE or record.get("pinned") else "drop"
        reranked.append(
            {
                **record,
                "rerank_score": round(score, 4),
                "rerank_decision": decision,
                "rerank_reason": (
                    "task-aware reranker: semantic/rule/token/reliability/"
                    "recency/path/risk weighted score"
                ),
                "rerank_features": {
                    "semantic": round(semantic, 4),
                    "rule": round(normalized_rule, 4),
                    "token_overlap": round(token_overlap, 4),
                    "reliability": round(reliability, 4),
                    "quality_score": round(
                        _float_record_value(record, "quality_score", 0.5),
                        4,
                    ),
                    "trust_score": round(
                        _float_record_value(record, "trust_score", 0.5),
                        4,
                    ),
                    "decay_score": round(
                        _float_record_value(record, "decay_score", 0.0),
                        4,
                    ),
                    "recency": round(recency, 4),
                    "path_overlap": round(path_overlap, 4),
                    "misleading_risk": round(risk, 4),
                    "failure_memory": failure_signal,
                    "preference_memory": preference_signal,
                    "pinned": bool(record.get("pinned")),
                },
            }
        )
    included = [record for record in reranked if record["rerank_decision"] == "include"]
    included.sort(key=lambda record: float(record.get("rerank_score", 0)), reverse=True)
    return included[:MAX_RETRIEVED_RECORDS]


def _retrieve_relevant_records(state: State) -> tuple[list[dict[str, Any]], str, str]:
    """检索和当前目标最相关的历史记忆。"""

    records, backend, backend_error = _list_memory_records()
    query_text = _query_text_for_state(state)
    vector_records, vector_backend, vector_error = _search_vector_records(query_text)
    records = [
        {
            **record,
            "access_control": _record_access_control(record, state),
        }
        for record in records
        if _record_visible_to_context(record, state) and _scope_matches_state(record, state)
    ]
    vector_records = [
        {
            **record,
            "access_control": _record_access_control(record, state),
        }
        for record in vector_records
        if _record_is_active(record)
        and _record_visible_to_context(record, state)
        and _scope_matches_state(record, state)
    ]
    vector_records = _dedupe_contradiction_records(vector_records)
    scored = [(record, _score_record(record, state)) for record in records]
    relevant = [item for item in scored if item[1] > 0]
    relevant.sort(key=lambda item: item[1], reverse=True)
    merged: dict[str, dict[str, Any]] = {}
    for record in vector_records:
        merged[str(record.get("id", ""))] = {
            **record,
            "retrieval_source": "vector",
            "retrieval_reason": "pgvector 相似度召回。",
            "retrieval_score": (
                1.0
                - float(record.get("vector_distance", 1.0))
                + float(record.get("importance", 0.5))
            ),
        }
    for record, score in relevant:
        record_id = str(record.get("id", ""))
        if record_id in merged:
            merged[record_id]["retrieval_source"] = "hybrid"
            merged[record_id]["rule_score"] = score
            merged[record_id]["retrieval_reason"] = "规则关键词 + pgvector 混合召回。"
            previous_score = float(merged[record_id].get("retrieval_score", 0))
            merged[record_id]["retrieval_score"] = previous_score + (score / 10)
        else:
            merged[record_id] = {
                **record,
                "retrieval_source": "rule",
                "retrieval_reason": "规则关键词召回。",
                "rule_score": score,
                "retrieval_score": score / 10,
            }
    candidates = sorted(
        merged.values(),
        key=lambda record: float(record.get("retrieval_score", 0)),
        reverse=True,
    )
    reranked = _rerank_memory_candidates(candidates, state)
    _audit_sensitive_memory_access(state, reranked, backend=backend)
    results = [
        record
        for record in reranked
        if record.get("access_control", {}).get("prompt_allowed", False)
    ]
    errors = "; ".join(error for error in (backend_error, vector_error) if error)
    if vector_records:
        backend = f"{backend}+vector"
    elif vector_error:
        backend = f"{backend}; vector_backend={vector_backend}"
    return results, backend, errors
