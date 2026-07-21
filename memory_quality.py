from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


MemoryQualityDecision = Literal["store", "deprioritize", "reject"]

EVIDENCE_KEYS = {
    "tests_passed",
    "build_passed",
    "verification_passed",
    "lint_passed",
    "typecheck_passed",
    "human_confirmed",
    "approval",
}

GENERIC_WORDS = {
    "done",
    "success",
    "failed",
    "task",
    "处理完成",
    "完成",
    "失败",
    "成功",
    "问题",
    "任务",
}


@dataclass(frozen=True)
class MemoryQualityScore:
    """MemoryQualityScore：评价一条记忆本身的内容质量。

    中文注释：
    大厂风格的 memory 不能“写进去就相信”。
    这层专门评估：
    - accurate：是否看起来准确。
    - fresh：是否新鲜，是否可能过期。
    - unique：是否重复。
    - specific：是否足够具体。
    - actionable：后续 agent 能不能根据它行动。
    - evidenced：有没有测试、构建、人工确认等证据。
    - proven_effective：是否被成功结果证明有效。
    """

    accurate: float
    fresh: float
    unique: float
    specific: float
    actionable: float
    evidenced: float
    proven_effective: float
    overall: float
    decision: MemoryQualityDecision
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """转成可以写入 JSON / Postgres JSONB 的普通 dict。"""

        return {
            "accurate": self.accurate,
            "fresh": self.fresh,
            "unique": self.unique,
            "specific": self.specific,
            "actionable": self.actionable,
            "evidenced": self.evidenced,
            "proven_effective": self.proven_effective,
            "overall": self.overall,
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MemoryDecay:
    """MemoryDecay：评价一条记忆是否正在过期。

    中文注释：
    不是所有记忆都应该永久有效。
    例如一次失败日志、某个临时文件路径，过几天可能就没价值了。
    decay_score 越高，说明越应该降低检索权重或设置 TTL。
    """

    age_days: float
    decay_score: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "age_days": self.age_days,
            "decay_score": self.decay_score,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MemoryTrustScore:
    """MemoryTrustScore：评价一条记忆可不可信。

    中文注释：
    quality 偏内容质量，trust 偏“这条记忆能不能被相信”。
    例如：
    - 有测试通过，trust 更高。
    - 是失败经验，但没有失败日志，trust 更低。
    - 被人工确认，trust 更高。
    """

    trust_score: float
    signals: tuple[str, ...]
    penalties: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "trust_score": self.trust_score,
            "signals": list(self.signals),
            "penalties": list(self.penalties),
        }


@dataclass(frozen=True)
class MemoryEvaluation:
    """MemoryEvaluator 的完整评估结果。"""

    quality: MemoryQualityScore
    decay: MemoryDecay
    trust: MemoryTrustScore

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality.as_dict(),
            "decay": self.decay.as_dict(),
            "trust": self.trust.as_dict(),
        }


def _clamp(value: float) -> float:
    """把分数限制在 0 到 1 之间。"""

    return max(0.0, min(1.0, round(value, 4)))


def _text(record: dict[str, Any]) -> str:
    """拼出用于质量评估的主要文本。"""

    return "\n".join(
        [
            str(record.get("title", "")),
            str(record.get("summary", "")),
            str(record.get("status", "")),
            str(record.get("tool_name", "")),
            str(record.get("tool_result_status", "")),
            " ".join(str(tag) for tag in record.get("tags", [])),
            " ".join(str(path) for path in record.get("paths", [])),
        ]
    ).strip()


def _tokens(text: str) -> set[str]:
    """轻量 token 化，用于重复检测和泛化检测。"""

    normalized = (
        text.lower()
        .replace("/", " ")
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .replace("|", " ")
    )
    return {token for token in normalized.split() if len(token) >= 2}


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    """安全读取 metadata。"""

    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _source_memory(record: dict[str, Any]) -> dict[str, Any]:
    """读取原始 pending_memory。"""

    source = _metadata(record).get("source_memory")
    return source if isinstance(source, dict) else {}


def _tool_result_data(record: dict[str, Any]) -> dict[str, Any]:
    """读取工具执行结果。"""

    data = _metadata(record).get("tool_result_data")
    return data if isinstance(data, dict) else {}


def _has_evidence(record: dict[str, Any]) -> bool:
    """判断是否有可验证证据。"""

    data = _tool_result_data(record)
    source = _source_memory(record)
    metadata = _metadata(record)
    for container in (data, source, metadata):
        if any(bool(container.get(key)) for key in EVIDENCE_KEYS):
            return True
    return False


def _has_actionable_detail(record: dict[str, Any]) -> bool:
    """判断这条记忆后续是否能指导 agent 行动。"""

    paths = record.get("paths", [])
    tags = record.get("tags", [])
    summary = str(record.get("summary", ""))
    source = _source_memory(record)
    has_path = isinstance(paths, list) and bool(paths)
    has_tags = isinstance(tags, list) and bool(tags)
    has_reason = bool(str(source.get("reason", "")).strip())
    has_tool = str(record.get("tool_name", "none")) != "none"
    return has_path or has_tags or has_reason or has_tool or len(summary) >= 80


def _specificity_score(record: dict[str, Any]) -> float:
    """评估记忆是否具体。"""

    text = _text(record)
    tokens = _tokens(text)
    if not text:
        return 0.0
    generic_count = len(tokens.intersection(GENERIC_WORDS))
    paths = record.get("paths", [])
    path_bonus = 0.25 if isinstance(paths, list) and paths else 0.0
    length_score = min(0.55, len(text) / 260)
    token_score = min(0.25, len(tokens) / 40)
    generic_penalty = min(0.3, generic_count * 0.08)
    return _clamp(length_score + token_score + path_bonus - generic_penalty)


def _duplicate_score(record: dict[str, Any], existing_records: list[dict[str, Any]]) -> float:
    """评估是否重复；返回 unique 分数，越高越不重复。"""

    current_tokens = _tokens(_text(record))
    if not current_tokens:
        return 0.2
    current_id = str(record.get("id", ""))
    current_paths = {str(path) for path in record.get("paths", [])}
    max_overlap = 0.0
    for existing in existing_records:
        if str(existing.get("id", "")) == current_id:
            continue
        same_tool = str(existing.get("tool_name", "")) == str(record.get("tool_name", ""))
        same_status = str(existing.get("tool_result_status", "")) == str(
            record.get("tool_result_status", "")
        )
        existing_paths = {str(path) for path in existing.get("paths", [])}
        path_overlap = bool(current_paths and current_paths.intersection(existing_paths))
        existing_tokens = _tokens(_text(existing))
        token_overlap = len(current_tokens.intersection(existing_tokens)) / max(
            1,
            len(current_tokens),
        )
        if same_tool and same_status:
            token_overlap += 0.15
        if path_overlap:
            token_overlap += 0.2
        max_overlap = max(max_overlap, min(1.0, token_overlap))
    return _clamp(1.0 - max_overlap)


def _created_at(record: dict[str, Any]) -> datetime:
    """安全解析 created_at。"""

    raw = str(record.get("created_at") or "").strip()
    if not raw:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def evaluate_memory_decay(record: dict[str, Any]) -> MemoryDecay:
    """MemoryDecay：根据年龄和保留策略给出过期风险。"""

    if record.get("pinned") or record.get("retention_policy") in {"pinned", "long_term"}:
        return MemoryDecay(0.0, 0.0, "pinned/long_term 记忆不做强衰减。")

    age_days = max(
        0.0,
        (datetime.now(timezone.utc) - _created_at(record)).total_seconds() / 86400,
    )
    if record.get("kind") == "failure":
        base = 0.25
    elif record.get("scope") in {"task", "tool", "file"}:
        base = 0.18
    else:
        base = 0.1

    if age_days <= 7:
        age_penalty = 0.0
    elif age_days <= 30:
        age_penalty = 0.18
    elif age_days <= 90:
        age_penalty = 0.38
    else:
        age_penalty = 0.65

    score = _clamp(base + age_penalty)
    return MemoryDecay(
        round(age_days, 2),
        score,
        "根据记忆年龄、kind、scope 计算衰减风险。",
    )


def evaluate_memory_trust(record: dict[str, Any]) -> MemoryTrustScore:
    """MemoryTrustScore：结合证据、结果状态和人工确认计算可信度。"""

    signals: list[str] = []
    penalties: list[str] = []
    score = float(record.get("confidence", 0.7)) * 0.45
    score += float(record.get("importance", 0.5)) * 0.2

    status = str(record.get("tool_result_status", "none"))
    if status == "success":
        score += 0.18
        signals.append("tool_success")
    elif status in {"failed", "blocked", "empty"}:
        score -= 0.08
        penalties.append(f"tool_result_{status}")

    if _has_evidence(record):
        score += 0.18
        signals.append("has_verifiable_evidence")
    else:
        penalties.append("missing_verifiable_evidence")

    source = _source_memory(record)
    metadata = _metadata(record)
    if source.get("human_confirmed") or metadata.get("human_confirmed"):
        score += 0.15
        signals.append("human_confirmed")

    if record.get("validity_status") not in {None, "active"}:
        score -= 0.3
        penalties.append("not_active")

    if record.get("sensitivity_level") in {"confidential", "secret"}:
        score -= 0.05
        penalties.append("sensitive_memory")

    return MemoryTrustScore(_clamp(score), tuple(signals), tuple(penalties))


def evaluate_memory_quality(
    record: dict[str, Any],
    existing_records: list[dict[str, Any]],
) -> MemoryQualityScore:
    """MemoryQualityScore：评估准确性、重复度、泛化程度和证据质量。"""

    reasons: list[str] = []
    status = str(record.get("tool_result_status", "none"))
    has_evidence = _has_evidence(record)
    actionable = 0.85 if _has_actionable_detail(record) else 0.35
    specific = _specificity_score(record)
    unique = _duplicate_score(record, existing_records)
    evidenced = 1.0 if has_evidence else 0.35
    proven_effective = 1.0 if status == "success" and has_evidence else 0.55
    if status in {"failed", "blocked", "empty"}:
        proven_effective = 0.25

    accurate = 0.72
    if status == "success":
        accurate += 0.12
    if has_evidence:
        accurate += 0.12
    if not _source_memory(record).get("reason"):
        accurate -= 0.12
        reasons.append("缺少 reason，准确性证据偏弱。")
    if status in {"failed", "blocked"} and not _tool_result_data(record):
        accurate -= 0.15
        reasons.append("失败/阻塞记忆缺少工具结果细节。")

    fresh = 1.0 - evaluate_memory_decay(record).decay_score
    if unique < 0.35:
        reasons.append("疑似重复记忆，降低长期价值。")
    if specific < 0.45:
        reasons.append("内容偏泛，后续 agent 难以直接复用。")
    if actionable < 0.5:
        reasons.append("缺少路径、工具、标签或明确行动线索。")
    if evidenced < 0.5:
        reasons.append("缺少测试/build/人工确认等可验证证据。")

    overall = (
        _clamp(accurate) * 0.22
        + fresh * 0.10
        + unique * 0.16
        + specific * 0.16
        + actionable * 0.14
        + evidenced * 0.12
        + proven_effective * 0.10
    )
    overall = _clamp(overall)
    if overall < 0.35:
        decision: MemoryQualityDecision = "reject"
    elif overall < 0.58:
        decision = "deprioritize"
    else:
        decision = "store"
    if not reasons:
        reasons.append("记忆质量达到当前治理规则要求。")

    return MemoryQualityScore(
        accurate=_clamp(accurate),
        fresh=_clamp(fresh),
        unique=unique,
        specific=specific,
        actionable=_clamp(actionable),
        evidenced=_clamp(evidenced),
        proven_effective=_clamp(proven_effective),
        overall=overall,
        decision=decision,
        reasons=tuple(reasons),
    )


class MemoryEvaluator:
    """MemoryEvaluator：写入前的记忆质量评估器。

    中文注释：
    这不是 LLM 评审，而是本地、可重复、可审计的第一层质量门禁。
    后续如果要接 cross-encoder / LLM judge，可以在这个类后面加第二阶段。
    """

    def evaluate(
        self,
        record: dict[str, Any],
        existing_records: list[dict[str, Any]],
    ) -> MemoryEvaluation:
        """返回质量、衰减、可信度三类评分。"""

        quality = evaluate_memory_quality(record, existing_records)
        decay = evaluate_memory_decay(record)
        trust = evaluate_memory_trust(record)
        return MemoryEvaluation(quality=quality, decay=decay, trust=trust)


def adjusted_memory_fields(
    record: dict[str, Any],
    evaluation: MemoryEvaluation,
) -> dict[str, Any]:
    """根据评估结果给出 MemoryRecord 字段调整建议。"""

    quality = evaluation.quality
    trust = evaluation.trust
    decay = evaluation.decay
    current_confidence = float(record.get("confidence", 0.7))
    current_importance = float(record.get("importance", 0.5))
    confidence = _clamp((current_confidence * 0.55) + (trust.trust_score * 0.45))
    importance = _clamp(
        (current_importance * 0.55)
        + (quality.overall * 0.30)
        + ((1.0 - decay.decay_score) * 0.15)
    )
    updates: dict[str, Any] = {
        "quality_score": quality.overall,
        "trust_score": trust.trust_score,
        "decay_score": decay.decay_score,
        "confidence": confidence,
        "importance": importance,
    }
    if quality.decision == "reject":
        updates["validity_status"] = "rejected"
        updates["importance"] = min(importance, 0.2)
        updates["confidence"] = min(confidence, 0.35)
    elif quality.decision == "deprioritize":
        updates["importance"] = min(importance, 0.45)
    return updates


def summarize_quality_for_path(record: dict[str, Any]) -> str:
    """给 CLI / 后续 UI 一个简短的人类可读摘要。"""

    paths = record.get("paths", [])
    first_path = str(paths[0]) if isinstance(paths, list) and paths else ""
    if first_path:
        return f"{Path(first_path).name}: quality={record.get('quality_score', 0.5)}"
    return f"{record.get('id', '')}: quality={record.get('quality_score', 0.5)}"
