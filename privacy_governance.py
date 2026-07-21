from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Literal


SensitivityLevel = Literal["public", "internal", "confidential", "secret"]

MAX_PRIVACY_SCAN_TEXT_CHARS = 12000
MAX_REDACTED_TEXT_CHARS = 2000
PRIVACY_HASH_SALT_ENV = "BEGINNER_AGENT_PRIVACY_HASH_SALT"


@dataclass(frozen=True)
class PrivacyFinding:
    """隐私/密钥扫描发现项。

    中文注释：
    这里故意不保存原始敏感值，只保存 kind、位置和 hash。
    这样 audit / memory 可以复盘“发生了什么”，但不会泄露真正密钥。
    """

    kind: str
    category: Literal["secret", "pii"]
    fingerprint: str
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "category": self.category,
            "fingerprint": self.fingerprint,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class PrivacyScanReport:
    """隐私治理扫描报告。"""

    sensitivity_level: SensitivityLevel
    prompt_allowed: bool
    storage_mode: Literal["raw_allowed", "redacted", "hash_summary_only"]
    findings: list[PrivacyFinding]
    redacted_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensitivity_level": self.sensitivity_level,
            "prompt_allowed": self.prompt_allowed,
            "storage_mode": self.storage_mode,
            "finding_count": len(self.findings),
            "findings": [finding.as_dict() for finding in self.findings],
        }


SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[a-z0-9._\-+/=]{16,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "database_url": re.compile(
        r"(?i)\b[a-z][a-z0-9+.-]*://[^:\s/@]+:[^@\s]+@[^/\s]+/[^\s]+"
    ),
    "assignment_secret": re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|authorization|credential)"
        r"\s*[:=]\s*['\"]?[^'\"\s,;|]{8,}"
    ),
}

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "us_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
}

SENSITIVE_FIELD_NAMES = {
    "api_key",
    "authorization",
    "cookie",
    "database_url",
    "password",
    "secret",
    "token",
    "credential",
}


def _privacy_hash(raw: str) -> str:
    """生成敏感值 fingerprint。

    中文注释：
    hash 不是为了还原原文，而是为了判断“是不是同一个秘密/PII”。
    salt 放 env，避免不同环境的 hash 结果完全一样。
    """

    salt = os.getenv(PRIVACY_HASH_SALT_ENV, "beginner-agent-local-privacy-salt")
    digest = hashlib.sha256(f"{salt}:{raw}".encode("utf-8")).hexdigest()
    return digest[:16]


def _flatten_value(value: Any, *, max_chars: int = MAX_PRIVACY_SCAN_TEXT_CHARS) -> str:
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.append(f"{key}: {_flatten_value(item, max_chars=max_chars)}")
        return "\n".join(parts)[:max_chars]
    if isinstance(value, list):
        return "\n".join(_flatten_value(item, max_chars=max_chars) for item in value[:50])[
            :max_chars
        ]
    return str(value)[:max_chars]


def scan_text_for_privacy(text: str) -> PrivacyScanReport:
    """扫描文本中的 secret 和 PII。"""

    findings: list[PrivacyFinding] = []
    redactions: list[tuple[int, int, str]] = []
    for kind, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            raw = match.group(0)
            fingerprint = _privacy_hash(raw)
            findings.append(
                PrivacyFinding(kind, "secret", fingerprint, match.start(), match.end())
            )
            redactions.append((match.start(), match.end(), f"[REDACTED_SECRET:{fingerprint}]"))
    for kind, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            raw = match.group(0)
            fingerprint = _privacy_hash(raw)
            findings.append(
                PrivacyFinding(kind, "pii", fingerprint, match.start(), match.end())
            )
            redactions.append((match.start(), match.end(), f"[REDACTED_PII:{fingerprint}]"))

    redacted = text
    for start, end, replacement in sorted(redactions, reverse=True):
        redacted = redacted[:start] + replacement + redacted[end:]
    if len(redacted) > MAX_REDACTED_TEXT_CHARS:
        redacted = redacted[:MAX_REDACTED_TEXT_CHARS] + "...[TRUNCATED]"

    has_secret = any(finding.category == "secret" for finding in findings)
    has_pii = any(finding.category == "pii" for finding in findings)
    if has_secret:
        return PrivacyScanReport("secret", False, "hash_summary_only", findings, redacted)
    if has_pii:
        return PrivacyScanReport("confidential", False, "redacted", findings, redacted)
    return PrivacyScanReport("internal", True, "raw_allowed", findings, redacted)


def scan_value_for_privacy(value: Any) -> PrivacyScanReport:
    """扫描任意 JSON-like 值。"""

    return scan_text_for_privacy(_flatten_value(value))


def stronger_sensitivity(
    current: SensitivityLevel,
    detected: SensitivityLevel,
) -> SensitivityLevel:
    """取更严格的敏感级别。"""

    order = ["public", "internal", "confidential", "secret"]
    return order[max(order.index(current), order.index(detected))]  # type: ignore[return-value]


def redact_text_for_memory(text: str) -> str:
    """按 privacy policy 脱敏文本。"""

    return scan_text_for_privacy(text).redacted_text


def redact_value_for_memory(value: Any, *, key: str = "") -> Any:
    """递归脱敏要写入 memory metadata 的值。"""

    normalized_key = key.lower()
    if any(name in normalized_key for name in SENSITIVE_FIELD_NAMES):
        fingerprint = _privacy_hash(str(value))
        return {"redacted": True, "fingerprint": fingerprint, "reason": "sensitive field"}
    if isinstance(value, str):
        return redact_text_for_memory(value)
    if isinstance(value, dict):
        return {
            str(item_key): redact_value_for_memory(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_value_for_memory(item) for item in value[:50]]
    return value


def memory_prompt_allowed_by_privacy(record: dict[str, Any]) -> bool:
    """判断这条 memory 是否允许进入 prompt。"""

    sensitivity = str(record.get("sensitivity_level", "internal"))
    visibility = str(record.get("visibility", "project"))
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    privacy = metadata.get("privacy_governance")
    if isinstance(privacy, dict) and privacy.get("prompt_allowed") is False:
        return False
    if visibility == "retrieval_only":
        return False
    return sensitivity in {"public", "internal"}


def storage_summary_for_sensitive_memory(
    *,
    title: str,
    summary: str,
    report: PrivacyScanReport,
) -> tuple[str, str]:
    """为敏感 memory 生成只含摘要/hash 的 title 和 summary。"""

    safe_title = redact_text_for_memory(title)[:200]
    if report.storage_mode != "hash_summary_only":
        return safe_title, redact_text_for_memory(summary)[:800]
    fingerprints = sorted({finding.fingerprint for finding in report.findings})
    summary_text = (
        "Privacy Governance：检测到 secret，原始内容未写入长期记忆；"
        f"finding_count={len(report.findings)}；"
        f"fingerprints={','.join(fingerprints[:8])}。"
    )
    return safe_title or "Privacy protected memory", summary_text[:800]


def privacy_metadata(report: PrivacyScanReport) -> dict[str, Any]:
    """导出可保存到 memory/audit 的隐私治理元数据。"""

    return report.as_dict()
