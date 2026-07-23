from __future__ import annotations

import re
from dataclasses import dataclass

from .security_builtin import PII_REGEXES, SECRET_REGEXES


# 中文注释：
# sanitization.py 负责 Router prompt 前的轻量脱敏。
#
# 注意：
# - 安全分类仍然看原始输入，避免漏判恶意意图。
# - LLM Router 看脱敏后的输入，避免 secret / PII 被送进模型上下文。


@dataclass(frozen=True)
class RouterSanitizedInput:
    original_text: str
    sanitized_text: str
    redaction_labels: tuple[str, ...] = ()

    @property
    def redacted(self) -> bool:
        return self.original_text != self.sanitized_text

    def as_dict(self) -> dict[str, object]:
        return {
            "redacted": self.redacted,
            "redaction_labels": list(self.redaction_labels),
        }


def sanitize_router_input_for_prompt(text: str) -> RouterSanitizedInput:
    """把输入中的 secret / PII 替换成占位符后再交给 LLM Router。"""

    sanitized = text
    labels: list[str] = []
    for label, pattern in SECRET_REGEXES:
        sanitized, count = re.subn(pattern, "[REDACTED_SECRET]", sanitized)
        if count:
            labels.append(label)
    for label, pattern in PII_REGEXES:
        sanitized, count = re.subn(pattern, "[REDACTED_PII]", sanitized)
        if count:
            labels.append(label)
    return RouterSanitizedInput(
        original_text=text,
        sanitized_text=sanitized,
        redaction_labels=tuple(dict.fromkeys(labels)),
    )
