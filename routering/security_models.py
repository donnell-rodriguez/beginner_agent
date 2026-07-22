from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .models import InjectionRisk, MaliciousIntent


# 中文注释：
# security_models.py 只定义 Router 安全分类需要的数据结构。
# 安全策略加载、secret 检测、分类编排分别放在其它文件里。

SecuritySeverity = Literal["low", "medium", "high", "critical"]
SecurityFindingKind = Literal[
    "prompt_injection",
    "data_exfiltration",
    "unsafe_code_action",
    "secret",
    "pii",
    "historical_abuse",
]


@dataclass(frozen=True)
class SecurityFinding:
    """一次安全命中结果。"""

    id: str
    kind: SecurityFindingKind
    label: str
    severity: SecuritySeverity
    confidence: float
    evidence: str
    reason: str

    def as_label(self) -> str:
        return self.label

    def as_reason_fragment(self) -> str:
        return (
            f"{self.id}[{self.severity}, confidence={self.confidence:.2f}, "
            f"evidence={self.evidence}]：{self.reason}"
        )


@dataclass(frozen=True)
class SecurityPattern:
    """安全策略规则。"""

    id: str
    kind: SecurityFindingKind
    label: str
    malicious_intent: MaliciousIntent
    injection_risk: InjectionRisk
    severity: SecuritySeverity
    confidence: float
    reason: str
    keywords: tuple[str, ...] = ()
    regexes: tuple[str, ...] = ()
    priority: int = 100
    enabled: bool = True

    def find(self, text: str) -> SecurityFinding | None:
        if not self.enabled:
            return None

        from .security_detectors import redact_evidence

        lowered = text.lower()
        for keyword in self.keywords:
            if keyword.lower() in lowered:
                return SecurityFinding(
                    id=self.id,
                    kind=self.kind,
                    label=self.label,
                    severity=self.severity,
                    confidence=self.confidence,
                    evidence=redact_evidence(keyword),
                    reason=self.reason,
                )

        for pattern in self.regexes:
            match = re.search(pattern, text)
            if match:
                return SecurityFinding(
                    id=self.id,
                    kind=self.kind,
                    label=self.label,
                    severity=self.severity,
                    confidence=self.confidence,
                    evidence=redact_evidence(match.group(0)),
                    reason=self.reason,
                )
        return None


@dataclass(frozen=True)
class SecurityPolicy:
    """Router 安全策略集合。"""

    version: str = "builtin-security-v1"
    source: str = "builtin"
    patterns: tuple[SecurityPattern, ...] = field(default_factory=tuple)
    historical_abuse_keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.patterns:
            from .security_builtin import default_security_patterns

            object.__setattr__(self, "patterns", default_security_patterns(self.version))
