from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ..config import load_project_env
from .models import InjectionRisk, MaliciousIntent, RouterSecuritySignal
from .rules import DEFAULT_HIGH_RISK_KEYWORDS


# 中文注释：
# security.py 是 Router 的第一层安全分类器。
#
# 旧版本只做关键词判断。现在改成更接近生产系统的分层分类：
#
#   1. SecurityPattern policy
#      识别 prompt injection、数据外泄、危险代码动作。
#
#   2. Secret / PII detector
#      检测用户输入里是否出现 token、私钥、邮箱、手机号等敏感内容。
#
#   3. Historical abuse patterns
#      从历史滥用模式文件加载可疑短语，避免同类攻击反复绕过。
#
#   4. Structured findings
#      每个命中项都有 id / severity / confidence / evidence。
#
# Router 主节点仍然只调用 classify_router_security(text)，
# 所以安全分类器可以独立升级，不影响 graph 编排。


SecuritySeverity = Literal["low", "medium", "high", "critical"]
SecurityFindingKind = Literal[
    "prompt_injection",
    "data_exfiltration",
    "unsafe_code_action",
    "secret",
    "pii",
    "historical_abuse",
]


PROMPT_INJECTION_KEYWORDS = (
    "忽略之前",
    "忽略以上",
    "ignore previous",
    "ignore above",
    "forget previous",
    "system prompt",
    "developer message",
    "越狱",
    "jailbreak",
    "不要遵守",
    "绕过",
)
DATA_EXFILTRATION_KEYWORDS = (
    "读取 .env",
    "读取.env",
    "api key",
    "apikey",
    "secret",
    "token",
    "密码",
    "私钥",
    "泄露",
    "导出密钥",
)
EXFILTRATION_ACTION_KEYWORDS = (
    "告诉我",
    "发给我",
    "打印",
    "输出",
    "泄露",
    "exfiltrate",
    "send me",
    "show me",
    "print",
    "dump",
)


SECRET_REGEXES: tuple[tuple[str, str], ...] = (
    ("secret.openai_key", r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    ("secret.aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("secret.private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    (
        "secret.assignment",
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*[^\s'\"`]{6,}",
    ),
)
PII_REGEXES: tuple[tuple[str, str], ...] = (
    ("pii.email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("pii.phone", r"(?<!\d)(?:\+?\d[\d\s-]{7,}\d)(?!\d)"),
)


@dataclass(frozen=True)
class SecurityFinding:
    """一次安全命中结果。

    中文注释：
    大厂安全分类不会只返回一个 label。
    它通常会保留结构化证据，方便后续审计：
    - 哪个 detector 命中了？
    - 严重等级是多少？
    - 置信度是多少？
    - 命中的证据是什么？

    注意 evidence 会做脱敏，避免把 secret 原文写进 RouterEvent。
    """

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
    """安全策略规则。

    中文注释：
    它类似 RouterRule，但专门服务安全分类：
    - keywords：简单语义命中。
    - regexes：更强的模式识别。
    - priority：多个风险同时命中时，优先级更高的风险更重要。
    - enabled：可临时关闭某条安全规则。
    """

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

        lowered = text.lower()
        for keyword in self.keywords:
            if keyword.lower() in lowered:
                return SecurityFinding(
                    id=self.id,
                    kind=self.kind,
                    label=self.label,
                    severity=self.severity,
                    confidence=self.confidence,
                    evidence=_redact_evidence(keyword),
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
                    evidence=_redact_evidence(match.group(0)),
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
            object.__setattr__(self, "patterns", _default_security_patterns(self.version))


def _default_security_patterns(version: str) -> tuple[SecurityPattern, ...]:
    return (
        SecurityPattern(
            id="security.prompt_injection.override_instruction",
            kind="prompt_injection",
            label="prompt_injection",
            malicious_intent="prompt_injection",
            injection_risk="suspected",
            severity="high",
            confidence=0.86,
            keywords=PROMPT_INJECTION_KEYWORDS,
            priority=300,
            reason="用户输入试图覆盖系统/开发者指令。",
        ),
        SecurityPattern(
            id="security.data_exfiltration.secret_request",
            kind="data_exfiltration",
            label="data_exfiltration",
            malicious_intent="data_exfiltration",
            injection_risk="none",
            severity="critical",
            confidence=0.92,
            keywords=DATA_EXFILTRATION_KEYWORDS,
            priority=400,
            reason="用户输入涉及读取、输出或泄露敏感信息。",
        ),
        SecurityPattern(
            id="security.unsafe_code_action.mutating_command",
            kind="unsafe_code_action",
            label="unsafe_code_action",
            malicious_intent="unsafe_code_action",
            injection_risk="none",
            severity="high",
            confidence=0.82,
            keywords=DEFAULT_HIGH_RISK_KEYWORDS,
            priority=250,
            reason="用户输入涉及修改、删除、回滚或执行命令。",
        ),
    )


def _load_security_policy() -> SecurityPolicy:
    """加载 Router 安全策略。

    中文注释：
    可通过 env 指定策略文件：

        BEGINNER_AGENT_ROUTER_SECURITY_POLICY_PATH=.agent_state/router/security_policy.json

    如果文件不存在或格式错误，回退到内置安全策略。
    """

    load_project_env()
    path = os.getenv("BEGINNER_AGENT_ROUTER_SECURITY_POLICY_PATH", "").strip()
    policy = SecurityPolicy()
    if path:
        loaded = _load_security_policy_file(_resolve_path(path))
        if loaded is not None:
            policy = loaded

    history = _load_historical_abuse_keywords()
    if history:
        return SecurityPolicy(
            version=policy.version,
            source=policy.source,
            patterns=policy.patterns,
            historical_abuse_keywords=history,
        )
    return policy


def _load_security_policy_file(path: Path) -> SecurityPolicy | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    version = str(data.get("version", "custom-security-v1")).strip() or "custom-security-v1"
    raw_patterns = data.get("patterns", [])
    if not isinstance(raw_patterns, list):
        return None

    patterns = tuple(
        pattern
        for item in raw_patterns
        if isinstance(item, dict)
        for pattern in [_security_pattern_from_dict(item, version=version)]
        if pattern is not None
    )
    if not patterns:
        return None
    return SecurityPolicy(version=version, source=str(path), patterns=patterns)


def _security_pattern_from_dict(data: dict[str, Any], *, version: str) -> SecurityPattern | None:
    kind = data.get("kind")
    malicious_intent = data.get("malicious_intent", "none")
    injection_risk = data.get("injection_risk", "none")
    severity = data.get("severity", "medium")
    if kind not in {
        "prompt_injection",
        "data_exfiltration",
        "unsafe_code_action",
        "secret",
        "pii",
        "historical_abuse",
    }:
        return None
    if malicious_intent not in {
        "none",
        "prompt_injection",
        "unsafe_code_action",
        "data_exfiltration",
    }:
        return None
    if injection_risk not in {"none", "suspected", "high"}:
        return None
    if severity not in {"low", "medium", "high", "critical"}:
        return None

    keywords = _tuple_from_list(data.get("keywords", []))
    regexes = _tuple_from_list(data.get("regexes", []))
    if not keywords and not regexes:
        return None

    return SecurityPattern(
        id=str(data.get("id", f"security.{kind}.{version}")).strip(),
        kind=cast(SecurityFindingKind, kind),
        label=str(data.get("label", kind)).strip() or str(kind),
        malicious_intent=cast(MaliciousIntent, malicious_intent),
        injection_risk=cast(InjectionRisk, injection_risk),
        severity=cast(SecuritySeverity, severity),
        confidence=_float_from_dict(data, "confidence", 0.75),
        reason=str(data.get("reason", "命中自定义安全策略。")).strip(),
        keywords=keywords,
        regexes=regexes,
        priority=_int_from_dict(data, "priority", 100),
        enabled=_bool_from_dict(data, "enabled", True),
    )


def _detect_secret_and_pii(text: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for finding_id, pattern in SECRET_REGEXES:
        for match in re.finditer(pattern, text):
            findings.append(
                SecurityFinding(
                    id=finding_id,
                    kind="secret",
                    label="sensitive_input_secret",
                    severity="critical",
                    confidence=0.95,
                    evidence=_redact_evidence(match.group(0)),
                    reason="用户输入中包含疑似 secret，不能把原文写入日志或提示词。",
                )
            )

    for finding_id, pattern in PII_REGEXES:
        for match in re.finditer(pattern, text):
            findings.append(
                SecurityFinding(
                    id=finding_id,
                    kind="pii",
                    label="sensitive_input_pii",
                    severity="medium",
                    confidence=0.75,
                    evidence=_redact_evidence(match.group(0)),
                    reason="用户输入中包含疑似 PII，需要避免进入不受控日志。",
                )
            )
    return findings


def _detect_historical_abuse(text: str, policy: SecurityPolicy) -> list[SecurityFinding]:
    lowered = text.lower()
    findings: list[SecurityFinding] = []
    for keyword in policy.historical_abuse_keywords:
        if keyword.lower() in lowered:
            findings.append(
                SecurityFinding(
                    id="security.history.abuse_pattern",
                    kind="historical_abuse",
                    label="historical_abuse_pattern",
                    severity="high",
                    confidence=0.84,
                    evidence=_redact_evidence(keyword),
                    reason="命中历史滥用模式库中的可疑短语。",
                )
            )
    return findings


def _load_historical_abuse_keywords() -> tuple[str, ...]:
    path = os.getenv("BEGINNER_AGENT_ROUTER_ABUSE_PATTERNS_PATH", "").strip()
    if not path:
        return ()
    resolved = _resolve_path(path)
    if not resolved.exists():
        return ()
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ()
    if isinstance(data, dict):
        return _tuple_from_list(data.get("abuse_keywords", []))
    if isinstance(data, list):
        return _tuple_from_list(data)
    return ()


def classify_router_security(text: str) -> RouterSecuritySignal:
    """识别 Router 层的恶意意图和 prompt injection 风险。"""

    policy = _load_security_policy()
    findings: list[SecurityFinding] = []

    for pattern in sorted(policy.patterns, key=lambda item: item.priority, reverse=True):
        finding = pattern.find(text)
        if finding is not None:
            findings.append(finding)

    findings.extend(_detect_secret_and_pii(text))
    findings.extend(_detect_historical_abuse(text, policy))

    malicious_intent = _select_malicious_intent(findings, text)
    injection_risk = _select_injection_risk(findings)
    labels = _dedupe_labels([finding.as_label() for finding in findings])

    if not findings:
        reason = f"未命中 Router 安全风险规则。policy={policy.version}, source={policy.source}。"
    else:
        reason = (
            f"policy={policy.version}, source={policy.source}；"
            + "；".join(finding.as_reason_fragment() for finding in findings[:8])
        )

    return RouterSecuritySignal(
        injection_risk=injection_risk,
        malicious_intent=malicious_intent,
        labels=labels,
        reason=reason,
    )


def _select_malicious_intent(
    findings: list[SecurityFinding],
    text: str,
) -> MaliciousIntent:
    labels = {finding.label for finding in findings}
    lowered = text.lower()

    if "data_exfiltration" in labels:
        return "data_exfiltration"
    if "sensitive_input_secret" in labels and any(
        keyword.lower() in lowered for keyword in EXFILTRATION_ACTION_KEYWORDS
    ):
        return "data_exfiltration"
    if "prompt_injection" in labels:
        return "prompt_injection"
    if "unsafe_code_action" in labels or "historical_abuse_pattern" in labels:
        return "unsafe_code_action"
    return "none"


def _select_injection_risk(findings: list[SecurityFinding]) -> InjectionRisk:
    labels = {finding.label for finding in findings}
    if "prompt_injection" in labels and (
        "data_exfiltration" in labels
        or "unsafe_code_action" in labels
        or "sensitive_input_secret" in labels
    ):
        return "high"
    if "prompt_injection" in labels:
        return "suspected"
    return "none"


def _redact_evidence(value: str) -> str:
    stripped = value.strip()
    if len(stripped) <= 4:
        return "***"
    return f"{stripped[:2]}***{stripped[-2:]}"


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def _tuple_from_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _int_from_dict(data: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _float_from_dict(data: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, 0.0), 1.0)


def _bool_from_dict(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default
