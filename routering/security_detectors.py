from __future__ import annotations

import re

from .models import InjectionRisk, MaliciousIntent
from .security_builtin import EXFILTRATION_ACTION_KEYWORDS, PII_REGEXES, SECRET_REGEXES
from .security_models import SecurityFinding, SecurityPolicy


# 中文注释：
# security_detectors.py 放具体检测器和结果选择逻辑。
# 它不读取 env，也不关心 Router 主流程。


def detect_secret_and_pii(text: str) -> list[SecurityFinding]:
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
                    evidence=redact_evidence(match.group(0)),
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
                    evidence=redact_evidence(match.group(0)),
                    reason="用户输入中包含疑似 PII，需要避免进入不受控日志。",
                )
            )
    return findings


def detect_historical_abuse(text: str, policy: SecurityPolicy) -> list[SecurityFinding]:
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
                    evidence=redact_evidence(keyword),
                    reason="命中历史滥用模式库中的可疑短语。",
                )
            )
    return findings


def select_malicious_intent(
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


def select_injection_risk(findings: list[SecurityFinding]) -> InjectionRisk:
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


def redact_evidence(value: str) -> str:
    stripped = value.strip()
    if len(stripped) <= 4:
        return "***"
    return f"{stripped[:2]}***{stripped[-2:]}"


def dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result
