from __future__ import annotations

from .models import RouterSecuritySignal
from .security_config import load_security_policy
from .security_detectors import (
    dedupe_labels,
    detect_historical_abuse,
    detect_secret_and_pii,
    select_injection_risk,
    select_malicious_intent,
)
from .security_models import SecurityFinding, SecurityPattern, SecurityPolicy


# 中文注释：
# security.py 现在是 Router 安全分类入口。
# 具体能力已经拆到：
# - security_models.py：安全数据结构。
# - security_builtin.py：内置安全策略和正则。
# - security_config.py：策略文件加载。
# - security_detectors.py：secret/PII/历史滥用检测和风险选择。


def classify_router_security(text: str) -> RouterSecuritySignal:
    """识别 Router 层的恶意意图和 prompt injection 风险。"""

    policy = load_security_policy()
    findings: list[SecurityFinding] = []

    for pattern in sorted(policy.patterns, key=lambda item: item.priority, reverse=True):
        finding = pattern.find(text)
        if finding is not None:
            findings.append(finding)

    findings.extend(detect_secret_and_pii(text))
    findings.extend(detect_historical_abuse(text, policy))

    malicious_intent = select_malicious_intent(findings, text)
    injection_risk = select_injection_risk(findings)
    labels = dedupe_labels([finding.as_label() for finding in findings])

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


__all__ = [
    "SecurityFinding",
    "SecurityPattern",
    "SecurityPolicy",
    "classify_router_security",
]
