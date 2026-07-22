from __future__ import annotations

from .models import RouterSecuritySignal
from .rules import DEFAULT_HIGH_RISK_KEYWORDS


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
)


def classify_router_security(text: str) -> RouterSecuritySignal:
    """识别 Router 层的恶意意图和 prompt injection 风险。"""

    lowered = text.lower()
    labels: list[str] = []
    reasons: list[str] = []

    if any(keyword.lower() in lowered for keyword in PROMPT_INJECTION_KEYWORDS):
        labels.append("prompt_injection")
        reasons.append("命中 prompt injection / 越权指令关键词。")

    if any(keyword.lower() in lowered for keyword in DATA_EXFILTRATION_KEYWORDS):
        labels.append("data_exfiltration")
        reasons.append("命中敏感信息读取或外泄关键词。")

    if any(keyword.lower() in lowered for keyword in DEFAULT_HIGH_RISK_KEYWORDS):
        labels.append("unsafe_code_action")
        reasons.append("命中修改/删除/执行类高风险动作关键词。")

    if "data_exfiltration" in labels:
        malicious_intent = "data_exfiltration"
    elif "prompt_injection" in labels:
        malicious_intent = "prompt_injection"
    elif "unsafe_code_action" in labels:
        malicious_intent = "unsafe_code_action"
    else:
        malicious_intent = "none"

    if "prompt_injection" in labels and (
        "data_exfiltration" in labels or "unsafe_code_action" in labels
    ):
        injection_risk = "high"
    elif "prompt_injection" in labels:
        injection_risk = "suspected"
    else:
        injection_risk = "none"

    return RouterSecuritySignal(
        injection_risk=injection_risk,
        malicious_intent=malicious_intent,
        labels=labels,
        reason="；".join(reasons) if reasons else "未命中 Router 安全风险规则。",
    )
