from __future__ import annotations

from .rules_builtin import DEFAULT_HIGH_RISK_KEYWORDS
from .security_models import SecurityPattern


# 中文注释：
# security_builtin.py 放 Router 内置安全策略和内置检测正则。
# 这些是配置文件缺失时的安全兜底。

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


def default_security_patterns(version: str) -> tuple[SecurityPattern, ...]:
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
