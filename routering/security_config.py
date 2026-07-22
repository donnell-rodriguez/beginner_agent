from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from ..config import load_project_env
from .models import InjectionRisk, MaliciousIntent
from .security_models import (
    SecurityFindingKind,
    SecurityPattern,
    SecurityPolicy,
    SecuritySeverity,
)


# 中文注释：
# security_config.py 负责加载安全策略文件和历史滥用模式文件。
# classify_router_security 不直接关心配置格式，只拿最终 SecurityPolicy 使用。


def load_security_policy() -> SecurityPolicy:
    """加载 Router 安全策略。"""

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
