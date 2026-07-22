from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

from ..config import load_project_env
from .rule_models import RuleCategory, RouterRule, RouterRuleSet
from .rules_builtin import (
    DEFAULT_AGENT_KEYWORDS,
    DEFAULT_HIGH_RISK_KEYWORDS,
    DEFAULT_MEDIUM_RISK_KEYWORDS,
    DEFAULT_SEARCH_KEYWORDS,
    DEFAULT_WRITE_KEYWORDS,
)


# 中文注释：
# rules_config.py 只负责“从配置文件加载规则”。
# 规则数据结构在 rule_models.py，内置规则在 rules_builtin.py。


def load_router_rules() -> RouterRuleSet:
    """加载 Router 规则。"""

    load_project_env()

    rollback_path = os.getenv("BEGINNER_AGENT_ROUTER_RULES_ROLLBACK_PATH", "").strip()
    if rollback_path:
        resolved = _resolve_config_path(rollback_path)
        if resolved.exists() and (ruleset := _load_rules_from_file(resolved, source=str(resolved))):
            return RouterRuleSet(
                version=ruleset.version,
                source=f"rollback:{ruleset.source}",
                rules=ruleset.rules,
                rollback_from=os.getenv("BEGINNER_AGENT_ROUTER_RULES_PATH", "").strip(),
            )

    path = os.getenv("BEGINNER_AGENT_ROUTER_RULES_PATH", "").strip()
    if not path:
        return RouterRuleSet()

    resolved = _resolve_config_path(path)
    if not resolved.exists():
        return RouterRuleSet()

    loaded = _load_rules_from_file(resolved, source=str(resolved))
    return loaded or RouterRuleSet()


def _load_rules_from_file(path: Path, *, source: str) -> RouterRuleSet | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    version = str(data.get("version", "custom-v1")).strip() or "custom-v1"
    modern_rules = _rules_from_modern_config(data)
    rules = modern_rules or _rules_from_legacy_config(data, version=version)
    if not rules:
        return None

    rollback_from = str(data.get("rollback_from", "")).strip()
    return RouterRuleSet(
        version=version,
        source=source,
        rules=rules,
        rollback_from=rollback_from,
    )


def _rules_from_modern_config(data: dict[str, Any]) -> tuple[RouterRule, ...]:
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        return ()

    version = str(data.get("version", "custom-v1")).strip() or "custom-v1"
    rules = tuple(
        rule
        for item in raw_rules
        if isinstance(item, dict)
        for rule in [_rule_from_dict(item, default_version=version)]
        if rule is not None
    )
    return rules


def _rules_from_legacy_config(data: dict[str, Any], *, version: str) -> tuple[RouterRule, ...]:
    """兼容旧配置格式。"""

    return (
        RouterRule(
            id="task.agent.legacy_config",
            version=version,
            category="task_type",
            outcome="agent",
            keywords=_tuple_from_config(data, "agent_keywords", DEFAULT_AGENT_KEYWORDS),
            priority=300,
            reason="命中旧格式配置里的 agent_keywords。",
        ),
        RouterRule(
            id="task.search.legacy_config",
            version=version,
            category="task_type",
            outcome="search",
            keywords=_tuple_from_config(data, "search_keywords", DEFAULT_SEARCH_KEYWORDS),
            priority=200,
            reason="命中旧格式配置里的 search_keywords。",
        ),
        RouterRule(
            id="task.write.legacy_config",
            version=version,
            category="task_type",
            outcome="write",
            keywords=_tuple_from_config(data, "write_keywords", DEFAULT_WRITE_KEYWORDS),
            priority=100,
            reason="命中旧格式配置里的 write_keywords。",
        ),
        RouterRule(
            id="risk.high.legacy_config",
            version=version,
            category="risk_level",
            outcome="high",
            keywords=_tuple_from_config(data, "high_risk_keywords", DEFAULT_HIGH_RISK_KEYWORDS),
            priority=300,
            reason="命中旧格式配置里的 high_risk_keywords。",
        ),
        RouterRule(
            id="risk.medium.legacy_config",
            version=version,
            category="risk_level",
            outcome="medium",
            keywords=_tuple_from_config(data, "medium_risk_keywords", DEFAULT_MEDIUM_RISK_KEYWORDS),
            priority=200,
            reason="命中旧格式配置里的 medium_risk_keywords。",
        ),
    )


def _rule_from_dict(data: dict[str, Any], *, default_version: str) -> RouterRule | None:
    category = data.get("category")
    outcome = data.get("outcome")
    if category not in {"task_type", "risk_level"}:
        return None
    if category == "task_type" and outcome not in {"search", "write", "chat", "agent"}:
        return None
    if category == "risk_level" and outcome not in {"low", "medium", "high"}:
        return None

    keywords = _tuple_from_config(data, "keywords", ())
    if not keywords:
        return None

    rule_id = str(data.get("id", "")).strip()
    if not rule_id:
        stable = hashlib.sha256(
            json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        rule_id = f"{category}.{outcome}.{stable}"

    rollout_percent = max(0, min(_int_from_config(data, "rollout_percent", 100), 100))
    return RouterRule(
        id=rule_id,
        version=str(data.get("version", default_version)).strip() or default_version,
        category=cast(RuleCategory, category),
        outcome=str(outcome),
        keywords=keywords,
        priority=_int_from_config(data, "priority", 100),
        enabled=_bool_from_config(data, "enabled", True),
        rollout_percent=rollout_percent,
        reason=str(data.get("reason", "")).strip() or "命中自定义 Router 规则。",
    )


def _tuple_from_config(data: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        return default
    cleaned = tuple(str(item).strip() for item in value if str(item).strip())
    return cleaned or default


def _int_from_config(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_from_config(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _resolve_config_path(path: str) -> Path:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return config_path
