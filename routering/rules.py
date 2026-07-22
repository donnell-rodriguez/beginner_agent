from __future__ import annotations

from .rule_models import RuleCategory, RuleDecision, RuleMatch, RouterRule, RouterRuleSet
from .rules_builtin import (
    DEFAULT_AGENT_KEYWORDS,
    DEFAULT_HIGH_RISK_KEYWORDS,
    DEFAULT_MEDIUM_RISK_KEYWORDS,
    DEFAULT_SEARCH_KEYWORDS,
    DEFAULT_WRITE_KEYWORDS,
)
from .rules_config import load_router_rules


# 中文注释：
# rules.py 现在是 Router 规则系统的“兼容入口”。
# 外部模块仍然可以 from .rules import RouterRuleSet / load_router_rules，
# 但真正实现已经拆到：
# - rule_models.py
# - rules_builtin.py
# - rules_config.py

__all__ = [
    "RuleCategory",
    "RuleDecision",
    "RuleMatch",
    "RouterRule",
    "RouterRuleSet",
    "DEFAULT_AGENT_KEYWORDS",
    "DEFAULT_SEARCH_KEYWORDS",
    "DEFAULT_WRITE_KEYWORDS",
    "DEFAULT_HIGH_RISK_KEYWORDS",
    "DEFAULT_MEDIUM_RISK_KEYWORDS",
    "load_router_rules",
]
