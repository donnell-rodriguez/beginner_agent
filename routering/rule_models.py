from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

from ..state import RiskLevel, TaskType


# 中文注释：
# rule_models.py 只定义 Router 规则系统的数据结构。
# 它不负责读配置文件，也不负责放默认关键词。

RuleCategory: TypeAlias = Literal["task_type", "risk_level"]


@dataclass(frozen=True)
class RuleMatch:
    """一次规则命中记录。"""

    rule_id: str
    rule_version: str
    category: RuleCategory
    outcome: str
    keyword: str
    priority: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "category": self.category,
            "outcome": self.outcome,
            "keyword": self.keyword,
            "priority": self.priority,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RuleDecision:
    """规则引擎的一次决策结果。"""

    outcome: str
    default_outcome: str
    ruleset_version: str
    ruleset_source: str
    selected_rule_id: str
    selected_rule_reason: str
    matches: tuple[RuleMatch, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        return bool(self.selected_rule_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "default_outcome": self.default_outcome,
            "ruleset_version": self.ruleset_version,
            "ruleset_source": self.ruleset_source,
            "selected_rule_id": self.selected_rule_id,
            "selected_rule_reason": self.selected_rule_reason,
            "matches": [match.as_dict() for match in self.matches],
        }


@dataclass(frozen=True)
class RouterRule:
    """一条可治理的 Router 规则。"""

    id: str
    category: RuleCategory
    outcome: str
    keywords: tuple[str, ...]
    priority: int
    reason: str
    version: str = "builtin-v1"
    enabled: bool = True
    rollout_percent: int = 100

    def first_match(self, text: str) -> RuleMatch | None:
        if not self.enabled:
            return None
        if not _rollout_enabled(text, self.id, self.rollout_percent):
            return None

        lowered = text.lower()
        for keyword in self.keywords:
            if keyword.lower() in lowered:
                return RuleMatch(
                    rule_id=self.id,
                    rule_version=self.version,
                    category=self.category,
                    outcome=self.outcome,
                    keyword=keyword,
                    priority=self.priority,
                    reason=self.reason,
                )
        return None


@dataclass(frozen=True)
class RouterRuleSet:
    """Router 本地规则集。"""

    version: str = "builtin-v1"
    source: str = "builtin"
    rules: tuple[RouterRule, ...] = field(default_factory=tuple)
    rollback_from: str = ""

    def __post_init__(self) -> None:
        if not self.rules:
            from .rules_builtin import default_rules

            object.__setattr__(self, "rules", default_rules(self.version))

    def explain_task_type(self, text: str) -> RuleDecision:
        return self._decide(
            text=text,
            category="task_type",
            default_outcome="chat",
        )

    def explain_risk_level(self, text: str) -> RuleDecision:
        return self._decide(
            text=text,
            category="risk_level",
            default_outcome="low",
        )

    def classify_task_type(self, text: str) -> TaskType:
        """兼容旧调用：只返回最终 task_type。"""

        return cast(TaskType, self.explain_task_type(text).outcome)

    def classify_risk_level(self, text: str) -> RiskLevel:
        """兼容旧调用：只返回最终 risk_level。"""

        return cast(RiskLevel, self.explain_risk_level(text).outcome)

    def _decide(
        self,
        *,
        text: str,
        category: RuleCategory,
        default_outcome: str,
    ) -> RuleDecision:
        matches = tuple(
            match
            for rule in self.rules
            if rule.category == category
            for match in [rule.first_match(text)]
            if match is not None
        )
        selected = sorted(matches, key=lambda item: item.priority, reverse=True)[:1]
        if not selected:
            return RuleDecision(
                outcome=default_outcome,
                default_outcome=default_outcome,
                ruleset_version=self.version,
                ruleset_source=self.source,
                selected_rule_id="",
                selected_rule_reason="没有规则命中，使用默认结果。",
                matches=matches,
            )

        top = selected[0]
        return RuleDecision(
            outcome=top.outcome,
            default_outcome=default_outcome,
            ruleset_version=self.version,
            ruleset_source=self.source,
            selected_rule_id=top.rule_id,
            selected_rule_reason=top.reason,
            matches=matches,
        )


def _rollout_enabled(text: str, rule_id: str, rollout_percent: int) -> bool:
    """判断当前请求是否进入这条规则的灰度范围。"""

    if rollout_percent >= 100:
        return True
    if rollout_percent <= 0:
        return False
    raw = f"{rule_id}:{text}".encode("utf-8")
    bucket = int(hashlib.sha256(raw).hexdigest()[:8], 16) % 100
    return bucket < rollout_percent
