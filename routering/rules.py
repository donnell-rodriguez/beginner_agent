from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from ..config import load_project_env
from ..state import RiskLevel, TaskType


# 中文注释：
# rules.py 是 Router 的本地规则引擎。
#
# 早期版本只是“关键词列表 -> 分类结果”。
# 现在改成更接近生产系统的 RuleSpec 风格：
#
#   RouterRule
#     -> 有 id / version / category / outcome / priority / rollout_percent / reason
#     -> 可以解释为什么命中
#     -> 可以通过 rollout_percent 做灰度
#     -> 可以通过 rollback path 切换到旧规则文件
#
# Router 主节点仍然只调用：
#
#   rules.classify_task_type(text)
#   rules.classify_risk_level(text)
#
# 所以业务入口稳定，规则系统可以独立升级。


RuleCategory: TypeAlias = Literal["task_type", "risk_level"]


DEFAULT_AGENT_KEYWORDS = (
    "项目结构",
    "整体",
    "流程",
    "拆解",
    "计划",
    "理解项目",
    "文件",
    "源码",
    "目录",
    "读取",
    "看一下",
    "列出",
    "修改代码",
    "修复",
    "测试",
    "patch",
    "diff",
    "回滚",
    "lint",
    "typecheck",
)
DEFAULT_SEARCH_KEYWORDS = ("搜索", "查找", "查询", "找一下")
DEFAULT_WRITE_KEYWORDS = ("写", "生成", "起草", "润色")
DEFAULT_HIGH_RISK_KEYWORDS = (
    "修改代码",
    "修复",
    "删除",
    "移除",
    "写入",
    "覆盖",
    "执行命令",
    "运行命令",
    "apply_patch",
    "patch",
    "回滚",
    "rollback",
    "format_apply",
)
DEFAULT_MEDIUM_RISK_KEYWORDS = (
    "运行测试",
    "测试",
    "lint",
    "typecheck",
    "mypy",
    "ruff",
    "cargo",
    "build",
    "编译",
)


@dataclass(frozen=True)
class RuleMatch:
    """一次规则命中记录。

    中文注释：
    生产级规则系统不能只告诉你“结果是 agent/high”，
    还要告诉你：
    - 哪条规则命中了？
    - 命中了哪个关键词？
    - 规则优先级是多少？
    - 这条规则属于哪个版本？
    """

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
    """一条可治理的 Router 规则。

    中文注释：
    这比简单关键词列表更接近生产级：
    - id：稳定规则 ID，方便审计和回滚。
    - version：规则版本，方便知道是哪一版规则做的判断。
    - category：这条规则判断 task_type 还是 risk_level。
    - outcome：命中后输出什么结果。
    - keywords：命中的关键词。
    - priority：优先级，数值越大越先采用。
    - rollout_percent：灰度比例，0-100。
    - reason：给人看的命中说明。
    """

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
    """Router 本地规则集。

    中文注释：
    它不是单纯保存关键词，而是保存一组 RouterRule。
    RouterRuleSet 负责：
    - 按 category 过滤规则。
    - 找出所有命中的规则。
    - 按 priority 选择最终规则。
    - 返回可审计的 RuleDecision。
    """

    version: str = "builtin-v1"
    source: str = "builtin"
    rules: tuple[RouterRule, ...] = field(default_factory=tuple)
    rollback_from: str = ""

    def __post_init__(self) -> None:
        if not self.rules:
            object.__setattr__(self, "rules", _default_rules(self.version))

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


def _default_rules(version: str = "builtin-v1") -> tuple[RouterRule, ...]:
    return (
        RouterRule(
            id="task.agent.project_or_code",
            version=version,
            category="task_type",
            outcome="agent",
            keywords=DEFAULT_AGENT_KEYWORDS,
            priority=300,
            reason="用户请求涉及项目、源码、文件、测试或代码修改，应进入复杂 agent loop。",
        ),
        RouterRule(
            id="task.search.knowledge_lookup",
            version=version,
            category="task_type",
            outcome="search",
            keywords=DEFAULT_SEARCH_KEYWORDS,
            priority=200,
            reason="用户请求更像资料检索或查找。",
        ),
        RouterRule(
            id="task.write.generation",
            version=version,
            category="task_type",
            outcome="write",
            keywords=DEFAULT_WRITE_KEYWORDS,
            priority=100,
            reason="用户请求更像写作、生成或润色。",
        ),
        RouterRule(
            id="risk.high.mutating_or_command",
            version=version,
            category="risk_level",
            outcome="high",
            keywords=DEFAULT_HIGH_RISK_KEYWORDS,
            priority=300,
            reason="用户请求涉及修改、删除、执行命令或 patch，属于高风险。",
        ),
        RouterRule(
            id="risk.medium.validation_or_build",
            version=version,
            category="risk_level",
            outcome="medium",
            keywords=DEFAULT_MEDIUM_RISK_KEYWORDS,
            priority=200,
            reason="用户请求涉及测试、lint、typecheck 或构建，属于中风险。",
        ),
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
    """兼容旧配置格式。

    中文注释：
    旧格式是：

        {"agent_keywords": ["..."], "high_risk_keywords": ["..."]}

    新引擎会把它转换成 RouterRule。
    这样你之前写过的 rules.json 不会突然失效。
    """

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


def _resolve_config_path(path: str) -> Path:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return config_path


def load_router_rules() -> RouterRuleSet:
    """加载 Router 规则。

    中文注释：
    规则加载顺序：

    1. BEGINNER_AGENT_ROUTER_RULES_ROLLBACK_PATH
       如果配置了 rollback path，优先加载回滚规则。

    2. BEGINNER_AGENT_ROUTER_RULES_PATH
       正常加载当前规则文件。

    3. builtin-v1
       文件不存在、JSON 错误、规则非法时，使用内置规则。

    这就是生产系统里常见的“可回滚配置”思路：
    规则文件出问题时，Router 不应该直接不可用。
    """

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
