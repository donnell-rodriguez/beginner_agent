from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

from ..state import RiskLevel, TaskType


# 中文注释：
# rule_models.py 只定义 Router 规则系统的数据结构。
# 它不负责读配置文件，也不负责放默认关键词。
#
# 你可以把这里理解成 Router 本地规则引擎的“表结构 / 数据模型”：
#
#   RouterRule
#     表示“一条规则”，例如：
#       如果用户输入里包含“修改代码”，就判断 task_type = agent。
#
#   RuleMatch
#     表示“某条规则真的命中了用户输入”，例如：
#       rule_id=task.code.modify 命中了 keyword=修改代码。
#
#   RuleDecision
#     表示“规则引擎最后选择了什么结果”，例如：
#       多条规则都命中时，选择 priority 最高的那条。
#
#   RouterRuleSet
#     表示“一整套规则”，里面有很多 RouterRule。

# 中文注释：
# TypeAlias 表示“给一个复杂类型起别名”。
#
# RuleCategory 只有两个允许值：
# - "task_type"：这类规则用来判断任务类型，例如 chat / agent。
# - "risk_level"：这类规则用来判断风险等级，例如 low / high。
#
# Literal[...] 的意思是：这个变量只能是列出来的几个字符串之一。
RuleCategory: TypeAlias = Literal["task_type", "risk_level"]


@dataclass(frozen=True)
class RuleMatch:
    """一次规则命中记录。

    中文注释：
    RuleMatch 不是“规则本身”，而是“规则命中的结果记录”。

    例子：

        用户输入：
            "请帮我修改 router.py"

        某条 RouterRule：
            keywords=("修改", "代码")
            outcome="agent"

        如果命中了 "修改"，就会生成一个 RuleMatch：

            rule_id="task.code.modify"
            keyword="修改"
            outcome="agent"

    它的作用是用于解释、审计和调试：
    为什么 Router 会把这个请求判断成 agent？
    因为哪条规则、哪个关键词、哪个优先级命中了。
    """

    # 中文注释：命中的规则 ID。生产系统里每条规则都应该有稳定 ID，方便审计和回滚。
    rule_id: str

    # 中文注释：命中的规则版本。规则升级后可以知道这次命中来自哪一版规则。
    rule_version: str

    # 中文注释：这条规则属于哪类判断：任务类型 task_type，还是风险等级 risk_level。
    category: RuleCategory

    # 中文注释：
    # 命中后给出的结果。
    # 如果 category 是 task_type，outcome 可能是 "chat" / "agent"。
    # 如果 category 是 risk_level，outcome 可能是 "low" / "medium" / "high"。
    outcome: str

    # 中文注释：真正命中的关键词。例如规则有多个关键词，这里记录具体命中的是哪一个。
    keyword: str

    # 中文注释：
    # 优先级。多个规则同时命中时，priority 越大越优先。
    # 例如“读取文件”是低风险，但“读取 .env”是高风险，高风险规则应该有更高优先级。
    priority: int

    # 中文注释：这条规则为什么这么判断。这个 reason 会进入报告，方便人理解。
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """把 RuleMatch 转成普通 dict。

        中文注释：
        dataclass 对 Python 来说很好用，
        但写入 JSONL、API 返回、日志审计时，通常要转成 dict。
        """

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
    """规则引擎的一次决策结果。

    中文注释：
    RuleDecision 表示“本地规则引擎最终给出的答案”。

    它和 RuleMatch 的区别：

    - RuleMatch：记录所有命中的规则。
    - RuleDecision：从命中的规则里选出最终结果。

    例子：

        用户输入：
            "帮我读取 .env"

        可能同时命中：
            1. "读取" -> needs tool / agent / low
            2. ".env" -> high risk

        RuleDecision 会保存：
            outcome="high"
            selected_rule_id="risk.secret.env"
            matches=(所有命中的 RuleMatch)
    """

    # 中文注释：最终选择的结果，例如 task_type 的 "agent"，或 risk_level 的 "high"。
    outcome: str

    # 中文注释：如果没有任何规则命中，就使用这个默认结果。
    default_outcome: str

    # 中文注释：当前规则集版本。用于排查“是哪一版规则做出的判断”。
    ruleset_version: str

    # 中文注释：规则来源。可能是 builtin、配置文件、config registry 等。
    ruleset_source: str

    # 中文注释：最终被选中的规则 ID。没有命中时为空字符串。
    selected_rule_id: str

    # 中文注释：最终被选中的规则原因。会用于 Router 报告和审计。
    selected_rule_reason: str

    # 中文注释：
    # 所有命中的规则记录。
    # field(default_factory=tuple) 表示默认给一个空 tuple，避免用可变默认值。
    matches: tuple[RuleMatch, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        """是否真的有规则命中。

        中文注释：
        @property 让你可以写 decision.matched，
        而不是 decision.matched()。
        """

        return bool(self.selected_rule_id)

    def as_dict(self) -> dict[str, Any]:
        """把规则决策结果转成 dict，方便写日志、做观测和返回 API。"""

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
    RouterRule 才是“规则本身”。

    例子：

        RouterRule(
            id="task.code.modify",
            category="task_type",
            outcome="agent",
            keywords=("修改代码", "修复测试"),
            priority=80,
            reason="代码修改请求应该进入 agent 分支。",
        )

    它表示：
        如果用户输入里包含 "修改代码" 或 "修复测试"，
        那么这条规则建议 Router 输出 task_type="agent"。
    """

    # 中文注释：规则的稳定 ID。不要随便改，日志和 eval 会依赖它。
    id: str

    # 中文注释：规则分类。决定这条规则是在判断 task_type 还是 risk_level。
    category: RuleCategory

    # 中文注释：规则命中后的输出结果。
    outcome: str

    # 中文注释：关键词列表。只要用户输入包含其中一个关键词，就算命中。
    keywords: tuple[str, ...]

    # 中文注释：优先级。多个规则同时命中时，数字越大越优先。
    priority: int

    # 中文注释：解释这条规则为什么存在。
    reason: str

    # 中文注释：规则版本。方便灰度、回滚和问题排查。
    version: str = "builtin-v1"

    # 中文注释：是否启用。False 时，这条规则存在但不会参与判断。
    enabled: bool = True

    # 中文注释：
    # 灰度比例，0 到 100。
    # 100 表示所有请求都启用这条规则。
    # 50 表示大约一半请求会使用这条规则，方便测试新规则效果。
    rollout_percent: int = 100

    def first_match(self, text: str) -> RuleMatch | None:
        """检查这条规则是否命中输入文本。

        中文注释：
        返回值有两种：

        - RuleMatch：表示命中了，并记录命中的关键词。
        - None：表示没命中，或者规则没有启用，或者灰度没有覆盖本次请求。
        """

        if not self.enabled:
            return None
        if not _rollout_enabled(text, self.id, self.rollout_percent):
            return None

        # 中文注释：
        # lower() 把英文转小写，避免大小写影响匹配。
        # 中文没有大小写，所以不受影响。
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
    RouterRuleSet 是“一组规则”的容器。

    RouterRule 是单条规则；
    RouterRuleSet 是很多 RouterRule 的集合。

    它提供两个主要能力：

    - explain_task_type(text)：判断任务类型，并返回可解释的 RuleDecision。
    - explain_risk_level(text)：判断风险等级，并返回可解释的 RuleDecision。
    """

    # 中文注释：整套规则的版本。规则集整体升级时改这个。
    version: str = "builtin-v1"

    # 中文注释：规则来源，例如 builtin / config_file / config_registry。
    source: str = "builtin"

    # 中文注释：具体规则列表。默认空时，会在 __post_init__ 里加载内置规则。
    rules: tuple[RouterRule, ...] = field(default_factory=tuple)

    # 中文注释：如果这套规则是从某个版本回滚来的，这里记录原版本。
    rollback_from: str = ""

    def __post_init__(self) -> None:
        """dataclass 初始化后自动执行的钩子。

        中文注释：
        如果创建 RouterRuleSet 时没有传 rules，
        就自动加载 rules_builtin.py 里的默认规则。

        因为当前 dataclass 是 frozen=True，不能直接 self.rules = ...
        所以这里使用 object.__setattr__(...) 做初始化阶段的赋值。
        """

        if not self.rules:
            from .rules_builtin import default_rules

            object.__setattr__(self, "rules", default_rules(self.version))

    def explain_task_type(self, text: str) -> RuleDecision:
        """用规则解释任务类型。

        中文注释：
        如果没有规则命中，默认 task_type 是 "chat"。
        """

        return self._decide(
            text=text,
            category="task_type",
            default_outcome="chat",
        )

    def explain_risk_level(self, text: str) -> RuleDecision:
        """用规则解释风险等级。

        中文注释：
        如果没有规则命中，默认 risk_level 是 "low"。
        """

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
        """通用规则决策函数。

        中文注释：
        explain_task_type(...) 和 explain_risk_level(...) 都会调用这里。

        处理流程：

        1. 遍历当前规则集里的所有规则。
        2. 只保留 category 相同的规则。
        3. 调用 rule.first_match(text) 看是否命中。
        4. 收集所有 RuleMatch。
        5. 按 priority 从高到低排序。
        6. 选择优先级最高的命中结果。
        7. 如果没有任何规则命中，就返回默认结果。
        """

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
    """判断当前请求是否进入这条规则的灰度范围。

    中文注释：
    灰度的目的：
    新规则不一定一上来就给 100% 请求使用。
    可以先让 10% 请求使用，观察效果后再扩大。

    这里用 rule_id + text 做 hash，得到一个稳定 bucket：

        同一个规则 + 同一个输入
          -> 每次都会落到同一个 bucket

    这样不会出现同一个用户输入一会儿命中新规则、一会儿不命中的随机抖动。
    """

    if rollout_percent >= 100:
        return True
    if rollout_percent <= 0:
        return False
    raw = f"{rule_id}:{text}".encode("utf-8")
    bucket = int(hashlib.sha256(raw).hexdigest()[:8], 16) % 100
    return bucket < rollout_percent
