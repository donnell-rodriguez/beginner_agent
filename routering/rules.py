from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_project_env
from ..state import RiskLevel, TaskType


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
class RouterRuleSet:
    """Router 本地规则集。

    中文注释：
    生产级 Router 不能只有 prompt。
    这些规则是 LLM 失败时的兜底，也可以覆盖明显的安全场景。

    默认规则在代码里，生产部署时可以用 JSON 文件覆盖：

        BEGINNER_AGENT_ROUTER_RULES_PATH=.agent_state/router/rules.json
    """

    agent_keywords: tuple[str, ...] = DEFAULT_AGENT_KEYWORDS
    search_keywords: tuple[str, ...] = DEFAULT_SEARCH_KEYWORDS
    write_keywords: tuple[str, ...] = DEFAULT_WRITE_KEYWORDS
    high_risk_keywords: tuple[str, ...] = DEFAULT_HIGH_RISK_KEYWORDS
    medium_risk_keywords: tuple[str, ...] = DEFAULT_MEDIUM_RISK_KEYWORDS

    def classify_task_type(self, text: str) -> TaskType:
        if self._contains_any(text, self.agent_keywords):
            return "agent"
        if self._contains_any(text, self.search_keywords):
            return "search"
        if self._contains_any(text, self.write_keywords):
            return "write"
        return "chat"

    def classify_risk_level(self, text: str) -> RiskLevel:
        if self._contains_any(text, self.high_risk_keywords):
            return "high"
        if self._contains_any(text, self.medium_risk_keywords):
            return "medium"
        return "low"

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)


def _tuple_from_config(data: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        return default
    cleaned = tuple(str(item).strip() for item in value if str(item).strip())
    return cleaned or default


def load_router_rules() -> RouterRuleSet:
    """加载 Router 规则。

    中文注释：
    配置优先来自 env 指定的 JSON 文件。
    如果没有配置文件，使用内置默认规则。
    这样本地学习可以零配置，后续生产化时可以把规则交给配置管理。
    """

    load_project_env()
    path = os.getenv("BEGINNER_AGENT_ROUTER_RULES_PATH", "").strip()
    if not path:
        return RouterRuleSet()

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        return RouterRuleSet()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RouterRuleSet()
    if not isinstance(data, dict):
        return RouterRuleSet()

    defaults = RouterRuleSet()
    return RouterRuleSet(
        agent_keywords=_tuple_from_config(data, "agent_keywords", defaults.agent_keywords),
        search_keywords=_tuple_from_config(data, "search_keywords", defaults.search_keywords),
        write_keywords=_tuple_from_config(data, "write_keywords", defaults.write_keywords),
        high_risk_keywords=_tuple_from_config(data, "high_risk_keywords", defaults.high_risk_keywords),
        medium_risk_keywords=_tuple_from_config(
            data, "medium_risk_keywords", defaults.medium_risk_keywords
        ),
    )
