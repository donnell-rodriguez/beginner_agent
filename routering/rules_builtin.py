from __future__ import annotations

from .rule_models import RouterRule


# 中文注释：
# rules_builtin.py 只放内置关键词和内置规则。
# 生产系统里这些通常会逐步迁移到配置中心，但本地 fallback 仍然需要兜底规则。

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


def default_rules(version: str = "builtin-v1") -> tuple[RouterRule, ...]:
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
