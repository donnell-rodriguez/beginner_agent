from __future__ import annotations

"""Unified tool facade for beginner_agent.

中文注释：
这个文件现在只是“工具统一入口”。

为什么不把所有工具都继续堆在 tools.py 里？

因为现在工具已经很多：

- 文件读取工具
- 搜索工具
- Python / Rust 代码结构分析工具
- 测试 / 诊断工具
- Python / Rust 白名单验证工具
- git 工具
- patch / rollback 工具
- checkpoint / 安全扫描工具

如果都放在一个文件里，后续会越来越难读。

所以具体实现已经拆到：

    beginner_agent/tooling/

外部模块仍然可以继续这样写：

    from .tools import run_tool, validate_tool_request

这样 Planner / Policy / Executor 不需要知道具体工具放在哪个文件里，
也就是降低模块之间的耦合。
"""

from .tooling.core import project_text_files, read_text_for_snapshot
from .tooling.registry import (
    ALL_TOOLS,
    FAILED_TOOL_RESULT_PREFIXES,
    READ_ONLY_TOOLS,
    WRITE_TOOLS,
    run_tool,
    run_tool_model,
    run_tool_result,
    tool_result_json_schema,
    validate_tool_request,
)

__all__ = [
    "ALL_TOOLS",
    "FAILED_TOOL_RESULT_PREFIXES",
    "READ_ONLY_TOOLS",
    "WRITE_TOOLS",
    "project_text_files",
    "read_text_for_snapshot",
    "run_tool",
    "run_tool_model",
    "run_tool_result",
    "tool_result_json_schema",
    "validate_tool_request",
]
