from __future__ import annotations

import re

from .core import json_dumps


# 中文注释：
# failure_tools.py 放“失败分析”工具。
#
# 修改代码的 agent 不只是会跑测试，
# 还要能理解失败属于哪一类：
# - 语法错误
# - 类型错误
# - 测试断言失败
# - 导入错误
# - 构建失败
#
# 这样 Evaluator 才能决定下一步是重试、回滚、继续修复，还是请求人工帮助。


def extract_stack_trace_tool(output: str) -> str:
    """从测试/运行日志里提取 traceback 或错误片段。"""

    lines = output.splitlines()
    traces: list[str] = []
    capture = False
    current: list[str] = []

    for line in lines:
        if line.startswith("Traceback ") or re.match(r"_{5,}\s+.*\s+_{5,}", line):
            if current:
                traces.append("\n".join(current))
                current = []
            capture = True
        if capture:
            current.append(line)
            if re.match(r"^\w*(Error|Exception|Failure):", line):
                traces.append("\n".join(current))
                current = []
                capture = False
    if current:
        traces.append("\n".join(current))

    return json_dumps({"trace_count": len(traces), "traces": traces[:5], "summary": output[:1200]})


def classify_failure_tool(output: str) -> str:
    """把失败日志粗略分类，给修复循环一个稳定判断。"""

    text = output.lower()
    if "syntaxerror" in text or "语法错误" in text:
        category = "syntax_error"
    elif "modulenotfounderror" in text or "importerror" in text:
        category = "import_error"
    elif "assertionerror" in text or "failed" in text or "测试失败" in text:
        category = "test_failure"
    elif "typeerror" in text or "mypy" in text or "pyright" in text:
        category = "type_error"
    elif "permission" in text or "权限" in text or "blocked" in text:
        category = "permission_blocked"
    elif "build" in text or "构建" in text:
        category = "build_failure"
    else:
        category = "unknown"

    locations = [
        {"file": file_name, "line": int(line)}
        for file_name, line in re.findall(r"([\w./-]+\.(?:py|md|json|txt)):(\d+)", output)
    ]
    return json_dumps({"category": category, "locations": locations[:30], "summary": output[:1200]})


def compare_failure_before_after_tool(before: str, after: str) -> str:
    """比较修复前后的失败是否变少或变严重。"""

    before_lines = [line for line in before.splitlines() if line.strip()]
    after_lines = [line for line in after.splitlines() if line.strip()]
    before_failed = len(re.findall(r"FAILED|Error|Exception|失败|发现问题", before))
    after_failed = len(re.findall(r"FAILED|Error|Exception|失败|发现问题", after))
    if after_failed < before_failed:
        status = "improved"
    elif after_failed > before_failed:
        status = "regressed"
    elif after == before:
        status = "unchanged"
    else:
        status = "changed"
    return json_dumps(
        {
            "status": status,
            "before_signal_count": before_failed,
            "after_signal_count": after_failed,
            "before_lines": len(before_lines),
            "after_lines": len(after_lines),
        }
    )
