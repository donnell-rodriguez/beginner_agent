from __future__ import annotations

import ast
import json
import re
from typing import Any

from .command_tools import (
    run_cargo_check_tool,
    run_cargo_clippy_tool,
    run_cargo_fmt_check_tool,
    run_cargo_test_tool,
    run_mypy_tool,
    run_pytest_tool,
    run_ruff_format_check_tool,
    run_ruff_tool,
    run_uv_import_graph_tool,
)
from .core import json_dumps, project_text_files, safe_text_file
from .environment_tools import detect_project_stack_tool


# 中文注释：
# check_tools.py 放“真实验证工具”。
#
# 这次不再把 run_tests 简单等同于 static_check。
# 新逻辑是：
#
#   1. 先做 Python AST 静态语法检查。
#   2. 如果环境里有 pytest，就真实运行 pytest。
#   3. 如果没有 pytest，明确返回 skipped，而不是假装测试通过。
#   4. lint/typecheck/build 也用同样策略：真实工具优先，缺失就结构化说明。
#
# 这更接近生产级 code agent：验证结果必须诚实、结构化、可审计。


def _status_failed(result: dict[str, Any] | str) -> bool:
    """判断一个结构化或字符串结果是否失败。"""

    if isinstance(result, dict):
        return str(result.get("status")) in ("failed", "timeout", "blocked")
    return any(word in result for word in ("失败", "failed", "timeout", "blocked", "发现问题"))


def _json_tool_result(text: str) -> dict[str, Any]:
    """把工具 JSON 字符串转成 dict；失败时保留原文。"""

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {"status": "unknown", "raw": text}


def static_check_tool() -> str:
    """用 ast.parse 做 Python 语法检查。"""

    errors: list[dict[str, Any]] = []
    checked = 0
    for file_name in project_text_files():
        if not file_name.endswith(".py"):
            continue
        checked += 1
        try:
            # 中文注释：
            # ast.parse 只做语法层检查。
            # 它不能证明业务逻辑正确，但能快速发现最基本的 Python 语法错误。
            ast.parse(safe_text_file(file_name).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            errors.append(
                {
                    "file": file_name,
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                }
            )
    return json_dumps(
        {
            "status": "failed" if errors else "success",
            "checked_python_files": checked,
            "errors": errors,
        }
    )


def _text_hygiene_check() -> dict[str, Any]:
    """检查文本层面的低级问题。"""

    issues: list[dict[str, Any]] = []
    checked = 0
    for file_name in project_text_files():
        text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
        checked += 1
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "\t" in line:
                issues.append({"file": file_name, "line": line_number, "message": "包含 Tab 字符。"})
            if line.rstrip() != line:
                issues.append({"file": file_name, "line": line_number, "message": "行尾存在多余空白。"})
    return {"status": "failed" if issues else "success", "checked_files": checked, "issues": issues[:100]}


def lint_typecheck_tool() -> str:
    """运行 lint/typecheck 验证。

    中文注释：
    优先运行 ruff；如果本机没有 ruff，也不会假装成功，
    而是返回 skipped，并同时给出静态语法检查和文本检查。
    """

    static_result = _json_tool_result(static_check_tool())
    hygiene_result = _text_hygiene_check()
    ruff_result = _json_tool_result(run_ruff_tool())
    status = "failed" if any(_status_failed(item) for item in (static_result, hygiene_result, ruff_result)) else "success"
    if ruff_result.get("status") == "skipped" and status == "success":
        status = "partial"
    return json_dumps(
        {
            "status": status,
            "static_check": static_result,
            "text_hygiene": hygiene_result,
            "ruff": ruff_result,
        }
    )


def run_tests_tool() -> str:
    """运行测试。

    中文注释：
    生产级工具要区分：
    - success：真实测试通过。
    - failed：真实测试失败。
    - partial：没有测试框架，只做了基础检查。
    - skipped：工具不存在或没有测试目标。
    """

    static_result = _json_tool_result(static_check_tool())
    pytest_result = _json_tool_result(run_pytest_tool("beginner_agent"))
    if _status_failed(static_result) or _status_failed(pytest_result):
        status = "failed"
    elif pytest_result.get("status") == "skipped":
        status = "partial"
    else:
        status = "success"
    return json_dumps(
        {
            "status": status,
            "static_check": static_result,
            "pytest": pytest_result,
        }
    )


def run_targeted_tests_tool(target: str = "beginner_agent") -> str:
    """运行白名单范围内的定向测试。"""

    if target not in ("beginner_agent", ".", ""):
        return json_dumps({"status": "blocked", "target": target, "reason": "target 不在白名单中。"})
    return run_tests_tool()


def run_typecheck_tool() -> str:
    """运行类型检查。"""

    mypy_result = _json_tool_result(run_mypy_tool())
    if mypy_result.get("status") == "skipped":
        return json_dumps(
            {
                "status": "partial",
                "mypy": mypy_result,
                "fallback": _json_tool_result(static_check_tool()),
            }
        )
    return json_dumps({"status": mypy_result.get("status", "unknown"), "mypy": mypy_result})


def run_build_tool() -> str:
    """运行构建/导入验证。"""

    import_result = _json_tool_result(run_uv_import_graph_tool())
    status = "success" if import_result.get("status") == "success" else import_result.get("status", "unknown")
    return json_dumps({"status": status, "uv_import_graph": import_result})


def parse_test_failure_tool(test_output: str) -> str:
    """把测试失败日志解析成结构化摘要。"""

    failed_tests = re.findall(r"FAILED\s+([^\s]+)", test_output)
    file_lines = re.findall(r"([\w./-]+\.py):(\d+)", test_output)
    tracebacks = test_output.count("Traceback (most recent call last)")
    status = "failed" if any(word in test_output for word in ("失败", "FAILED", "Error", "Traceback")) else "unknown"
    return json_dumps(
        {
            "status": status,
            "failed_tests": failed_tests[:30],
            "locations": [{"file": file, "line": int(line)} for file, line in file_lines[:50]],
            "traceback_count": tracebacks,
            "summary": test_output[:2000],
        }
    )


def get_diagnostics_tool() -> str:
    """返回完整结构化诊断信息。"""

    stack = _json_tool_result(detect_project_stack_tool())
    static_result = _json_tool_result(static_check_tool())
    lint_result = _json_tool_result(lint_typecheck_tool())
    type_result = _json_tool_result(run_typecheck_tool())
    test_result = _json_tool_result(run_tests_tool())
    build_result = _json_tool_result(run_build_tool())
    cargo_check = _json_tool_result(run_cargo_check_tool())
    cargo_test = _json_tool_result(run_cargo_test_tool())
    cargo_clippy = _json_tool_result(run_cargo_clippy_tool())
    cargo_fmt = _json_tool_result(run_cargo_fmt_check_tool())
    rust_results = {
        "cargo_check": cargo_check,
        "cargo_test": cargo_test,
        "cargo_clippy": cargo_clippy,
        "cargo_fmt_check": cargo_fmt,
    }
    parts = (static_result, lint_result, type_result, test_result, build_result, *rust_results.values())
    status = "failed" if any(_status_failed(item) for item in parts) else "success"
    if status == "success" and any(item.get("status") in ("partial", "skipped") for item in parts):
        status = "partial"
    return json_dumps(
        {
            "status": status,
            "project_stack": stack,
            "static_check": static_result,
            "lint_typecheck": lint_result,
            "typecheck": type_result,
            "tests": test_result,
            "build": build_result,
            "rust": rust_results,
        }
    )


def format_check_tool() -> str:
    """检查格式。"""

    ruff_format = _json_tool_result(run_ruff_format_check_tool())
    hygiene = _text_hygiene_check()
    status = "failed" if _status_failed(hygiene) or _status_failed(ruff_format) else "success"
    if ruff_format.get("status") == "skipped" and status == "success":
        status = "partial"
    return json_dumps({"status": status, "ruff_or_fallback": ruff_format, "text_hygiene": hygiene})
