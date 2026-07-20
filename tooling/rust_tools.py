from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .command_tools import (
    run_cargo_check_tool,
    run_cargo_clippy_tool,
    run_cargo_fmt_check_tool,
    run_cargo_test_tool,
)
from .core import json_dumps, project_text_files, safe_resolve, safe_text_file, truncate


# 中文注释：
# rust_tools.py 是 Rust 语言适配器。
#
# 生产级 code agent 不应该只懂 Python。
# 对 Rust 项目，核心工具链通常是：
#
#   Cargo.toml
#   cargo check
#   cargo test
#   cargo clippy
#   cargo fmt --check
#
# 这里保持和 Python 工具同样的原则：
# - 只读工具只分析源码。
# - 验证工具走 command_tools.py 的白名单 profile。
# - 不允许 LLM 拼接任意 cargo 命令。


RUST_SYMBOL_PATTERN = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?P<kind>fn|struct|enum|trait|impl)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)?"
)


def _rust_files(root: str = ".") -> tuple[str, ...]:
    """返回 root 下的 Rust 源文件。"""

    root_path = safe_resolve(root)
    files: list[str] = []
    for file_name in project_text_files():
        if not file_name.endswith(".rs"):
            continue
        file_path = safe_text_file(file_name)
        try:
            file_path.relative_to(root_path)
        except ValueError:
            continue
        files.append(file_name)
    return tuple(files)


def detect_rust_project_tool(path: str = ".") -> str:
    """识别给定目录下是否像 Rust 项目。"""

    root = safe_resolve(path)
    cargo_toml = root / "Cargo.toml"
    rust_files = _rust_files(path)
    return json_dumps(
        {
            "path": path,
            "is_rust_project": cargo_toml.exists() or bool(rust_files),
            "has_cargo_toml": cargo_toml.exists(),
            "rust_file_count": len(rust_files),
            "rust_files": list(rust_files[:40]),
        }
    )


def inspect_rust_symbols_tool(path: str = ".") -> str:
    """扫描 Rust fn/struct/enum/trait/impl。"""

    symbols: list[dict[str, Any]] = []
    for file_name in _rust_files(path):
        text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = RUST_SYMBOL_PATTERN.match(line)
            if not match:
                continue
            kind = match.group("kind")
            name = match.group("name") or "<anonymous-impl>"
            symbols.append({"file": file_name, "line": line_number, "kind": kind, "name": name})
    return truncate(json_dumps({"path": path, "symbols": symbols[:200]}))


def inspect_rust_references_tool(symbol: str, path: str = ".") -> str:
    """查找 Rust 符号文本引用。"""

    symbol = symbol.strip()
    if len(symbol) < 2:
        return "inspect_rust_references 需要至少 2 个字符的 symbol。"
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    references: list[dict[str, Any]] = []
    for file_name in _rust_files(path):
        text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                references.append(
                    {
                        "file": file_name,
                        "line": line_number,
                        "preview": line.strip()[:180],
                    }
                )
    return truncate(json_dumps({"symbol": symbol, "references": references[:120]}))


def parse_rust_errors_tool(output: str) -> str:
    """解析 rustc/cargo 错误输出。"""

    errors: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in output.splitlines():
        error_match = re.match(r"error(?:\[(?P<code>E\d+)\])?: (?P<message>.*)", line)
        location_match = re.match(r"\s+-->\s+(?P<file>[^:]+):(?P<line>\d+):(?P<column>\d+)", line)
        if error_match:
            current = {
                "code": error_match.group("code") or "",
                "message": error_match.group("message"),
                "file": "",
                "line": 0,
                "column": 0,
            }
            errors.append(current)
        elif current and location_match:
            current["file"] = location_match.group("file")
            current["line"] = int(location_match.group("line"))
            current["column"] = int(location_match.group("column"))

    warnings = re.findall(r"warning: (.*)", output)
    status = "failed" if errors else "unknown"
    return json_dumps(
        {
            "status": status,
            "errors": errors[:50],
            "warnings": warnings[:50],
            "summary": output[:2000],
        }
    )


def parse_cargo_test_failure_tool(output: str) -> str:
    """解析 cargo test 失败输出。"""

    failed_tests = re.findall(r"----\s+([^\s]+)\s+stdout\s+----", output)
    panic_locations = [
        {"file": file_name, "line": int(line), "column": int(column)}
        for file_name, line, column in re.findall(r"panicked at ([^:]+):(\d+):(\d+)", output)
    ]
    status = "failed" if failed_tests or "test result: FAILED" in output else "unknown"
    return json_dumps(
        {
            "status": status,
            "failed_tests": failed_tests[:50],
            "panic_locations": panic_locations[:50],
            "summary": output[:2000],
        }
    )


def map_changed_rust_files_to_tests_tool(changed_files: list[str] | str) -> str:
    """根据 Rust 改动文件推断相关测试目标。"""

    if isinstance(changed_files, str):
        files = [item.strip() for item in changed_files.split(",") if item.strip()]
    else:
        files = [str(item).strip() for item in changed_files if str(item).strip()]

    mapping: dict[str, list[str]] = {}
    for file_name in files:
        if not file_name.endswith(".rs"):
            mapping[file_name] = []
            continue
        path = Path(file_name)
        stem = path.stem
        candidates = [
            f"cargo test {stem}",
            "cargo test",
        ]
        if "/src/" in file_name or file_name.startswith("src/"):
            candidates.insert(0, "cargo test --lib")
        if "/tests/" in file_name or file_name.startswith("tests/"):
            candidates.insert(0, f"cargo test --test {stem}")
        mapping[file_name] = list(dict.fromkeys(candidates))
    return json_dumps({"changed_files": files, "mapping": mapping})


def run_cargo_check_project_tool() -> str:
    """运行 cargo check 白名单 profile。"""

    return run_cargo_check_tool()


def run_cargo_test_project_tool() -> str:
    """运行 cargo test 白名单 profile。"""

    return run_cargo_test_tool()


def run_cargo_clippy_project_tool() -> str:
    """运行 cargo clippy 白名单 profile。"""

    return run_cargo_clippy_tool()


def run_cargo_fmt_check_project_tool() -> str:
    """运行 cargo fmt --check 白名单 profile。"""

    return run_cargo_fmt_check_tool()
