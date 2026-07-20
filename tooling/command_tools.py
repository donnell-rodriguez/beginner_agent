from __future__ import annotations

import subprocess
import time
from shutil import which
from typing import Any

from .core import (
    MAX_TOOL_OUTPUT_CHARS,
    active_project_root,
    json_dumps,
    safe_resolve,
    truncate,
)
from .runtime import resolve_command_runtime, runtime_catalog


# 中文注释：
# command_tools.py 是“受控命令执行层”。
#
# 生产级 code agent 需要运行测试、lint、build，但不能让 LLM 随便执行 shell。
# 所以这里采用白名单 profile：
#
#   run_allowed_command("python_compileall")
#
# LLM 只能选择 profile，不能传入任意命令字符串。


ALLOWED_COMMANDS: dict[str, dict[str, Any]] = {
    "python_compileall": {
        "cmd": ["python3", "-B", "-m", "compileall", "beginner_agent"],
        "runtime": "python",
        "cwd": "repo_root",
        "timeout": 30,
        "writes_cache": True,
    },
    "python_compile_state": {
        "cmd": ["python3", "-B", "-m", "py_compile", "beginner_agent/state.py"],
        "runtime": "python",
        "cwd": "repo_root",
        "timeout": 15,
        "writes_cache": False,
    },
    "uv_import_graph": {
        "cmd": [
            "uv",
            "run",
            "--no-dev",
            "--project",
            "libs/langgraph",
            "python",
            "-B",
            "-c",
            "from beginner_agent.graph import build_graph; print(type(build_graph()).__name__)",
        ],
        "runtime": "python",
        "cwd": "repo_root",
        "timeout": 30,
        "writes_cache": True,
    },
    "ruff_check": {
        "cmd": ["ruff", "check", "beginner_agent"],
        "runtime": "python",
        "cwd": "repo_root",
        "timeout": 30,
        "writes_cache": False,
    },
    "ruff_format_check": {
        "cmd": ["ruff", "format", "--check", "beginner_agent"],
        "runtime": "python",
        "cwd": "repo_root",
        "timeout": 30,
        "writes_cache": False,
    },
    "mypy_beginner_agent": {
        "cmd": ["mypy", "beginner_agent"],
        "runtime": "python",
        "cwd": "repo_root",
        "timeout": 45,
        "writes_cache": True,
    },
    "pytest_beginner_agent": {
        "cmd": ["python3", "-B", "-m", "pytest", "beginner_agent"],
        "runtime": "python",
        "cwd": "repo_root",
        "timeout": 60,
        "writes_cache": True,
    },
    "cargo_check": {
        "cmd": ["cargo", "check"],
        "runtime": "rust",
        "cwd": "active_project",
        "timeout": 90,
        "writes_cache": True,
        "requires": "Cargo.toml",
    },
    "cargo_test": {
        "cmd": ["cargo", "test"],
        "runtime": "rust",
        "cwd": "active_project",
        "timeout": 120,
        "writes_cache": True,
        "requires": "Cargo.toml",
    },
    "cargo_clippy": {
        "cmd": ["cargo", "clippy", "--", "-D", "warnings"],
        "runtime": "rust",
        "cwd": "active_project",
        "timeout": 120,
        "writes_cache": True,
        "requires": "Cargo.toml",
    },
    "cargo_fmt_check": {
        "cmd": ["cargo", "fmt", "--check"],
        "runtime": "rust",
        "cwd": "active_project",
        "timeout": 60,
        "writes_cache": False,
        "requires": "Cargo.toml",
    },
}


def _command_available(cmd: list[str]) -> tuple[bool, str]:
    """检查白名单命令的第一个可执行文件是否存在。"""

    executable = cmd[0]
    if which(executable):
        return True, "命令可用。"
    return False, f"命令不可用：{executable}"


def _run_profile(profile: str) -> dict[str, Any]:
    """执行一个白名单 profile，并返回结构化结果。"""

    spec = ALLOWED_COMMANDS.get(profile)
    if spec is None:
        return {
            "status": "blocked",
            "profile": profile,
            "reason": "profile 不在 ALLOWED_COMMANDS 白名单中。",
        }

    cmd = list(spec["cmd"])
    resolved_runtime = resolve_command_runtime(spec)
    cwd = resolved_runtime["cwd"]
    env = resolved_runtime["env"]
    runtime_name = resolved_runtime["runtime"]
    required_file = spec.get("requires")
    if required_file and not (cwd / str(required_file)).exists():
        return {
            "status": "skipped",
            "profile": profile,
            "reason": f"当前项目缺少 {required_file}，跳过该语言工具。",
            "cmd": cmd,
        }
    available, reason = _command_available(cmd)
    if not available:
        return {
            "status": "skipped",
            "profile": profile,
            "reason": reason,
            "cmd": cmd,
        }

    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(spec.get("timeout", 30)),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "profile": profile,
            "cmd": cmd,
            "timeout": spec.get("timeout", 30),
            "stdout": truncate(exc.stdout or "", MAX_TOOL_OUTPUT_CHARS),
            "stderr": truncate(exc.stderr or "", MAX_TOOL_OUTPUT_CHARS),
        }

    elapsed_ms = round((time.monotonic() - started) * 1000)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 0:
        status = "success"
    elif "No module named pytest" in stderr:
        status = "skipped"
    else:
        status = "failed"
    return {
        "status": status,
        "profile": profile,
        "cmd": cmd,
        "cwd": str(cwd),
        "runtime": runtime_name,
        "returncode": result.returncode,
        "elapsed_ms": elapsed_ms,
        "stdout": truncate(stdout, MAX_TOOL_OUTPUT_CHARS),
        "stderr": truncate(stderr, MAX_TOOL_OUTPUT_CHARS),
        "writes_cache": bool(spec.get("writes_cache", False)),
    }


def list_allowed_commands_tool() -> str:
    """列出 Agent 允许执行的命令 profile。"""

    commands: dict[str, dict[str, Any]] = {}
    for profile, spec in ALLOWED_COMMANDS.items():
        resolved_runtime = resolve_command_runtime(spec)
        commands[profile] = {
            "cmd": spec["cmd"],
            "runtime": resolved_runtime["runtime"],
            "cwd": str(resolved_runtime["cwd"]),
            "timeout": spec.get("timeout", 30),
            "writes_cache": bool(spec.get("writes_cache", False)),
        }
    return json_dumps(
        {
            "commands": commands,
            "runtime_catalog": runtime_catalog(),
        }
    )


def run_allowed_command_tool(profile: str) -> str:
    """执行一个白名单命令 profile。"""

    return json_dumps(_run_profile(profile))


def run_package_script_tool(script: str) -> str:
    """运行预定义的包管理器脚本。

    中文注释：
    这里仍然不是任意 shell。
    script 只能映射到 ALLOWED_COMMANDS 中已有 profile。
    """

    mapping = {
        "test": "pytest_beginner_agent",
        "lint": "ruff_check",
        "format_check": "ruff_format_check",
        "typecheck": "mypy_beginner_agent",
        "build": "uv_import_graph",
        "cargo_check": "cargo_check",
        "cargo_test": "cargo_test",
        "cargo_clippy": "cargo_clippy",
        "cargo_fmt_check": "cargo_fmt_check",
    }
    profile = mapping.get(script)
    if profile is None:
        return json_dumps({"status": "blocked", "script": script, "reason": "脚本不在白名单中。"})
    return run_allowed_command_tool(profile)


def run_pytest_tool(target: str = "beginner_agent") -> str:
    """运行 pytest 白名单目标。"""

    if target not in ("beginner_agent", ".", ""):
        return json_dumps({"status": "blocked", "target": target, "reason": "pytest target 不在白名单中。"})
    return run_allowed_command_tool("pytest_beginner_agent")


def run_ruff_tool() -> str:
    """运行 ruff check。"""

    return run_allowed_command_tool("ruff_check")


def run_ruff_format_check_tool() -> str:
    """运行 ruff format --check。"""

    return run_allowed_command_tool("ruff_format_check")


def run_mypy_tool() -> str:
    """运行 mypy。"""

    return run_allowed_command_tool("mypy_beginner_agent")


def run_uv_import_graph_tool() -> str:
    """验证 LangGraph beginner_agent 能否构建图。"""

    return run_allowed_command_tool("uv_import_graph")


def run_cargo_check_tool() -> str:
    """运行 cargo check。"""

    return run_allowed_command_tool("cargo_check")


def run_cargo_test_tool() -> str:
    """运行 cargo test。"""

    return run_allowed_command_tool("cargo_test")


def run_cargo_clippy_tool() -> str:
    """运行 cargo clippy。"""

    return run_allowed_command_tool("cargo_clippy")


def run_cargo_fmt_check_tool() -> str:
    """运行 cargo fmt --check。"""

    return run_allowed_command_tool("cargo_fmt_check")


def safe_path_exists_tool(path: str) -> str:
    """安全检查 active project 内的相对路径是否存在。"""

    safe_path = safe_resolve(path)
    project_root = active_project_root()
    return json_dumps(
        {
            "path": path,
            "exists": safe_path.exists(),
            "is_file": safe_path.is_file(),
            "is_dir": safe_path.is_dir(),
            "relative_to_active_project": safe_path.relative_to(project_root).as_posix()
            if safe_path.exists()
            else path,
        }
    )
