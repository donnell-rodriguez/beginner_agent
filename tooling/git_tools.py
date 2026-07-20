from __future__ import annotations

import subprocess
from pathlib import Path

from .core import MAX_TOOL_OUTPUT_CHARS, active_project_root, safe_resolve, truncate


# 中文注释：
# git_tools.py 放 Git 观察工具。
#
# 注意：
#   这里只执行固定 git 命令：
#   - git status
#   - git diff
#
# 不提供任意 shell command runner。
# LLM 不能决定执行任意命令。
def _git_root() -> Path | None:
    """查找 active project 所属 Git 仓库根。"""

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=active_project_root(),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _run_git(args: list[str]) -> str:
    """执行固定范围内的 git 命令。"""

    git_root = _git_root()
    if git_root is None:
        return "git 失败：当前 active project 不在 Git 仓库内。"
    result = subprocess.run(
        ["git", *args],
        cwd=git_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return f"git {' '.join(args)} 失败：{result.stderr.strip()}"
    return truncate(result.stdout.strip() or "没有输出。", MAX_TOOL_OUTPUT_CHARS)


def git_status_tool() -> str:
    """查看 active project 的 git 状态。"""

    return _run_git(["status", "--short", "--", active_project_root().as_posix()])


def git_diff_tool() -> str:
    """查看 active project 的整体 diff。"""

    output = _run_git(["diff", "--", active_project_root().as_posix()])
    return output if output != "没有输出。" else "active project 当前没有 git diff。"


def git_diff_file_tool(path: str) -> str:
    """查看单个文件 diff。"""

    safe_path = safe_resolve(path)
    output = _run_git(["diff", "--", safe_path.as_posix()])
    return output if output != "没有输出。" else f"{path} 当前没有 git diff。"
