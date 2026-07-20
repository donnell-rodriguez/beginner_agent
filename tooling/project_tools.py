from __future__ import annotations

from .core import json_dumps, project_text_files, safe_text_file


# 中文注释：
# project_tools.py 放“理解项目整体信息”的工具。
# 它们不是直接修代码，而是帮助 agent 建立项目上下文。
#
def dependency_inspect_tool() -> str:
    """查看项目依赖配置。"""

    # 中文注释：
    # 这里列出常见依赖文件。
    # 当前 beginner_agent 可能没有独立依赖文件，所以会返回提示。
    candidates = [
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "Cargo.toml",
        "../pyproject.toml",
    ]
    found: dict[str, str] = {}
    for candidate in candidates:
        try:
            found[candidate] = safe_text_file(candidate).read_text(encoding="utf-8", errors="replace")[:3000]
        except ValueError:
            continue
    return json_dumps(found or {"message": "没有找到常见依赖配置文件。"})


def summarize_file_tool(path: str) -> str:
    """生成简单文件摘要。"""

    # 中文注释：
    # 这个摘要是规则版，不调用 LLM。
    # 它会提取：
    # - 行数
    # - import 行
    # - def / class 定义
    # - 文件前 20 行预览
    text = safe_text_file(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    functions = [line.strip() for line in lines if line.lstrip().startswith(("def ", "class "))]
    imports = [line.strip() for line in lines if line.lstrip().startswith(("import ", "from "))]
    return json_dumps(
        {
            "path": path,
            "line_count": len(lines),
            "imports": imports[:20],
            "definitions": functions[:40],
            "preview": "\n".join(lines[:20]),
        }
    )
