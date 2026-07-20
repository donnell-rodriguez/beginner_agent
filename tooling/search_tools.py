from __future__ import annotations

import re

from .core import MAX_SEARCH_RESULTS, project_text_files, safe_text_file


# 中文注释：
# search_tools.py 放“搜索代码”的工具。
#
# search_code_tool 是普通关键词搜索。
# grep_regex_tool 是正则搜索，更灵活但也更容易写错 pattern。
#
def search_code_tool(query: str) -> str:
    """普通字符串搜索。"""

    query = query.strip()
    if len(query) < 2:
        return "search_code 需要至少 2 个字符的 query。"

    matches: list[str] = []

    # 中文注释：
    # project_text_files() 会返回所有允许读取的文本文件。
    # 每个文件逐行扫描，找到 query 就记录：文件名:行号:内容。
    for file_name in project_text_files():
        text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if query.lower() in line.lower():
                matches.append(f"{file_name}:{line_number}: {line.strip()}")
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return "\n".join(matches) + "\n...搜索结果过多，已截断..."
    return "\n".join(matches) if matches else f"没有搜索到：{query}"


def grep_regex_tool(pattern: str, path: str = ".") -> str:
    """正则搜索代码。"""

    if len(pattern.strip()) < 2:
        return "grep_regex 需要至少 2 个字符的 pattern。"
    try:
        # 中文注释：
        # re.compile 会把字符串 pattern 编译成正则表达式。
        # 如果用户给了非法正则，这里会抛 re.error。
        regex = re.compile(pattern)
    except re.error as exc:
        return f"正则表达式错误：{exc}"

    matches: list[str] = []
    for file_name in project_text_files():
        # 中文注释：
        # path 可以限制搜索范围。
        # 例如 path="tooling" 就只搜索 tooling/ 下面的文件。
        if path not in (".", "") and not file_name.startswith(path.rstrip("/") + "/"):
            continue
        text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{file_name}:{line_number}: {line.strip()}")
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return "\n".join(matches) + "\n...搜索结果过多，已截断..."
    return "\n".join(matches) if matches else f"没有匹配正则：{pattern}"
