from __future__ import annotations

from .core import MAX_READ_CHARS, IGNORED_DIRS, safe_resolve, safe_text_file, truncate


# 中文注释：
# read_tools.py 放“读取文件/目录”的工具。
#
# 这些工具都是只读工具：
#   不修改文件
#   不删除文件
#   不执行命令
#
def list_files_tool(path: str = ".") -> str:
    """列出指定目录下一层文件。"""

    # 中文注释：
    # safe_resolve 会确保 path 没有跑出 beginner_agent 目录。
    root = safe_resolve(path)
    if not root.exists():
        return f"路径不存在：{path}"
    if not root.is_dir():
        return f"不是目录：{path}"

    lines: list[str] = []

    # 中文注释：
    # iterdir() 只看当前目录的一层。
    # 如果要递归看目录树，用 list_tree_tool。
    for item in sorted(root.iterdir(), key=lambda value: value.name):
        if item.name in IGNORED_DIRS:
            continue
        relative = item.relative_to(safe_resolve("."))
        suffix = "/" if item.is_dir() else ""
        lines.append(f"{relative.as_posix()}{suffix}")
    return "\n".join(lines) if lines else "目录为空。"


def list_tree_tool(path: str = ".", max_depth: int = 2) -> str:
    """递归列出项目目录树。"""

    root = safe_resolve(path)
    if not root.exists():
        return f"路径不存在：{path}"
    if not root.is_dir():
        return f"不是目录：{path}"

    # 中文注释：
    # 限制 max_depth 最大为 5。
    # 防止 agent 一次递归读取太深，输出过大。
    max_depth = max(0, min(int(max_depth), 5))
    base = safe_resolve(".")
    lines: list[str] = []

    def walk(current, depth: int) -> None:
        # 中文注释：
        # 这是一个递归函数。
        # current 是当前目录，depth 是当前递归深度。
        if depth > max_depth:
            return
        for item in sorted(current.iterdir(), key=lambda value: value.name):
            if item.name in IGNORED_DIRS:
                continue
            relative = item.relative_to(base).as_posix()
            prefix = "  " * depth
            suffix = "/" if item.is_dir() else ""
            lines.append(f"{prefix}{relative}{suffix}")
            if item.is_dir():
                walk(item, depth + 1)

    walk(root, 0)
    return "\n".join(lines) if lines else "目录为空。"


def read_file_tool(path: str) -> str:
    """读取完整文本文件，超长会截断。"""

    # 中文注释：
    # safe_text_file 会确认：
    # - 文件在 beginner_agent 里面。
    # - 文件存在。
    # - 文件后缀是允许的文本类型。
    text = safe_text_file(path).read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + "\n\n...内容过长，已截断..."
    return text


def read_file_slice_tool(path: str, start: int = 1, end: int = 120) -> str:
    """按行号读取文件片段。"""

    # 中文注释：
    # 真实项目里文件可能很大。
    # read_file_slice 可以只读第 start 到 end 行，
    # 这比一次性读取整个文件更适合 agent。
    file_path = safe_text_file(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start))
    end = max(start, min(int(end), len(lines)))
    selected = [
        f"{line_number}: {lines[line_number - 1]}"
        for line_number in range(start, end + 1)
    ]
    return truncate("\n".join(selected))
