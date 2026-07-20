from __future__ import annotations

import ast
from collections import defaultdict

from .core import json_dumps, project_text_files, safe_text_file


# 中文注释：
# inspect_tools.py 放“理解代码结构”的工具。
#
# 它主要使用 Python 标准库 ast。
# ast 可以把 Python 源码解析成语法树，
# 然后我们就能找 class、function、import、函数调用关系。
#
def _python_files() -> tuple[str, ...]:
    """只返回 Python 文件。"""

    return tuple(file_name for file_name in project_text_files() if file_name.endswith(".py"))


def inspect_symbol_tool(symbol: str) -> str:
    """查找函数、类定义位置。"""

    symbol = symbol.strip()
    if len(symbol) < 2:
        return "inspect_symbol 需要至少 2 个字符的 symbol。"

    matches: list[str] = []
    for file_name in _python_files():
        path = safe_text_file(file_name)
        try:
            # 中文注释：
            # ast.parse 把 Python 源码转成语法树。
            # 后面 ast.walk 会遍历这棵树。
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            matches.append(f"{file_name}: 语法错误，无法索引：{exc}")
            continue
        for node in ast.walk(tree):
            # 中文注释：
            # ClassDef 表示 class 定义。
            # FunctionDef 表示普通函数定义。
            # AsyncFunctionDef 表示 async 函数定义。
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == symbol:
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    matches.append(f"{file_name}:{node.lineno}: {kind} {node.name}")
    return "\n".join(matches) if matches else f"没有找到符号：{symbol}"


def inspect_references_tool(symbol: str) -> str:
    """查找符号引用。"""

    symbol = symbol.strip()
    if len(symbol) < 2:
        return "inspect_references 需要至少 2 个字符的 symbol。"

    references: list[dict[str, object]] = []
    for file_name in _python_files():
        path = safe_text_file(file_name)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            references.append({"file": file_name, "line": 0, "kind": "syntax_error", "message": str(exc)})
            continue

        for node in ast.walk(tree):
            # 中文注释：
            # ast.Name 覆盖直接引用，例如 run_tool(...)
            # ast.Attribute 覆盖属性引用，例如 module.run_tool(...)
            if isinstance(node, ast.Name) and node.id == symbol:
                references.append({"file": file_name, "line": node.lineno, "kind": "name"})
            elif isinstance(node, ast.Attribute) and node.attr == symbol:
                references.append({"file": file_name, "line": node.lineno, "kind": "attribute"})

    return json_dumps({"symbol": symbol, "references": references[:80]})


def inspect_import_graph_tool() -> str:
    """分析模块 import 关系。"""

    graph: dict[str, list[str]] = {}
    for file_name in _python_files():
        imports: list[str] = []
        try:
            tree = ast.parse(safe_text_file(file_name).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            graph[file_name] = ["<syntax-error>"]
            continue
        for node in ast.walk(tree):
            # 中文注释：
            # import x 会生成 ast.Import。
            # from x import y 会生成 ast.ImportFrom。
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                imports.append(module)
        graph[file_name] = sorted(set(imports))
    return json_dumps(graph)


def inspect_call_graph_tool(function: str = "") -> str:
    """建立简化函数调用图。"""

    # 中文注释：
    # call_graph 的结构大致是：
    #
    #   {
    #     "graph.py:build_graph": ["StateGraph", "add_node", ...]
    #   }
    #
    # 它不是完整 IDE 级调用图，但足够帮助小项目理解调用关系。
    call_graph: dict[str, list[str]] = defaultdict(list)
    for file_name in _python_files():
        try:
            tree = ast.parse(safe_text_file(file_name).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller = f"{file_name}:{node.name}"
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # 中文注释：
                        # ast.Call 表示一次函数调用。
                        # 例如 foo() 或 obj.foo()。
                        func = child.func
                        if isinstance(func, ast.Name):
                            call_graph[caller].append(func.id)
                        elif isinstance(func, ast.Attribute):
                            call_graph[caller].append(func.attr)

    if function:
        filtered = {key: value for key, value in call_graph.items() if key.endswith(f":{function}")}
        return json_dumps(filtered or {function: []})
    return json_dumps({key: sorted(set(value)) for key, value in call_graph.items()})
