from __future__ import annotations

import ast
import json
from collections import defaultdict

from .core import STATE_DIR, ensure_state_dirs, json_dumps, project_text_files, safe_text_file, truncate


# 中文注释：
# index_tools.py 放“项目索引”工具。
#
# 对 code agent 来说，索引就像“项目地图”：
# - 哪些文件里有哪些 class / function。
# - 哪些文件 import 了哪些模块。
# - 用户查询一个关键词时，哪些文件最相关。
#
# 大厂 code agent 通常不会每次都盲目全文搜索，
# 而是会先建立索引，再基于索引做定位和任务拆解。

INDEX_FILE = STATE_DIR / "project_index.json"


def _read_python_tree(file_name: str) -> ast.AST | None:
    """安全读取 Python 文件并解析成 AST。"""

    text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def build_project_index_tool() -> str:
    """扫描 beginner_agent，生成一个轻量代码索引。"""

    ensure_state_dirs()
    files: list[dict[str, object]] = []
    symbols: list[dict[str, object]] = []
    imports: dict[str, list[str]] = {}

    for file_name in project_text_files():
        file_path = safe_text_file(file_name)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        files.append(
            {
                "path": file_name,
                "suffix": file_path.suffix,
                "lines": len(text.splitlines()),
                "chars": len(text),
            }
        )

        if not file_name.endswith(".py"):
            continue

        tree = _read_python_tree(file_name)
        if tree is None:
            continue

        file_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(
                    {
                        "name": node.name,
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "file": file_name,
                        "line": node.lineno,
                    }
                )
            elif isinstance(node, ast.Import):
                file_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                file_imports.append("." * node.level + module)
        imports[file_name] = sorted(set(file_imports))

    index = {
        "files": files,
        "symbols": sorted(symbols, key=lambda item: (str(item["file"]), int(item["line"]))),
        "imports": imports,
    }
    INDEX_FILE.write_text(json_dumps(index), encoding="utf-8")
    return json_dumps(
        {
            "status": "success",
            "index_file": ".agent_state/project_index.json",
            "file_count": len(files),
            "symbol_count": len(symbols),
        }
    )


def _load_or_build_index() -> dict[str, object]:
    """读取已有索引；如果没有，就先构建。"""

    if not INDEX_FILE.exists():
        build_project_index_tool()
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def query_project_index_tool(query: str) -> str:
    """从索引里查询文件名、符号名和 import。"""

    query = query.strip().lower()
    if len(query) < 2:
        return "query_project_index 需要至少 2 个字符的 query。"

    index = _load_or_build_index()
    file_hits: list[dict[str, object]] = []
    symbol_hits: list[dict[str, object]] = []
    import_hits: dict[str, list[str]] = defaultdict(list)

    for item in index.get("files", []):
        if isinstance(item, dict) and query in str(item.get("path", "")).lower():
            file_hits.append(item)

    for item in index.get("symbols", []):
        if isinstance(item, dict) and query in str(item.get("name", "")).lower():
            symbol_hits.append(item)

    imports = index.get("imports", {})
    if isinstance(imports, dict):
        for file_name, names in imports.items():
            if isinstance(names, list):
                matched = [str(name) for name in names if query in str(name).lower()]
                if matched:
                    import_hits[str(file_name)] = matched

    return truncate(
        json_dumps(
            {
                "query": query,
                "files": file_hits[:20],
                "symbols": symbol_hits[:40],
                "imports": dict(list(import_hits.items())[:20]),
            }
        )
    )


def inspect_function_signature_tool(symbol: str) -> str:
    """查找函数或方法签名。"""

    symbol = symbol.strip()
    if len(symbol) < 2:
        return "inspect_function_signature 需要至少 2 个字符的 symbol。"

    matches: list[dict[str, object]] = []
    for file_name in project_text_files():
        if not file_name.endswith(".py"):
            continue
        tree = _read_python_tree(file_name)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
                args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                if node.args.vararg:
                    args.append("*" + node.args.vararg.arg)
                args.extend(arg.arg for arg in node.args.kwonlyargs)
                if node.args.kwarg:
                    args.append("**" + node.args.kwarg.arg)
                returns = ast.unparse(node.returns) if node.returns else "None"
                matches.append(
                    {
                        "file": file_name,
                        "line": node.lineno,
                        "signature": f"{node.name}({', '.join(args)}) -> {returns}",
                    }
                )
    return json_dumps({"symbol": symbol, "matches": matches[:30]})


def inspect_class_hierarchy_tool(class_name: str) -> str:
    """查找类定义及其父类。"""

    class_name = class_name.strip()
    if len(class_name) < 2:
        return "inspect_class_hierarchy 需要至少 2 个字符的 class_name。"

    matches: list[dict[str, object]] = []
    for file_name in project_text_files():
        if not file_name.endswith(".py"):
            continue
        tree = _read_python_tree(file_name)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                matches.append(
                    {
                        "file": file_name,
                        "line": node.lineno,
                        "bases": [ast.unparse(base) for base in node.bases],
                        "methods": [
                            child.name
                            for child in node.body
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        ],
                    }
                )
    return json_dumps({"class_name": class_name, "matches": matches[:20]})
