from __future__ import annotations

import json

from .command_tools import run_pytest_tool
from .core import json_dumps, project_text_files, safe_text_file


# 中文注释：
# test_selection_tools.py 放“影响测试选择”工具。
#
# 真实 code agent 不会每次都跑全量测试。
# 更常见做法是：
#   1. 看改了哪些文件。
#   2. 找可能相关的测试。
#   3. 先跑 impacted tests。
#   4. 关键改动再跑全量测试。


def _is_test_file(path: str) -> bool:
    """判断一个文件名是否像测试文件。"""

    parts = path.split("/")
    name = parts[-1]
    return "test" in parts or name.startswith("test_") or name.endswith("_test.py")


def map_changed_files_to_tests_tool(changed_files: list[str] | str) -> str:
    """根据变更文件推断相关测试文件。"""

    if isinstance(changed_files, str):
        files = [item.strip() for item in changed_files.split(",") if item.strip()]
    else:
        files = [str(item).strip() for item in changed_files if str(item).strip()]

    known_files = set(project_text_files())
    test_files = sorted(file_name for file_name in known_files if _is_test_file(file_name))
    mapping: dict[str, list[str]] = {}

    for changed in files:
        if changed not in known_files:
            mapping[changed] = []
            continue
        stem = changed.rsplit("/", 1)[-1].removesuffix(".py")
        candidates = [
            test_file
            for test_file in test_files
            if stem in test_file or stem.removeprefix("_") in test_file
        ]
        if not candidates and test_files:
            candidates = test_files[:5]
        mapping[changed] = candidates[:10]

    return json_dumps({"changed_files": files, "mapping": mapping, "known_test_files": test_files[:30]})


def select_relevant_tests_tool(query: str = "", changed_files: list[str] | str = "") -> str:
    """选择最值得优先运行的测试。"""

    known_files = set(project_text_files())
    tests = sorted(file_name for file_name in known_files if _is_test_file(file_name))
    selected: list[str] = []

    if changed_files:
        mapped = json.loads(map_changed_files_to_tests_tool(changed_files))
        mapping = mapped.get("mapping", {})
        if isinstance(mapping, dict):
            for candidates in mapping.values():
                if isinstance(candidates, list):
                    selected.extend(str(item) for item in candidates)

    query_lower = query.lower().strip()
    if query_lower:
        for test_file in tests:
            text = safe_text_file(test_file).read_text(encoding="utf-8", errors="replace").lower()
            if query_lower in test_file.lower() or query_lower in text:
                selected.append(test_file)

    if not selected:
        selected = tests[:10]

    unique = list(dict.fromkeys(selected))
    return json_dumps({"selected_tests": unique[:20], "reason": "优先运行与变更文件或查询目标相关的测试。"})


def run_impacted_tests_tool(changed_files: list[str] | str = "") -> str:
    """运行受控 impacted tests。

    中文注释：
    当前本地白名单只允许 pytest beginner_agent。
    如果后续项目有 tests 目录，可以把 command profile 扩展成更细的 targeted pytest。
    """

    selected = select_relevant_tests_tool(changed_files=changed_files)
    pytest_result = json.loads(run_pytest_tool("beginner_agent"))
    status = pytest_result.get("status", "unknown")
    if status == "skipped":
        status = "partial"
    return json_dumps({"status": status, "selection": json.loads(selected), "pytest": pytest_result})
