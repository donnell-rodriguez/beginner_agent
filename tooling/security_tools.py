from __future__ import annotations

import re

from .core import json_dumps, project_text_files, safe_text_file


# 中文注释：
# security_tools.py 放安全扫描工具。
#
# 当前只有一个轻量 secret_scan：
# 用正则扫描明显的 api_key / token / password / private key。
# 生产级可以接入更专业的 secret scanner。
SECRET_PATTERNS = {
    "api_key": re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def secret_scan_tool(path: str = ".") -> str:
    """扫描明显密钥模式。"""

    findings: list[dict[str, object]] = []
    for file_name in project_text_files():
        # 中文注释：
        # path 可以限制扫描范围。
        # 默认 "." 表示扫描整个 beginner_agent。
        if path not in (".", "") and not file_name.startswith(path.rstrip("/") + "/") and file_name != path:
            continue
        text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({"file": file_name, "line": line_number, "kind": name})
    return json_dumps({"status": "failed" if findings else "success", "findings": findings})
