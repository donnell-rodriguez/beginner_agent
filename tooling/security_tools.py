from __future__ import annotations

from .core import json_dumps, project_text_files, safe_text_file
from ..privacy_governance import scan_text_for_privacy


# 中文注释：
# security_tools.py 放安全扫描工具。
#
# 当前 secret_scan 复用 privacy_governance.py。
# 这样工具扫描、memory 写入、API 脱敏使用同一套规则，
# 不会出现“工具说安全，但 memory 层又认为敏感”的策略分裂。


def secret_scan_tool(path: str = ".") -> str:
    """扫描明显密钥模式。"""

    findings: list[dict[str, object]] = []
    for file_name in project_text_files():
        # 中文注释：
        # path 可以限制扫描范围。
        # 默认 "." 表示扫描整个 beginner_agent。
        is_target = (
            path in (".", "")
            or file_name.startswith(path.rstrip("/") + "/")
            or file_name == path
        )
        if not is_target:
            continue
        text = safe_text_file(file_name).read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            report = scan_text_for_privacy(line)
            for finding in report.findings:
                findings.append(
                    {
                        "file": file_name,
                        "line": line_number,
                        "kind": finding.kind,
                        "category": finding.category,
                        "fingerprint": finding.fingerprint,
                    }
                )
    return json_dumps({"status": "failed" if findings else "success", "findings": findings})
