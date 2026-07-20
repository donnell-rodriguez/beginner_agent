from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .core import STATE_DIR, ensure_state_dirs, json_dumps, safe_resolve


# 中文注释：
# audit_tools.py 放“审计记录”工具。
#
# 生产级 code agent 必须回答两个问题：
# - agent 做过什么？
# - 为什么允许它这么做？
#
# 所以工具调用、patch、审批结果都应该记录下来。

AUDIT_LOG_FILE = STATE_DIR / "audit_log.jsonl"


def _append_audit_event(event: dict[str, Any]) -> None:
    """追加一条 JSONL 审计事件。"""

    ensure_state_dirs()
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def audit_tool_call_tool(tool_name: str, tool_args: dict[str, Any], decision: str = "record") -> str:
    """记录一次工具调用。"""

    _append_audit_event(
        {
            "kind": "tool_call",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "decision": decision,
        }
    )
    return "audit_tool_call 成功：已记录工具调用。"


def audit_patch_tool(path: str, reason: str = "", patch_plan_id: str = "") -> str:
    """记录一次代码修改相关事件。"""

    safe_resolve(path)
    _append_audit_event(
        {
            "kind": "patch",
            "path": path,
            "reason": reason,
            "patch_plan_id": patch_plan_id,
        }
    )
    return "audit_patch 成功：已记录 patch 事件。"


def read_audit_log_tool(limit: int = 20) -> str:
    """读取最近的审计事件。"""

    ensure_state_dirs()
    if not AUDIT_LOG_FILE.exists():
        return json_dumps({"events": []})
    lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines[-max(1, limit) :]]
    return json_dumps({"events": events})
