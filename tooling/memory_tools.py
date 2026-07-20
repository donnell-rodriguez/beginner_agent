from __future__ import annotations

import json
import re
from typing import Any

from .core import CHECKPOINT_DIR, ensure_state_dirs, json_dumps


# 中文注释：
# memory_tools.py 放本地 checkpoint 工具。
#
# 注意：
#   这不是 LangGraph 底层 checkpointer。
#   它是工具层自己的持久化记录，用 JSON 保存 agent 中间数据。
#   生产部署时可以替换成 SQLite/Postgres/S3，但调用接口可以保持不变。
#
def _safe_name(name: str) -> str:
    """把 checkpoint 名字转换成安全文件名。"""

    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name or "checkpoint")


def checkpoint_save_tool(name: str, data: Any = "") -> str:
    """保存 checkpoint 数据。"""

    # 中文注释：
    # ensure_state_dirs 会创建 .agent_state/checkpoints。
    ensure_state_dirs()
    path = CHECKPOINT_DIR / f"{_safe_name(name)}.json"
    path.write_text(json_dumps({"name": name, "data": data}), encoding="utf-8")
    return f"checkpoint_save 成功：{path.name}"


def checkpoint_load_tool(name: str) -> str:
    """读取 checkpoint 数据。"""

    # 中文注释：
    # 这里返回 JSON 字符串，方便 agent 或用户查看。
    ensure_state_dirs()
    path = CHECKPOINT_DIR / f"{_safe_name(name)}.json"
    if not path.exists():
        return f"checkpoint 不存在：{name}"
    try:
        return json_dumps(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
