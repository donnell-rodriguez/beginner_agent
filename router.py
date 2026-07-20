from __future__ import annotations

import json
from typing import Any, Literal

from .llm_client import chat_completion
from .node_utils import json_loads_from_model
from .state import State, TaskType


RouterRoute = Literal["search", "write", "chat", "agent"]


def _fallback_task_type(text: str) -> TaskType:
    """Router 失败时使用的本地规则。"""

    if any(
        keyword in text
        for keyword in (
            "项目结构",
            "整体",
            "流程",
            "拆解",
            "计划",
            "理解项目",
            "文件",
            "源码",
            "目录",
            "读取",
            "看一下",
            "列出",
            "修改代码",
            "修复",
            "测试",
            "patch",
            "diff",
            "回滚",
            "lint",
            "typecheck",
        )
    ):
        return "agent"
    if any(keyword in text for keyword in ("搜索", "查找", "查询", "找一下")):
        return "search"
    if any(keyword in text for keyword in ("写", "生成", "起草", "润色")):
        return "write"
    return "chat"


def router_classifier_node(state: State) -> dict[str, Any]:
    """1. Router / Classifier：判断任务类型、风险等级、是否需要工具。"""

    text = state["user_input"]

    try:
        response = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 agent 的 Router / Classifier。"
                        "请判断任务类型、风险等级、是否需要工具。"
                        "task_type 只能是 search、write、chat、agent。"
                        "risk_level 只能是 low、medium、high。"
                        "如果用户需要读取文件、理解项目、查看源码，task_type=agent，needs_tool=true。"
                        "如果用户要求修复代码、修改代码、运行测试、查看 diff，也应该 task_type=agent。"
                        "如果用户要求修改、删除、执行命令，risk_level=high。"
                        "只返回严格 JSON，不要解释。"
                        '格式：{"task_type":"agent","risk_level":"low",'
                        '"needs_tool":true,"reason":"一句话原因"}'
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=240,
        )
        data = json_loads_from_model(response)
        task_type = str(data.get("task_type", "")).lower().strip()
        risk_level = str(data.get("risk_level", "low")).lower().strip()
        if task_type not in ("search", "write", "chat", "agent"):
            raise ValueError("invalid task_type")
        if risk_level not in ("low", "medium", "high"):
            risk_level = "low"
        needs_tool = bool(data.get("needs_tool", task_type == "agent"))
        reason = str(data.get("reason") or "Router 未提供原因。")
    except (RuntimeError, ValueError, json.JSONDecodeError, AttributeError):
        task_type = _fallback_task_type(text)
        risk_level = "low"
        needs_tool = task_type == "agent"
        reason = "Router 兜底规则：根据关键词判断任务类型。"

    return {
        "task_type": task_type,
        "risk_level": risk_level,
        "needs_tool": needs_tool,
        "route_reason": reason,
        "next_action": "schedule" if task_type == "agent" else "finish",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Router / Classifier："
                    f"task_type={task_type}, risk_level={risk_level}, "
                    f"needs_tool={needs_tool}。原因：{reason}"
                ),
            }
        ],
    }


def route_by_task(state: State) -> RouterRoute:
    """Router 后的 LangGraph 条件路由函数。"""

    return state["task_type"]
