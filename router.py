from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .llm_client import chat_completion
from .node_utils import json_loads_from_model
from .state import RiskLevel, State, TaskType


RouterRoute = Literal["search", "write", "chat", "agent"]

TASK_TYPES: tuple[TaskType, ...] = ("search", "write", "chat", "agent")
ROUTER_FALLBACK_ERRORS = (
    RuntimeError,
    ValueError,
    json.JSONDecodeError,
    AttributeError,
    ValidationError,
)

HIGH_RISK_KEYWORDS = (
    "修改代码",
    "修复",
    "删除",
    "移除",
    "写入",
    "覆盖",
    "执行命令",
    "运行命令",
    "apply_patch",
    "patch",
    "回滚",
    "rollback",
    "format_apply",
)
MEDIUM_RISK_KEYWORDS = (
    "运行测试",
    "测试",
    "lint",
    "typecheck",
    "mypy",
    "ruff",
    "cargo",
    "build",
    "编译",
)
AGENT_KEYWORDS = (
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


class RouterDecision(BaseModel):
    """Router / Classifier 的结构化输出。

    中文注释：
    大模型输出不能直接信任。
    这里用 Pydantic 做运行时校验：
    - 不允许 task_type/risk_level 出现未知值。
    - 不允许多余字段混进来。
    - needs_tool 必须能被解析成真正的 bool。
    """

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    risk_level: RiskLevel = "low"
    needs_tool: bool
    reason: str = Field(default="Router 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("reason")
    @classmethod
    def _clean_reason(cls, value: str) -> str:
        cleaned = value.strip()
        return cleaned or "Router 未提供原因。"


def _fallback_task_type(text: str) -> TaskType:
    """Router 失败时使用的本地规则。"""

    if any(keyword in text for keyword in AGENT_KEYWORDS):
        return "agent"
    if any(keyword in text for keyword in ("搜索", "查找", "查询", "找一下")):
        return "search"
    if any(keyword in text for keyword in ("写", "生成", "起草", "润色")):
        return "write"
    return "chat"


def _fallback_risk_level(text: str) -> RiskLevel:
    """Router 失败时使用的本地风险分级。

    中文注释：
    生产级 Router 不能因为 LLM 失败就把风险降成 low。
    高风险关键词必须在本地规则里也能命中。
    """

    if any(keyword in text for keyword in HIGH_RISK_KEYWORDS):
        return "high"
    if any(keyword in text for keyword in MEDIUM_RISK_KEYWORDS):
        return "medium"
    return "low"


def _fallback_decision(text: str, reason: str) -> RouterDecision:
    """构造 fallback RouterDecision。"""

    task_type = _fallback_task_type(text)
    return RouterDecision(
        task_type=task_type,
        risk_level=_fallback_risk_level(text),
        needs_tool=task_type == "agent",
        reason=reason,
        confidence=0.45,
    )


def _parse_router_decision(response: str) -> RouterDecision:
    """把 LLM JSON 转成受控 RouterDecision。"""

    data = json_loads_from_model(response)
    return RouterDecision.model_validate(data)


def router_classifier_node(state: State) -> dict[str, Any]:
    """1. Router / Classifier：判断任务类型、风险等级、是否需要工具。"""

    text = state["user_input"]
    decision_source = "llm"

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
                        '"needs_tool":true,"reason":"一句话原因","confidence":0.8}'
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=240,
        )
        decision = _parse_router_decision(response)
    except ROUTER_FALLBACK_ERRORS:
        decision = _fallback_decision(
            text,
            "Router 兜底规则：LLM 不可用或输出不符合 schema，根据本地规则判断。",
        )
        decision_source = "fallback"

    return {
        "task_type": decision.task_type,
        "risk_level": decision.risk_level,
        "needs_tool": decision.needs_tool,
        "route_reason": decision.reason,
        "next_action": "schedule" if decision.task_type == "agent" else "finish",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Router / Classifier："
                    f"task_type={decision.task_type}, "
                    f"risk_level={decision.risk_level}, "
                    f"needs_tool={decision.needs_tool}, "
                    f"source={decision_source}, "
                    f"confidence={decision.confidence:.2f}。"
                    f"原因：{decision.reason}"
                ),
            }
        ],
    }


def route_by_task(state: State) -> RouterRoute:
    """Router 后的 LangGraph 条件路由函数。"""

    task_type = state["task_type"]
    if task_type not in TASK_TYPES:
        return "chat"
    return task_type
