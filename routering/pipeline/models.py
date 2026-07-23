from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ...state import RiskLevel, TaskType
from ..models import DecisionSource, RouterDecision


# 中文注释：
# 这里放“多阶段 Router pipeline”内部的数据结构。
# 它们服务 intent/risk/tool/security 子阶段，不直接负责 graph 路由。

StageName = Literal[
    "intent_router",
    "risk_router",
    "tool_needs_router",
    "security_classifier",
    "security_router",
]

# 中文注释：
# StageModelT 是一个“泛型类型变量”。
#
# 从语法上看：
#
#     TypeVar("StageModelT", bound=BaseModel)
#
# 可以拆成三层理解：
#
# 1. TypeVar(...)
#    定义一个临时的类型占位符。
#    它不是运行时业务数据，而是给类型检查器看的。
#
# 2. "StageModelT"
#    这是这个类型变量的名字。
#    通常名字后面带 T，表示 Type。
#
# 3. bound=BaseModel
#    表示 StageModelT 只能代表 BaseModel 或 BaseModel 的子类。
#    也就是说，它可以是 IntentStageModel / RiskStageModel / ToolNeedsStageModel，
#    但不能是普通 str、dict、int。
#
# 为什么这里需要它？
#
# 在 repair.py 里有一个通用函数：
#
#     parse_stage_model_with_repair(..., model_cls: type[StageModelT]) -> tuple[StageModelT, RepairInfo]
#
# 当你传入 IntentStageModel 时，返回值里的 parsed 就会被类型系统理解为 IntentStageModel。
# 当你传入 RiskStageModel 时，返回值里的 parsed 就会被理解为 RiskStageModel。
#
# 这比直接写 BaseModel 更精确：
#
#     不够精确：返回 BaseModel，后面访问 parsed.task_type 类型检查器不一定知道。
#     更精确：返回 StageModelT，传入什么模型类，就返回什么模型实例。
StageModelT = TypeVar("StageModelT", bound=BaseModel)

# 中文注释：
# ROUTER_DECISION_FIELDS 是“Router 子阶段允许输出的字段白名单”。
#
# frozenset(...) 可以理解成“不可修改的 set”：
#
#     普通 set：
#         fields = {"task_type", "risk_level"}
#         fields.add("extra")  # 可以继续修改
#
#     frozenset：
#         fields = frozenset({"task_type", "risk_level"})
#         fields.add("extra")  # 会报错，因为它是不可变的
#
# 为什么这里用 frozenset？
#
# 因为这些字段是固定协议：
# - task_type
# - risk_level
# - needs_tool
# - reason
# - confidence
#
# 后面 repair.py 会用它检查模型输出：
#
#     extra = set(data) - ROUTER_DECISION_FIELDS
#
# 这句话的意思是：
#
#     模型实际返回的字段 - 允许字段白名单 = 多余字段
#
# 如果 extra 不为空，说明 LLM 返回了未治理字段，
# 系统就会进入 repair 或 fallback，避免脏字段混进 Router 决策。
ROUTER_DECISION_FIELDS = frozenset(
    {
        "task_type",
        "risk_level",
        "needs_tool",
        "reason",
        "confidence",
    }
)


class IntentStageModel(BaseModel):
    """Intent Router 的模型输出 schema。"""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    reason: str = Field(default="Intent Router 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class RiskStageModel(BaseModel):
    """Risk Router 的模型输出 schema。"""

    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel
    reason: str = Field(default="Risk Router 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ToolNeedsStageModel(BaseModel):
    """Tool Needs Router 的模型输出 schema。"""

    model_config = ConfigDict(extra="forbid")

    needs_tool: bool
    reason: str = Field(default="Tool Needs Router 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class SecurityClassifierStageModel(BaseModel):
    """LLM Security Classifier 的模型输出 schema。

    中文注释：
    这个阶段只允许“补充/提高风险”，不能降低本地安全策略判断。
    合并逻辑在 security_classifier.py 里。
    """

    model_config = ConfigDict(extra="forbid")

    injection_risk: Literal["none", "suspected", "high"] = "none"
    malicious_intent: Literal[
        "none",
        "prompt_injection",
        "unsafe_code_action",
        "data_exfiltration",
    ] = "none"
    labels: list[str] = Field(default_factory=list)
    reason: str = Field(default="Security Classifier 未提供原因。", min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    deny_reason: str = ""


@dataclass(frozen=True)
class RepairInfo:
    """Router 子阶段 JSON repair 的结果。"""

    attempt_count: int = 0
    success: bool = False
    raw_invalid_response: str = ""
    validation_error_type: str = ""
    final_response: str = ""


@dataclass(frozen=True)
class RouterStageDecision:
    """一个独立 Router 子阶段的判断结果。"""

    stage: StageName
    decision: str
    reason: str
    confidence: float
    source: DecisionSource
    model_response: str = ""
    model_error: str = ""
    fallback_reason: str = ""
    repair_attempt_count: int = 0
    repair_success: bool = False
    raw_invalid_response: str = ""
    validation_error_type: str = ""
    failure_policy_applied: str = ""
    model_name: str = ""
    model_tier: str = ""
    escalation_reason: str = ""

    def failure_audit(self) -> dict[str, Any] | None:
        """生成结构化失败审计信息。"""

        if (
            not self.model_error
            and not self.fallback_reason
            and not self.repair_attempt_count
            and not self.failure_policy_applied
        ):
            return None
        return {
            "stage": self.stage,
            "source": self.source,
            "decision": self.decision,
            "model_error": self.model_error,
            "fallback_reason": self.fallback_reason,
            "repair_attempt_count": self.repair_attempt_count,
            "repair_success": self.repair_success,
            "raw_invalid_response": self.raw_invalid_response,
            "validation_error_type": self.validation_error_type,
            "failure_policy_applied": self.failure_policy_applied,
            "model_name": self.model_name,
            "model_tier": self.model_tier,
            "escalation_reason": self.escalation_reason,
        }


@dataclass(frozen=True)
class MultiStageRouterResult:
    """多阶段 Router 聚合后的结果。"""

    decision: RouterDecision
    stage_decisions: tuple[RouterStageDecision, ...]
    source: DecisionSource
    security: Any | None = None
    model_response: str = ""
    model_error: str = ""
    fallback_reason: str = ""
    failure_audit: tuple[dict[str, Any], ...] = ()
