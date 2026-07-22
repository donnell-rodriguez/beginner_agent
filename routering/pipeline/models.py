from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ...state import RiskLevel, TaskType
from ..models import DecisionSource, RouterDecision


# 中文注释：
# 这里放“多阶段 Router pipeline”内部的数据结构。
# 它们服务 intent/risk/tool/security 子阶段，不直接负责 graph 路由。

StageName = Literal["intent_router", "risk_router", "tool_needs_router", "security_router"]
StageModelT = TypeVar("StageModelT", bound=BaseModel)

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
        }


@dataclass(frozen=True)
class MultiStageRouterResult:
    """多阶段 Router 聚合后的结果。"""

    decision: RouterDecision
    stage_decisions: tuple[RouterStageDecision, ...]
    source: DecisionSource
    model_response: str = ""
    model_error: str = ""
    fallback_reason: str = ""
    failure_audit: tuple[dict[str, Any], ...] = ()
