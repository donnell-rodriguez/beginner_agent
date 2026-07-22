from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..state import RiskLevel, TaskType


DecisionSource = Literal["llm", "fallback", "security_override"]
InjectionRisk = Literal["none", "suspected", "high"]
MaliciousIntent = Literal["none", "prompt_injection", "unsafe_code_action", "data_exfiltration"]


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


@dataclass(frozen=True)
class RouterSecuritySignal:
    """Router 安全分类结果。

    中文注释：
    这不是 Tool Policy 的替代品。
    它是在更早的 Router 层先判断：
    - 用户是否试图覆盖系统规则。
    - 是否有明显的数据外泄意图。
    - 是否要求危险代码动作。

    命中后 Router 会提高 risk_level，后面的 policy/approval 再继续拦截。
    """

    injection_risk: InjectionRisk
    malicious_intent: MaliciousIntent
    labels: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "injection_risk": self.injection_risk,
            "malicious_intent": self.malicious_intent,
            "labels": self.labels,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RouterEvent:
    """Router 可观测事件。

    中文注释：
    大厂系统会把每一次 Router 决策记录下来，方便后续回答：
    - 为什么这个请求进入 agent loop？
    - 为什么风险是 high？
    - 是模型判断的，还是 fallback 判断的？
    - 有没有安全信号命中？
    """

    run_id: str
    user_input: str
    decision: RouterDecision
    source: DecisionSource
    security: RouterSecuritySignal
    latency_ms: int
    fallback_reason: str = ""
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_input": self.user_input,
            "decision": self.decision.model_dump(mode="json"),
            "source": self.source,
            "security": self.security.as_dict(),
            "latency_ms": self.latency_ms,
            "fallback_reason": self.fallback_reason,
            "created_at": self.created_at
            or datetime.now(timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class RouterEvalCase:
    """Router 离线评估样本。

    中文注释：
    线上发现分类错误时，可以把输入和期望输出保存成 eval case。
    后续改 Router prompt / rules / security classifier 时，用这些 case 回放，
    避免修一个场景又弄坏另一个场景。
    """

    user_input: str
    expected_task_type: TaskType
    expected_risk_level: RiskLevel
    expected_needs_tool: bool
    reason: str = ""
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "expected_task_type": self.expected_task_type,
            "expected_risk_level": self.expected_risk_level,
            "expected_needs_tool": self.expected_needs_tool,
            "reason": self.reason,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }
