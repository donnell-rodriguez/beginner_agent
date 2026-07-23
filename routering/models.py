from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..state import RiskLevel, TaskType


# 中文注释：
# routering/models.py 专门放 Router 层的数据结构。
#
# 你可以把它理解成“Router 这个 node 相关的数据合同”：
#
#     routering/nodes.py 负责做事情
#     models.py 负责定义事情产生的数据长什么样
#
# 对于每个 node 来说，先定义清楚数据结构很重要：
# - 输入输出更稳定。
# - 测试更容易写。
# - 后续 observability / eval / audit 可以直接复用这些结构。
# - 不同模块之间不会靠散乱 dict 猜字段。
#
# 在这个文件里：
# - RouterDecision：Router 最终决策。
# - RouterSecuritySignal：安全风险判断。
# - RouterContext：tenant/project/user 上下文。
# - RouterStageReport：多级 Router 每一层的判断记录。
# - RouterEvent：一次 Router 决策的完整审计事件。
# - RouterEvalCase：离线评估样本。


# 中文注释：
# DecisionSource 表示“最终 Router 决策来自哪里”。
# llm：模型输出通过校验后被采用。
# fallback：模型失败、输出非法、或置信度太低，改用本地规则。
# security_override：安全规则或上下文策略把风险提升了。
DecisionSource = Literal["llm", "fallback", "security_override"]

# 中文注释：
# InjectionRisk 表示 prompt injection 风险等级。
# none：没发现注入风险。
# suspected：有可疑注入指令。
# high：注入指令和危险意图同时出现。
InjectionRisk = Literal["none", "suspected", "high"]

# 中文注释：
# MaliciousIntent 表示 Router 层识别到的恶意/危险意图类型。
# 注意它不是最终拒绝策略，只是第一层风险信号。
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

    # 中文注释：
    # task_type 决定 graph.py 下一步走哪个分支。
    # 它必须是 state.py 里定义的 TaskType：
    # search / write / chat / agent。
    task_type: TaskType

    # 中文注释：
    # risk_level 给后续 Tool Policy 使用。
    # high 通常意味着后面需要人工审批或更严格的安全检查。
    risk_level: RiskLevel = "low"

    # 中文注释：
    # needs_tool 表示这个任务是否需要工具。
    # agent 分支通常是 True，普通 chat 通常是 False。
    needs_tool: bool

    # 中文注释：
    # reason 是给人看的解释。
    # 后续调试时，你可以看它理解 Router 为什么这样判断。
    reason: str = Field(default="Router 未提供原因。", min_length=1)

    # 中文注释：
    # confidence 是 Router 对自己判断的置信度。
    # routering/nodes.py 里会用它做低置信度 fallback。
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("reason")
    @classmethod
    def _clean_reason(cls, value: str) -> str:
        # 中文注释：
        # 即使模型返回的是空字符串，也统一替换成默认说明。
        # 这样外部展示和日志里不会出现空 reason。
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

    # 中文注释：
    # malicious_intent 是更具体的危险类型。
    # 例如 data_exfiltration 表示用户可能想读取/泄露敏感信息。
    malicious_intent: MaliciousIntent

    # 中文注释：
    # labels 保存命中的安全标签列表。
    # 一个输入可能同时命中 prompt_injection 和 data_exfiltration。
    labels: list[str]

    # 中文注释：
    # reason 解释为什么命中了这些安全标签。
    reason: str

    # 中文注释：
    # confidence 表示安全分类本身的置信度。
    # 它不是最终 RouterDecision 的 confidence，但会进入审计报告。
    confidence: float = 0.7

    # 中文注释：
    # deny_reason 是安全分类层给后续 Policy / Approval 看的拒绝原因建议。
    # Router 本身不直接拒绝执行，但会把这个原因传下去。
    deny_reason: str = ""

    # 中文注释：
    # source 表示安全信号来自哪里：
    # - local_security_policy：本地规则/正则/历史模式。
    # - llm_security_classifier：LLM 安全分类器提升了风险。
    # - local_security_policy+llm_security_classifier：两者都有贡献。
    source: str = "local_security_policy"

    def as_dict(self) -> dict[str, Any]:
        """把 dataclass 转成普通 dict，方便写入 State / JSONL。"""

        return {
            "injection_risk": self.injection_risk,
            "malicious_intent": self.malicious_intent,
            "labels": self.labels,
            "reason": self.reason,
            "confidence": self.confidence,
            "deny_reason": self.deny_reason,
            "source": self.source,
        }


@dataclass(frozen=True)
class RouterContext:
    """Router 的租户/工作区/项目/用户上下文。

    中文注释：
    大厂系统里，Router 决策通常不是“所有用户一套规则”。
    不同 tenant / workspace / project / user 可能有不同安全策略。
    本地项目先把这些字段记录进事件里，并支持策略提升风险。
    """

    tenant_id: str
    workspace_id: str
    project_id: str
    user_id: str

    def as_dict(self) -> dict[str, str]:
        """把 RouterContext 转成 JSON 友好的 dict。"""

        return {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
        }


@dataclass(frozen=True)
class RouterStageReport:
    """多级 Router 子决策报告。

    中文注释：
    真正生产级 Router 通常会拆成多级：
    - intent：用户想做什么。
    - risk：风险等级。
    - tool_needs：是否需要工具。
    - security：是否命中恶意/注入风险。

    这里先用结构化报告把每级判断记录下来。
    """

    stage: str

    # 中文注释：
    # decision 是这一层的判断结果。
    # 例如 stage="risk" 时，decision 可能是 "high"。
    decision: str

    # 中文注释：
    # reason 解释这一层为什么这样判断。
    reason: str

    # 中文注释：
    # confidence 是这一层判断的置信度。
    confidence: float = 0.7

    def as_dict(self) -> dict[str, Any]:
        """把单个 stage report 转成 dict。"""

        return {
            "stage": self.stage,
            "decision": self.decision,
            "reason": self.reason,
            "confidence": self.confidence,
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

    # 中文注释：
    # decision_id 是这次 Router 决策的唯一标识。
    # 后续审计、查询、排障可以用它串起来。
    decision_id: str

    # 中文注释：
    # run_id 是整次 agent 运行的唯一 ID。
    # 一个 run 里可能有很多事件，RouterEvent 是其中一个。
    run_id: str

    # 中文注释：
    # event_type 表示事件类型。
    # 当前固定是 router_decision，后续可以扩展 router_eval / router_feedback。
    event_type: str

    # 中文注释：
    # user_input 是原始用户输入。
    # 记录它是为了后续复盘 Router 为什么这样分类。
    user_input: str

    # 中文注释：
    # decision 是最终 RouterDecision。
    # 它是经过 LLM / fallback / security / context policy 后的最终结果。
    decision: RouterDecision

    # 中文注释：
    # source 表示最终决策来源。
    source: DecisionSource

    # 中文注释：
    # context 记录当前租户/工作区/项目/用户。
    context: RouterContext

    # 中文注释：
    # stage_reports 记录多级 Router 每一层的判断。
    stage_reports: list[RouterStageReport]

    # 中文注释：
    # security 保存安全分类结果。
    security: RouterSecuritySignal

    # 中文注释：
    # latency_ms 是 Router 本次决策耗时。
    # 这是可观测性的重要指标。
    latency_ms: int

    # 中文注释：
    # model_response 保存模型原始响应的截断版本。
    # 它用于排查模型为什么输出某个结果。
    model_response: str = ""

    # 中文注释：
    # model_error 保存模型调用或解析失败的错误信息。
    model_error: str = ""

    # 中文注释：
    # fallback_reason 记录为什么使用 fallback。
    fallback_reason: str = ""

    # 中文注释：
    # failure_audit 保存 Router 子阶段失败/修复/保守策略的结构化记录。
    # 它避免把关键失败信息只塞进 reason 字符串里。
    failure_audit: tuple[dict[str, Any], ...] = ()

    # 中文注释：
    # governance_contract 记录本次 Router 用到的版本合同：
    # router / prompt / rules / security policy / stage budget。
    governance_contract: dict[str, Any] = field(default_factory=dict)

    # 中文注释：
    # conflicts 记录 LLM、规则、安全、上下文策略之间的冲突。
    conflicts: tuple[dict[str, Any], ...] = ()

    # 中文注释：
    # metrics_snapshot 是本次事件写入后得到的指标快照。
    # 它让本地调试也能看到 request_total、fallback_total、latency 等趋势。
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)

    # 中文注释：
    # review 记录是否进入人工复核队列。
    review: dict[str, Any] = field(default_factory=dict)

    # 中文注释：
    # sanitized_input 记录是否对 prompt 输入做了 secret / PII 脱敏。
    sanitized_input: dict[str, Any] = field(default_factory=dict)

    # 中文注释：
    # created_at 是事件创建时间。
    # 如果外部没有传入，就在 as_dict() 里自动生成当前 UTC 时间。
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        """把 RouterEvent 转成普通 dict，方便写入 State / JSONL。"""

        return {
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "user_input": self.user_input,
            "decision": self.decision.model_dump(mode="json"),
            "source": self.source,
            "context": self.context.as_dict(),
            "stage_reports": [report.as_dict() for report in self.stage_reports],
            "security": self.security.as_dict(),
            "latency_ms": self.latency_ms,
            "model_response": self.model_response,
            "model_error": self.model_error,
            "fallback_reason": self.fallback_reason,
            "failure_audit": list(self.failure_audit),
            "governance_contract": self.governance_contract,
            "conflicts": list(self.conflicts),
            "metrics_snapshot": self.metrics_snapshot,
            "review": self.review,
            "sanitized_input": self.sanitized_input,
            "created_at": self.created_at
            # .isoformat() 会把 datetime 对象转换为字符串
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

    # 中文注释：
    # 下面三个 expected_* 字段是“期望 Router 输出”。
    # 后续 evaluate_router_prediction(...) 会拿当前 Router 的实际输出和它对比。
    expected_task_type: TaskType
    expected_risk_level: RiskLevel
    expected_needs_tool: bool

    # 中文注释：
    # reason 说明为什么这个 case 的期望结果是这样。
    reason: str = ""

    # 中文注释：
    # category 用于把 eval dataset 分层。
    # 例如 normal_chat / code_agent / high_risk / prompt_injection / secret_pii。
    category: str = "general"

    # 中文注释：
    # tags 保存更细的样本标签，方便后续统计哪个能力退化。
    tags: tuple[str, ...] = ()

    # 中文注释：
    # created_at 保存 eval case 生成时间。
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        """把 RouterEvalCase 转成 JSON 友好的 dict。"""

        return {
            "user_input": self.user_input,
            "expected_task_type": self.expected_task_type,
            "expected_risk_level": self.expected_risk_level,
            "expected_needs_tool": self.expected_needs_tool,
            "reason": self.reason,
            "category": self.category,
            "tags": list(self.tags),
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }
