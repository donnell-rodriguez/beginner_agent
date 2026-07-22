from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .settings import (
    DEFAULT_PROJECT_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_USER_ID,
    DEFAULT_WORKSPACE_ID,
)

MemoryKind = Literal["task", "failure", "patch", "project", "user", "tool", "eval"]
MemoryScope = Literal["global", "user", "project", "thread", "task", "tool", "file"]
MemoryVisibility = Literal[
    "private",
    "project",
    "workspace",
    "tenant",
    "public",
    "retrieval_only",
]
SensitivityLevel = Literal["public", "internal", "confidential", "secret"]
RetentionPolicy = Literal["none", "session", "ttl", "long_term", "pinned"]
ValidityStatus = Literal["active", "superseded", "deprecated", "disputed", "rejected"]
MemoryPolicyAction = Literal["store", "discard"]
MemoryAuditAction = Literal[
    "store",
    "discard",
    "supersede",
    "promote",
    "compact",
    "expire",
    "deprioritize",
    "contradiction_check",
    "summarize",
    "rebuild_embedding",
    "sensitive_access",
    "fallback",
    "retrieve",
]
MemoryWriterRoute = Literal["schedule", "compact", "finish"]

class MemoryRecord(BaseModel):
    """结构化记忆记录。

    中文注释：
    生产级 agent 的 memory 不应该只是随便塞一个 dict。
    至少要知道：
    - 这条记忆是什么类型。
    - 来自哪个 task/tool。
    - 是否成功。
    - 跟哪些路径相关。
    - 什么时候产生。
    - 是否可信。
    - 质量评分、信任评分、衰减评分是多少。
    - 适用范围是什么。
    - 保留多久。
    - 当前是否仍然有效。

    现在使用 Pydantic，而不是普通 dict / dataclass。
    好处是：
    - 写入前做运行时校验。
    - 字段类型更明确。
    - 后续可以直接导出 JSON Schema。
    - 更接近生产级 agent 的 memory record 设计。
    """
    # 不允许出现未定义字段。
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: MemoryKind
    task_id: str
    title: str
    summary: str
    status: str
    tool_name: str
    tool_result_status: str
    paths: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    importance: float = 0.5
    quality_score: float = 0.5
    trust_score: float = 0.5
    decay_score: float = 0.0
    scope: MemoryScope = "project"
    visibility: MemoryVisibility = "project"
    sensitivity_level: SensitivityLevel = "internal"
    tenant_id: str = DEFAULT_TENANT_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    project_id: str = DEFAULT_PROJECT_ID
    user_id: str = DEFAULT_USER_ID
    retention_policy: RetentionPolicy = "ttl"
    validity_status: ValidityStatus = "active"
    pinned: bool = False
    expires_at: str | None = None
    supersedes: str | None = None
    contradiction_key: str | None = None
    source: str = "memory_writer_node"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "confidence",
        "importance",
        "quality_score",
        "trust_score",
        "decay_score",
    )
    @classmethod
    def _confidence_between_zero_and_one(cls, value: float) -> float:
        """限制 confidence / importance 在 0 到 1 之间。"""

        if value < 0 or value > 1:
            raise ValueError("memory score 字段必须在 0 到 1 之间。")
        return value


class MemoryAuditEvent(BaseModel):
    """记忆治理审计事件。

    中文注释：
    生产级 memory 系统不能只保存“最后结果”，
    还要保存“为什么这么做”。
    例如：
    - 为什么这条记忆被保存？
    - 为什么这条记忆被丢弃？
    - 哪条旧记忆被 superseded？
    - 哪条记忆因为多次成功被 promotion？
    - 检索时哪些记忆进入了上下文？

    这类信息不直接参与 agent 推理，但对排查问题、复盘策略、
    调整 memory policy 很重要。
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    action: MemoryAuditAction
    memory_id: str
    reason: str
    backend: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# frozen=True = 这个对象创建后就冻结，不允许再改字段
# 这种结果最好是“确定后不再被随手改掉”。否则后面某个函数
# 不小心把 action 从 "discard" 改成 "store"，memory 系统就会出现难查的 bug。
@dataclass(frozen=True)
class MemoryPolicyDecision:
    """MemoryPolicy 的判断结果。

    中文注释：
    这层专门回答一个问题：

        pending_memory 到底要不要写入长期记忆？

    它让 Memory Writer 不再“看到 pending_memory 就无脑写入”。
    """

    action: MemoryPolicyAction
    reason: str
    scope: MemoryScope = "project"
    retention_policy: RetentionPolicy = "ttl"
    importance: float = 0.5
    pinned: bool = False
    expires_at: str | None = None
    validity_status: ValidityStatus = "active"

def memory_record_json_schema() -> dict[str, Any]:
    """导出 MemoryRecord 的 JSON Schema。"""
    # model_json_schema() 是从 Pydantic 的 BaseModel 继承来的方法
    return MemoryRecord.model_json_schema()


class MemoryStore(Protocol):
    """长期记忆存储协议。

    中文注释：
    节点和治理逻辑只依赖这个协议，不直接绑定 Postgres 或 JSONL。
    这样后续替换成 Redis、专用向量库或远程 memory service 时，
    上层节点不需要大改。
    """

    backend_name: str

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        """列出最近的 memory records。"""

    def search_similar_records(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        """按语义相似度检索 memory records。"""

    def upsert_record(self, record: MemoryRecord) -> None:
        """写入或更新 memory record。"""

    def mark_records_status(
        self,
        memory_ids: list[str],
        status: ValidityStatus,
        *,
        superseded_by: str | None = None,
    ) -> None:
        """批量更新 memory 状态。"""

    def cleanup_expired_records(self) -> int:
        """清理过期 memory。"""

    def rebuild_embeddings(self, limit: int) -> int:
        """重建 embedding。"""

    def upsert_audit_event(self, event: MemoryAuditEvent) -> None:
        """写入 memory audit event。"""
