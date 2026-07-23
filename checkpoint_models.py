from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CheckpointBackend = Literal["memory", "postgres"]
CheckpointHealthStatus = Literal["healthy", "warning", "degraded", "blocked"]
CheckpointPersistenceGuarantee = Literal["none", "process", "durable"]
CheckpointFallbackPolicy = Literal["allow_memory", "fail_fast"]


class CheckpointBackendConfig(BaseModel):
    """Checkpoint 后端配置。

    中文注释：
    这是 checkpoint runtime 的配置模型。
    以前 checkpointing.py 直接读 env 并返回字符串；
    现在先把 env 解析成结构化对象，后续 node、observability、测试都能复用。
    """

    model_config = ConfigDict(extra="forbid")

    requested_backend: CheckpointBackend
    effective_backend: CheckpointBackend
    database_url_configured: bool
    allow_memory_fallback: bool
    require_thread_id: bool
    healthcheck_enabled: bool
    checkpoint_namespace: str
    fallback_policy: CheckpointFallbackPolicy
    fallback_reason: str = ""


class CheckpointRecoveryContract(BaseModel):
    """Checkpoint 恢复合同。

    中文注释：
    恢复合同说明“如果 agent 中断了，后续怎么恢复”。
    这里不保存真实业务数据，而是把恢复所需的前提条件写清楚。
    """

    model_config = ConfigDict(extra="forbid")

    requires_thread_id: bool
    thread_id_present: bool
    thread_id_source: str
    checkpoint_namespace: str
    backend_name: CheckpointBackend
    persistence_guarantee: CheckpointPersistenceGuarantee
    fallback_policy: CheckpointFallbackPolicy
    resume_supported: bool
    runtime_entry: str = "builder.compile(checkpointer=build_checkpointer())"
    configured_in: str = "checkpointing.py"


class CheckpointHealth(BaseModel):
    """Checkpoint 健康检查结果。

    中文注释：
    这不是 LangGraph 内部 checkpoint 数据，
    而是我们给 agent 自己看的健康状态：
    当前后端是否可靠？能不能恢复？配置是否完整？
    """

    model_config = ConfigDict(extra="forbid")

    status: CheckpointHealthStatus
    backend: CheckpointBackend
    requested_backend: CheckpointBackend
    persistent: bool
    database_url_configured: bool
    setup_status: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CheckpointReport(BaseModel):
    """写入 State.checkpoint_report 的结构化报告。

    中文注释：
    checkpoint_node 不再手写 dict，而是先构造这个模型。
    最后用 model_dump(mode="json") 转成普通 dict，方便 LangGraph State 保存。
    """

    model_config = ConfigDict(extra="forbid")

    event_type: str = "checkpoint_health"
    run_id: str
    backend: CheckpointBackend
    requested_backend: CheckpointBackend
    persistent: bool
    runtime_owned: bool = True
    state_keys_tracked_by_graph: bool = True
    health: CheckpointHealth
    recovery_contract: CheckpointRecoveryContract
    observability_event: dict[str, Any]
    reason: str

