from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CheckpointBackend = Literal["memory", "postgres"]
CheckpointHealthStatus = Literal["healthy", "warning", "degraded", "blocked"]
CheckpointPersistenceGuarantee = Literal["none", "process", "durable"]
CheckpointFallbackPolicy = Literal["allow_memory", "fail_fast"]
CheckpointSetupMode = Literal["auto", "manual"]


class CheckpointFallbackRiskDecision(BaseModel):
    """运行时 fallback 风险决策。

    中文注释：
    env 只能说明“技术上允不允许 fallback”。
    但生产级 agent 还要结合当前任务风险判断：

    - 高风险代码修改不能悄悄 fallback 到 memory。
    - 长任务不能悄悄 fallback 到 memory。
    - 需要人工审批的任务不能悄悄 fallback 到 memory。

    所以这个模型记录“本轮任务是否允许使用 memory fallback”以及原因。
    """

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    requires_persistence: bool
    reason: str
    risk_factors: list[str] = Field(default_factory=list)
    policy_source: str = "checkpoint_policy.py"
    long_task_step_threshold: int = 8


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
    roundtrip_probe_enabled: bool
    auto_setup_enabled: bool
    setup_mode: CheckpointSetupMode
    checkpoint_namespace: str
    fallback_policy: CheckpointFallbackPolicy
    fallback_reason: str = ""


class CheckpointSetupReport(BaseModel):
    """Checkpoint schema setup / migration 报告。

    中文注释：
    生产环境不建议应用启动时自动建表。
    这个模型给独立运维脚本使用：

        python scripts/manage_checkpoint_schema.py setup

    它会记录 setup 是否执行、执行前后 schema 状态是什么。
    """

    model_config = ConfigDict(extra="forbid")

    backend: CheckpointBackend
    setup_mode: CheckpointSetupMode
    setup_executed: bool
    status_before: str
    status_after: str
    migration_version_before: str = "unknown"
    migration_version_after: str = "unknown"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CheckpointPostgresDiagnostics(BaseModel):
    """Postgres checkpoint 深度健康诊断。

    中文注释：
    生产级 health check 不只看“能不能连上数据库”。
    它还会看延迟、checkpoint 表、迁移状态、锁等待、数据库大小、
    以及能不能真实写入再读回一条 probe 记录。
    """

    model_config = ConfigDict(extra="forbid")

    connection_latency_ms: int | None = None
    roundtrip_status: str = "not_run"
    roundtrip_latency_ms: int | None = None
    migration_version: str = "unknown"
    checkpoint_table_count: int = 0
    checkpoint_table_bytes: int = 0
    checkpoint_index_bytes: int = 0
    waiting_lock_count: int = 0
    database_size_bytes: int = 0
    probe_table: str = "beginner_agent_checkpoint_health_probe"
    notes: list[str] = Field(default_factory=list)


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
    diagnostics: CheckpointPostgresDiagnostics = Field(default_factory=CheckpointPostgresDiagnostics)
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
    fallback_risk_decision: CheckpointFallbackRiskDecision
    observability_event: dict[str, Any]
    reason: str
