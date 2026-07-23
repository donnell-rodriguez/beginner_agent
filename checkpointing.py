from __future__ import annotations

import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from beginner_agent.config import load_project_env
from beginner_agent.checkpoint_models import CheckpointBackendConfig, CheckpointHealth


# 中文注释：
# checkpointing.py 是直接读取 DATABASE_URL 的模块。
# 这里显式加载 .env，让这个文件单独阅读时也能看清楚配置来源。
load_project_env()


def checkpoint_backend_name() -> str:
    """读取当前 checkpoint backend。

    中文注释：
    checkpoint 和 memory 不是一回事：

    - memory.py 保存“经验、记忆、历史结果”。
    - checkpointing.py 保存“LangGraph 运行中间状态”。

    如果你已经配置 BEGINNER_AGENT_MEMORY_BACKEND=postgres，
    这里默认也会使用 postgres checkpoint。
    """

    configured = os.getenv("BEGINNER_AGENT_CHECKPOINT_BACKEND", "").strip().lower()
    if configured:
        return configured
    memory_backend = os.getenv("BEGINNER_AGENT_MEMORY_BACKEND", "").strip().lower()
    if memory_backend == "postgres":
        return "postgres"
    return "memory"


def checkpoint_allow_memory_fallback() -> bool:
    """是否允许 Postgres checkpoint 配置异常时降级到 memory。

    中文注释：
    本地开发可以设置为 true，避免数据库没启动时完全不能跑。
    生产环境通常应该设置为 false，让配置错误尽早暴露。
    """

    return _bool_env("BEGINNER_AGENT_CHECKPOINT_ALLOW_MEMORY_FALLBACK", False)


def checkpoint_require_thread_id() -> bool:
    """是否要求运行时必须具备 thread_id。"""

    return _bool_env("BEGINNER_AGENT_CHECKPOINT_REQUIRE_THREAD_ID", True)


def checkpoint_healthcheck_enabled() -> bool:
    """是否启用真实 Postgres 连接健康检查。"""

    return _bool_env("BEGINNER_AGENT_CHECKPOINT_HEALTHCHECK_ENABLED", True)


def checkpoint_namespace() -> str:
    """Checkpoint 命名空间。

    中文注释：
    当前 LangGraph runtime 的 namespace 主要由 thread_id/config 控制。
    这里的 namespace 是 beginner_agent 自己的报告字段，
    用来把不同环境或不同 agent 的 checkpoint 报告区分开。
    """

    load_project_env()
    namespace = os.getenv("BEGINNER_AGENT_CHECKPOINT_NAMESPACE", "beginner_agent").strip()
    return namespace or "beginner_agent"


def checkpoint_database_url_configured() -> bool:
    """是否配置了 checkpoint 可用的数据库连接串。"""

    load_project_env()
    return bool(
        os.getenv("BEGINNER_AGENT_CHECKPOINT_DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )


def checkpoint_backend_config() -> CheckpointBackendConfig:
    """返回结构化 checkpoint 配置。"""

    requested = checkpoint_backend_name()
    if requested not in {"memory", "postgres"}:
        raise ValueError(
            "BEGINNER_AGENT_CHECKPOINT_BACKEND 只能是 memory 或 postgres，"
            f"当前是：{requested}"
        )

    database_url_configured = checkpoint_database_url_configured()
    allow_fallback = checkpoint_allow_memory_fallback()
    effective = requested
    fallback_reason = ""
    if requested == "postgres" and not database_url_configured and allow_fallback:
        effective = "memory"
        fallback_reason = "Postgres checkpoint 缺少数据库连接串，已按配置 fallback 到 memory。"

    return CheckpointBackendConfig(
        requested_backend=requested,
        effective_backend=effective,
        database_url_configured=database_url_configured,
        allow_memory_fallback=allow_fallback,
        require_thread_id=checkpoint_require_thread_id(),
        healthcheck_enabled=checkpoint_healthcheck_enabled(),
        checkpoint_namespace=checkpoint_namespace(),
        fallback_policy="allow_memory" if allow_fallback else "fail_fast",
        fallback_reason=fallback_reason,
    )


def _postgres_database_url() -> str:
    """返回 checkpoint 使用的 Postgres 连接串。

    中文注释：
    生产级代码不应该在源码里写死数据库账号、密码、主机和端口。

    原因很简单：
    - 本地、测试、生产环境的数据库地址通常不同。
    - 数据库密码属于敏感配置，不应该进入 Git。
    - 如果缺少配置，应该明确报错，而不是悄悄连接某个默认数据库。

    所以这里只读取环境变量：
    - BEGINNER_AGENT_CHECKPOINT_DATABASE_URL：checkpoint 专用数据库。
    - DATABASE_URL：项目通用数据库。
    """

    database_url = (
        os.getenv("BEGINNER_AGENT_CHECKPOINT_DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if not database_url:
        raise RuntimeError(
            "Postgres checkpoint 需要配置数据库连接串。"
            "请设置 BEGINNER_AGENT_CHECKPOINT_DATABASE_URL 或 DATABASE_URL。"
        )
    return database_url


def check_checkpoint_health() -> CheckpointHealth:
    """检查 checkpoint runtime 健康状态。

    中文注释：
    这一步给 checkpoint_node / observability 用。
    它不会保存 checkpoint；真正保存仍由 LangGraph runtime 负责。
    """

    config = checkpoint_backend_config()
    warnings: list[str] = []
    errors: list[str] = []
    setup_status = "not_required"

    if config.fallback_reason:
        warnings.append(config.fallback_reason)

    if config.effective_backend == "memory":
        if config.requested_backend == "postgres":
            status = "degraded" if config.allow_memory_fallback else "blocked"
        else:
            status = "warning"
            warnings.append("Memory checkpoint 只适合本地单进程实验；进程退出后无法恢复。")
        return CheckpointHealth(
            status=status,
            backend="memory",
            requested_backend=config.requested_backend,
            persistent=False,
            database_url_configured=config.database_url_configured,
            setup_status=setup_status,
            warnings=warnings,
            errors=errors,
        )

    if not config.database_url_configured:
        errors.append("Postgres checkpoint 需要 BEGINNER_AGENT_CHECKPOINT_DATABASE_URL 或 DATABASE_URL。")
        return CheckpointHealth(
            status="blocked",
            backend="postgres",
            requested_backend=config.requested_backend,
            persistent=True,
            database_url_configured=False,
            setup_status="unknown",
            warnings=warnings,
            errors=errors,
        )

    setup_status = "assumed_runtime_setup"
    if config.healthcheck_enabled:
        setup_status = _postgres_checkpoint_setup_status()
        if setup_status == "missing_tables":
            warnings.append("Postgres checkpoint 表尚未发现；build_checkpointer().setup() 应在图编译时创建。")
        elif setup_status.startswith("error:"):
            errors.append(setup_status)

    status = "healthy"
    if errors:
        status = "blocked"
    elif warnings:
        status = "warning"
    return CheckpointHealth(
        status=status,
        backend="postgres",
        requested_backend=config.requested_backend,
        persistent=True,
        database_url_configured=True,
        setup_status=setup_status,
        warnings=warnings,
        errors=errors,
    )


def _postgres_checkpointer() -> Any:
    """创建 PostgresSaver checkpointer。

    中文注释：
    这里不使用：

        with PostgresSaver.from_conn_string(...) as checkpointer:

    原因是 build_graph() 需要返回一个长期可用的 compiled graph。
    如果用 with，build_graph() 返回后连接就关闭了。

    所以这里手动创建 psycopg connection，并把连接交给 PostgresSaver 保存。
    """

    try:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "Postgres checkpoint 需要安装 langgraph-checkpoint-postgres 和 psycopg。"
            "请运行 uv sync，或确认 pyproject.toml 里包含 langgraph-checkpoint-postgres。"
        ) from exc

    conn = psycopg.connect(
        _postgres_database_url(),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    checkpointer = PostgresSaver(conn)
    # 中文注释：
    # 第一次使用 Postgres checkpoint 时必须 setup。
    # 它会创建 checkpoint_migrations / checkpoints / checkpoint_writes 等表。
    checkpointer.setup()
    return checkpointer


def build_checkpointer() -> Any:
    """创建 graph.py 使用的 LangGraph checkpointer。

    中文注释：
    这个函数是 graph.py 和具体 checkpoint 后端之间的隔离层。
    graph.py 不需要知道 PostgresSaver 怎么初始化，也不需要知道数据库表怎么创建。

    当前支持：
    - memory：单进程内存 checkpoint，适合教学和临时实验。
    - postgres：持久化 checkpoint，适合本地长任务和恢复。
    """

    config = checkpoint_backend_config()
    backend = config.effective_backend
    if backend == "memory":
        return MemorySaver()
    if backend == "postgres":
        return _postgres_checkpointer()
    raise ValueError(
        "BEGINNER_AGENT_CHECKPOINT_BACKEND 只能是 memory 或 postgres，"
        f"当前是：{backend}"
    )


def _postgres_checkpoint_setup_status() -> str:
    """检查 Postgres checkpoint 表是否已经存在。"""

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        return "error: missing psycopg dependency"

    try:
        with psycopg.connect(
            _postgres_database_url(),
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) AS table_count
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name IN (
                        'checkpoint_migrations',
                        'checkpoints',
                        'checkpoint_writes'
                      )
                    """
                )
                row = cursor.fetchone() or {}
                return "ready" if int(row.get("table_count", 0)) >= 3 else "missing_tables"
    except Exception as exc:  # pragma: no cover - depends on local Postgres availability.
        return f"error: {type(exc).__name__}: {exc}"


def _bool_env(name: str, default: bool) -> bool:
    load_project_env()
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
