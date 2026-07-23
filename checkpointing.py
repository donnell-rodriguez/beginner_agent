from __future__ import annotations

import os
import time
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from beginner_agent.config import load_project_env
from beginner_agent.checkpoint_models import (
    CheckpointBackendConfig,
    CheckpointHealth,
    CheckpointPostgresDiagnostics,
)


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


def checkpoint_roundtrip_probe_enabled() -> bool:
    """是否启用 Postgres 写入/读取 roundtrip probe。

    中文注释：
    这个 probe 会在数据库里创建一张很小的 health probe 表，
    然后插入、读取、删除一条测试记录。
    它能验证数据库不只是“能连上”，而是真的能写、能读。
    """

    return _bool_env("BEGINNER_AGENT_CHECKPOINT_HEALTHCHECK_ROUNDTRIP_ENABLED", True)


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
        roundtrip_probe_enabled=checkpoint_roundtrip_probe_enabled(),
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
    diagnostics = CheckpointPostgresDiagnostics()

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
            diagnostics=diagnostics,
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
            diagnostics=diagnostics,
            warnings=warnings,
            errors=errors,
        )

    setup_status = "assumed_runtime_setup"
    if config.healthcheck_enabled:
        diagnostics = _postgres_checkpoint_diagnostics(
            roundtrip_probe_enabled=config.roundtrip_probe_enabled
        )
        setup_status = _setup_status_from_diagnostics(diagnostics)
        if setup_status == "missing_tables":
            warnings.append("Postgres checkpoint 表尚未发现；build_checkpointer().setup() 应在图编译时创建。")
        elif setup_status.startswith("error:"):
            errors.append(setup_status)
        if diagnostics.roundtrip_status.startswith("error:"):
            errors.append(f"Postgres checkpoint roundtrip failed: {diagnostics.roundtrip_status}")
        if diagnostics.waiting_lock_count > 0:
            warnings.append(f"Postgres checkpoint 存在等待锁：{diagnostics.waiting_lock_count}。")

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
        diagnostics=diagnostics,
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


def _postgres_checkpoint_diagnostics(
    *,
    roundtrip_probe_enabled: bool,
) -> CheckpointPostgresDiagnostics:
    """采集 Postgres checkpoint 深度诊断。

    中文注释：
    这里做的是生产级 health check 的本地版本：
    - SELECT 1 延迟。
    - checkpoint 表是否存在。
    - checkpoint_migrations 行数。
    - checkpoint 表/索引大小。
    - 等待锁数量。
    - 当前数据库大小。
    - 可选写入/读取 roundtrip probe。
    """

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        return CheckpointPostgresDiagnostics(notes=["error: missing psycopg dependency"])

    try:
        with psycopg.connect(
            _postgres_database_url(),
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            with conn.cursor() as cursor:
                started = time.perf_counter()
                cursor.execute("SELECT 1 AS ok")
                cursor.fetchone()
                connection_latency_ms = int((time.perf_counter() - started) * 1000)

                checkpoint_tables = _checkpoint_table_names(cursor)
                table_count = len(checkpoint_tables)
                migration_version = _checkpoint_migration_version(cursor, checkpoint_tables)
                table_bytes, index_bytes = _checkpoint_relation_sizes(cursor, checkpoint_tables)
                waiting_lock_count = _checkpoint_waiting_lock_count(cursor, checkpoint_tables)
                database_size_bytes = _database_size_bytes(cursor)
                roundtrip_status = "disabled"
                roundtrip_latency_ms = None
                if roundtrip_probe_enabled:
                    roundtrip_status, roundtrip_latency_ms = _checkpoint_roundtrip_probe(cursor)
                return CheckpointPostgresDiagnostics(
                    connection_latency_ms=connection_latency_ms,
                    roundtrip_status=roundtrip_status,
                    roundtrip_latency_ms=roundtrip_latency_ms,
                    migration_version=migration_version,
                    checkpoint_table_count=table_count,
                    checkpoint_table_bytes=table_bytes,
                    checkpoint_index_bytes=index_bytes,
                    waiting_lock_count=waiting_lock_count,
                    database_size_bytes=database_size_bytes,
                )
    except Exception as exc:  # pragma: no cover - depends on local Postgres availability.
        return CheckpointPostgresDiagnostics(notes=[f"error: {type(exc).__name__}: {exc}"])


def _setup_status_from_diagnostics(diagnostics: CheckpointPostgresDiagnostics) -> str:
    if diagnostics.notes and diagnostics.notes[0].startswith("error:"):
        return diagnostics.notes[0]
    return "ready" if diagnostics.checkpoint_table_count >= 3 else "missing_tables"


def _checkpoint_table_names(cursor: Any) -> list[str]:
    cursor.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
            'checkpoint_migrations',
            'checkpoints',
            'checkpoint_writes'
          )
        ORDER BY table_name
        """
    )
    return [str(row.get("table_name", "")) for row in cursor.fetchall()]


def _checkpoint_migration_version(cursor: Any, checkpoint_tables: list[str]) -> str:
    if "checkpoint_migrations" not in checkpoint_tables:
        return "missing"
    try:
        cursor.execute("SELECT COUNT(*) AS migration_count FROM checkpoint_migrations")
        row = cursor.fetchone() or {}
        return f"rows:{int(row.get('migration_count', 0))}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def _checkpoint_relation_sizes(cursor: Any, checkpoint_tables: list[str]) -> tuple[int, int]:
    if not checkpoint_tables:
        return 0, 0
    cursor.execute(
        """
        SELECT
          COALESCE(SUM(pg_total_relation_size(format('public.%%I', table_name)::regclass)), 0)
            AS table_bytes,
          COALESCE(SUM(pg_indexes_size(format('public.%%I', table_name)::regclass)), 0)
            AS index_bytes
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = ANY(%s)
        """,
        (checkpoint_tables,),
    )
    row = cursor.fetchone() or {}
    return int(row.get("table_bytes", 0)), int(row.get("index_bytes", 0))


def _checkpoint_waiting_lock_count(cursor: Any, checkpoint_tables: list[str]) -> int:
    if not checkpoint_tables:
        return 0
    cursor.execute(
        """
        SELECT COUNT(*) AS waiting_lock_count
        FROM pg_locks
        WHERE NOT granted
          AND relation IN (
            SELECT format('public.%%I', table_name)::regclass
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = ANY(%s)
          )
        """,
        (checkpoint_tables,),
    )
    row = cursor.fetchone() or {}
    return int(row.get("waiting_lock_count", 0))


def _database_size_bytes(cursor: Any) -> int:
    cursor.execute("SELECT pg_database_size(current_database()) AS database_size_bytes")
    row = cursor.fetchone() or {}
    return int(row.get("database_size_bytes", 0))


def _checkpoint_roundtrip_probe(cursor: Any) -> tuple[str, int | None]:
    probe_id = f"probe-{uuid.uuid4()}"
    payload = f"checkpoint-health-{time.time_ns()}"
    started = time.perf_counter()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS beginner_agent_checkpoint_health_probe (
                probe_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO beginner_agent_checkpoint_health_probe (probe_id, payload)
            VALUES (%s, %s)
            """,
            (probe_id, payload),
        )
        cursor.execute(
            """
            SELECT payload
            FROM beginner_agent_checkpoint_health_probe
            WHERE probe_id = %s
            """,
            (probe_id,),
        )
        row = cursor.fetchone() or {}
        cursor.execute(
            "DELETE FROM beginner_agent_checkpoint_health_probe WHERE probe_id = %s",
            (probe_id,),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if str(row.get("payload", "")) != payload:
            return "error: payload_mismatch", latency_ms
        return "ok", latency_ms
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return f"error: {type(exc).__name__}: {exc}", latency_ms


def _bool_env(name: str, default: bool) -> bool:
    load_project_env()
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
