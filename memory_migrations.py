from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal


MigrationDirection = Literal["up", "down"]


@dataclass(frozen=True)
class MemoryMigration:
    """一条 Postgres memory schema migration。

    中文注释：
    生产级数据库结构不能散落在业务类里。
    每次 schema 变化都应该有稳定版本号、up SQL、down SQL。
    这样才能知道：
    - 当前数据库迁移到了哪个版本。
    - 新版本如何升级。
    - 出问题时如何回滚。
    """

    version: int
    name: str
    up_sql: tuple[str, ...]
    down_sql: tuple[str, ...]


MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS beginner_agent_memory_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL,
    checksum TEXT NOT NULL
)
"""


MIGRATIONS: tuple[MemoryMigration, ...] = (
    MemoryMigration(
        version=1,
        name="base_memory_tables",
        up_sql=(
            "CREATE EXTENSION IF NOT EXISTS vector",
            """
            CREATE TABLE IF NOT EXISTS beginner_agent_memory (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                task_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                status TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_result_status TEXT NOT NULL,
                paths JSONB NOT NULL DEFAULT '[]'::jsonb,
                tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 0.7,
                importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                trust_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                decay_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                scope TEXT NOT NULL DEFAULT 'project',
                visibility TEXT NOT NULL DEFAULT 'project',
                sensitivity_level TEXT NOT NULL DEFAULT 'internal',
                tenant_id TEXT NOT NULL DEFAULT 'local-tenant',
                workspace_id TEXT NOT NULL DEFAULT 'local-workspace',
                project_id TEXT NOT NULL DEFAULT 'beginner_agent',
                user_id TEXT NOT NULL DEFAULT 'local-user',
                retention_policy TEXT NOT NULL DEFAULT 'ttl',
                validity_status TEXT NOT NULL DEFAULT 'active',
                pinned BOOLEAN NOT NULL DEFAULT FALSE,
                expires_at TIMESTAMPTZ,
                supersedes TEXT,
                contradiction_key TEXT,
                source TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS beginner_agent_memory_audit (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                backend TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
        ),
        down_sql=(
            "DROP TABLE IF EXISTS beginner_agent_memory_audit",
            "DROP TABLE IF EXISTS beginner_agent_memory",
        ),
    ),
    MemoryMigration(
        version=2,
        name="compat_columns_for_existing_memory",
        up_sql=(
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS importance DOUBLE PRECISION NOT NULL DEFAULT 0.5",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.5",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS trust_score DOUBLE PRECISION NOT NULL DEFAULT 0.5",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS decay_score DOUBLE PRECISION NOT NULL DEFAULT 0.0",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS scope TEXT NOT NULL DEFAULT 'project'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'project'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS sensitivity_level TEXT NOT NULL DEFAULT 'internal'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'local-tenant'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'local-workspace'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS project_id TEXT NOT NULL DEFAULT 'beginner_agent'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'local-user'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS retention_policy TEXT NOT NULL DEFAULT 'ttl'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS validity_status TEXT NOT NULL DEFAULT 'active'",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS supersedes TEXT",
            "ALTER TABLE beginner_agent_memory ADD COLUMN IF NOT EXISTS contradiction_key TEXT",
        ),
        down_sql=(
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS contradiction_key",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS supersedes",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS expires_at",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS pinned",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS validity_status",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS retention_policy",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS user_id",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS project_id",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS workspace_id",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS tenant_id",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS sensitivity_level",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS visibility",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS scope",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS decay_score",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS trust_score",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS quality_score",
            "ALTER TABLE beginner_agent_memory DROP COLUMN IF EXISTS importance",
        ),
    ),
    MemoryMigration(
        version=3,
        name="memory_indexes",
        up_sql=(
            "CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_kind ON beginner_agent_memory (kind)",
            "CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_task_id ON beginner_agent_memory (task_id)",
            "CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_tool_status ON beginner_agent_memory (tool_name, tool_result_status)",
            "CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_created_at ON beginner_agent_memory (created_at DESC)",
            """
            CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_governance
            ON beginner_agent_memory (
                validity_status, retention_policy, scope, pinned,
                expires_at, quality_score, trust_score
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_acl
            ON beginner_agent_memory (
                tenant_id, workspace_id, project_id, user_id,
                visibility, sensitivity_level
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_tags ON beginner_agent_memory USING GIN (tags)",
            "CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_audit_memory_id ON beginner_agent_memory_audit (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_audit_created_at ON beginner_agent_memory_audit (created_at DESC)",
        ),
        down_sql=(
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_audit_created_at",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_audit_memory_id",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_tags",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_acl",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_governance",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_created_at",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_tool_status",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_task_id",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_kind",
        ),
    ),
    MemoryMigration(
        version=4,
        name="embedding_index_cleanup",
        up_sql=("DROP INDEX IF EXISTS idx_beginner_agent_memory_embeddings_vector",),
        down_sql=(),
    ),
    MemoryMigration(
        version=5,
        name="memory_lifecycle_run_history",
        up_sql=(
            """
            CREATE TABLE IF NOT EXISTS beginner_agent_memory_lifecycle_runs (
                run_id TEXT PRIMARY KEY,
                run_key TEXT NOT NULL,
                job_name TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                locked BOOLEAN NOT NULL DEFAULT FALSE,
                skipped_reason TEXT NOT NULL DEFAULT '',
                backend TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                report JSONB NOT NULL DEFAULT '{}'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_lifecycle_runs_key_success
            ON beginner_agent_memory_lifecycle_runs (job_name, run_key)
            WHERE status = 'success'
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_beginner_agent_memory_lifecycle_runs_status
            ON beginner_agent_memory_lifecycle_runs (job_name, status, started_at DESC)
            """,
        ),
        down_sql=(
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_lifecycle_runs_status",
            "DROP INDEX IF EXISTS idx_beginner_agent_memory_lifecycle_runs_key_success",
            "DROP TABLE IF EXISTS beginner_agent_memory_lifecycle_runs",
        ),
    ),
)


def _checksum(migration: MemoryMigration) -> str:
    """生成简单 checksum，防止同一版本 SQL 被无声修改。"""

    import hashlib

    raw = "\n".join([str(migration.version), migration.name, *migration.up_sql])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _connect(database_url: str):
    import psycopg

    return psycopg.connect(database_url)


def _applied_versions(conn: Any) -> dict[int, str]:
    conn.execute(MIGRATION_TABLE_SQL)
    rows = conn.execute(
        """
        SELECT version, checksum
        FROM beginner_agent_memory_schema_migrations
        ORDER BY version
        """
    ).fetchall()
    return {int(row[0]): str(row[1]) for row in rows}


def pending_memory_migrations(database_url: str) -> list[dict[str, Any]]:
    """查看待执行 migrations。"""

    with _connect(database_url) as conn:
        applied = _applied_versions(conn)
    return [
        {
            "version": migration.version,
            "name": migration.name,
            "checksum": _checksum(migration),
        }
        for migration in MIGRATIONS
        if migration.version not in applied
    ]


def current_memory_schema_version(database_url: str) -> int:
    """读取当前 memory schema version。"""

    with _connect(database_url) as conn:
        applied = _applied_versions(conn)
    return max(applied.keys(), default=0)


def run_memory_migrations(database_url: str) -> dict[str, Any]:
    """执行所有待升级 memory migrations。

    中文注释：
    PostgresMemoryStore 只调用这个函数。
    具体建表、加列、建索引都集中在这里，避免业务 store 类膨胀。
    """

    applied_now: list[dict[str, Any]] = []
    with _connect(database_url) as conn:
        applied = _applied_versions(conn)
        for migration in MIGRATIONS:
            checksum = _checksum(migration)
            if migration.version in applied:
                if applied[migration.version] != checksum:
                    raise RuntimeError(
                        "memory migration checksum mismatch: "
                        f"version={migration.version} name={migration.name}"
                    )
                continue
            with conn.transaction():
                for sql in migration.up_sql:
                    if sql.strip():
                        conn.execute(sql)
                conn.execute(
                    """
                    INSERT INTO beginner_agent_memory_schema_migrations (
                        version, name, applied_at, checksum
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        migration.version,
                        migration.name,
                        datetime.now(timezone.utc),
                        checksum,
                    ),
                )
            applied_now.append(
                {
                    "version": migration.version,
                    "name": migration.name,
                    "checksum": checksum,
                }
            )
    return {
        "current_version": current_memory_schema_version(database_url),
        "applied": applied_now,
        "pending": pending_memory_migrations(database_url),
    }


def rollback_memory_migration(database_url: str, *, target_version: int) -> dict[str, Any]:
    """回滚到指定 schema version。

    中文注释：
    生产环境回滚需要极其谨慎。
    这里提供的是“有版本、有 down SQL、有记录”的基础能力。
    真正多人生产环境还应该接审批和备份。
    """

    if target_version < 0:
        raise ValueError("target_version 不能小于 0。")

    rolled_back: list[dict[str, Any]] = []
    with _connect(database_url) as conn:
        applied = _applied_versions(conn)
        migrations_by_version = {migration.version: migration for migration in MIGRATIONS}
        for version in sorted(applied.keys(), reverse=True):
            if version <= target_version:
                continue
            migration = migrations_by_version.get(version)
            if migration is None:
                raise RuntimeError(f"找不到 version={version} 的 migration 定义。")
            with conn.transaction():
                for sql in migration.down_sql:
                    if sql.strip():
                        conn.execute(sql)
                conn.execute(
                    """
                    DELETE FROM beginner_agent_memory_schema_migrations
                    WHERE version = %s
                    """,
                    (version,),
                )
            rolled_back.append({"version": version, "name": migration.name})
    return {
        "current_version": current_memory_schema_version(database_url),
        "rolled_back": rolled_back,
    }


def backfill_memory_governance_fields(database_url: str, *, limit: int = 1000) -> dict[str, Any]:
    """数据回填 job：修正旧数据缺失的治理字段。

    中文注释：
    schema migration 负责“表结构变了”。
    backfill job 负责“历史数据补齐默认值/修复异常”。
    两者分开，才更接近生产级数据库治理。
    """

    if limit <= 0:
        raise ValueError("limit 必须大于 0。")

    run_memory_migrations(database_url)
    with _connect(database_url) as conn:
        with conn.transaction():
            result = conn.execute(
                """
                UPDATE beginner_agent_memory
                SET
                    paths = COALESCE(paths, '[]'::jsonb),
                    tags = COALESCE(tags, '[]'::jsonb),
                    metadata = COALESCE(metadata, '{}'::jsonb),
                    confidence = COALESCE(confidence, 0.7),
                    importance = COALESCE(importance, 0.5),
                    quality_score = COALESCE(quality_score, 0.5),
                    trust_score = COALESCE(trust_score, 0.5),
                    decay_score = COALESCE(decay_score, 0.0),
                    scope = COALESCE(NULLIF(scope, ''), 'project'),
                    visibility = COALESCE(NULLIF(visibility, ''), 'project'),
                    sensitivity_level = COALESCE(NULLIF(sensitivity_level, ''), 'internal'),
                    tenant_id = COALESCE(NULLIF(tenant_id, ''), 'local-tenant'),
                    workspace_id = COALESCE(NULLIF(workspace_id, ''), 'local-workspace'),
                    project_id = COALESCE(NULLIF(project_id, ''), 'beginner_agent'),
                    user_id = COALESCE(NULLIF(user_id, ''), 'local-user'),
                    retention_policy = COALESCE(NULLIF(retention_policy, ''), 'ttl'),
                    validity_status = COALESCE(NULLIF(validity_status, ''), 'active'),
                    pinned = COALESCE(pinned, FALSE)
                WHERE id IN (
                    SELECT id
                    FROM beginner_agent_memory
                    ORDER BY created_at DESC
                    LIMIT %s
                )
                """,
                (limit,),
            )
    return {
        "updated": int(result.rowcount or 0),
        "limit": limit,
        "current_version": current_memory_schema_version(database_url),
    }
