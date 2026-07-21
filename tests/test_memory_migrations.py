from __future__ import annotations

import pytest

from beginner_agent.memory_migrations import (
    MIGRATIONS,
    current_memory_schema_version,
    pending_memory_migrations,
    run_memory_migrations,
)


def test_memory_migrations_are_ordered_and_include_lifecycle_history() -> None:
    versions = [migration.version for migration in MIGRATIONS]
    names = {migration.name for migration in MIGRATIONS}

    assert versions == sorted(versions)
    assert len(versions) == len(set(versions))
    assert "memory_lifecycle_run_history" in names
    assert any(
        "beginner_agent_memory_lifecycle_runs" in sql
        for migration in MIGRATIONS
        for sql in migration.up_sql
    )


@pytest.mark.postgres
def test_postgres_memory_migrations_apply_idempotently(postgres_database_url: str) -> None:
    first = run_memory_migrations(postgres_database_url)
    second = run_memory_migrations(postgres_database_url)

    assert first["current_version"] >= 5
    assert second["pending"] == []
    assert current_memory_schema_version(postgres_database_url) >= 5
    assert pending_memory_migrations(postgres_database_url) == []
