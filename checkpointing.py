from __future__ import annotations

import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver


DEFAULT_DATABASE_URL = "postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent"


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


def _postgres_database_url() -> str:
    """返回 checkpoint 使用的 Postgres 连接串。"""

    return (
        os.getenv("BEGINNER_AGENT_CHECKPOINT_DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
        or DEFAULT_DATABASE_URL
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

    backend = checkpoint_backend_name()
    if backend == "memory":
        return MemorySaver()
    if backend == "postgres":
        return _postgres_checkpointer()
    raise ValueError(
        "BEGINNER_AGENT_CHECKPOINT_BACKEND 只能是 memory 或 postgres，"
        f"当前是：{backend}"
    )
