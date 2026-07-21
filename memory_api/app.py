from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from .models import AuditQuery, MemoryApiResponse, MemoryQuery
from .repository import MemoryQueryRepository


def _response(data: Any, backend: str, error: str = "") -> MemoryApiResponse:
    count = len(data) if isinstance(data, list) else (1 if data else 0)
    return MemoryApiResponse(
        ok=not bool(error),
        backend=backend,
        count=count,
        data=data,
        error=error,
    )


def create_app() -> FastAPI:
    """创建 Memory Query API。

    中文注释：
    这是只读 Admin API。
    当前用于本地查询 memory 和 audit，后续可以接：
    - Web dashboard。
    - 登录鉴权。
    - 分页和索引。
    - 更细粒度的 RBAC / audit viewer。
    """

    app = FastAPI(
        title="Beginner Agent Memory Query API",
        version="0.1.0",
        description="Read-only API for memory, audit, failure patterns, and preferences.",
    )
    repository = MemoryQueryRepository()

    @app.get("/health", response_model=MemoryApiResponse)
    def health() -> MemoryApiResponse:
        return _response({"status": "ok"}, "memory-api")

    @app.get("/memories", response_model=MemoryApiResponse)
    def list_memories(
        limit: int = Query(default=50, ge=1, le=500),
        kind: str | None = None,
        task_id: str | None = None,
        tool_name: str | None = None,
        contradiction_key: str | None = None,
        file_path: str | None = None,
        pinned: bool | None = None,
        failure_category: str | None = None,
        failure_pattern_id: str | None = None,
        include_sensitive: bool = False,
    ) -> MemoryApiResponse:
        query = MemoryQuery(
            limit=limit,
            kind=kind,
            task_id=task_id,
            tool_name=tool_name,
            contradiction_key=contradiction_key,
            file_path=file_path,
            pinned=pinned,
            failure_category=failure_category,
            failure_pattern_id=failure_pattern_id,
            include_sensitive=include_sensitive,
        )
        records, backend, error = repository.list_memories(query)
        return _response(records, backend, error)

    @app.get("/memories/{memory_id}", response_model=MemoryApiResponse)
    def get_memory(memory_id: str, include_sensitive: bool = False) -> MemoryApiResponse:
        record, backend, error = repository.get_memory(
            memory_id,
            include_sensitive=include_sensitive,
        )
        if record is None:
            raise HTTPException(status_code=404, detail=f"memory not found: {memory_id}")
        return _response(record, backend, error)

    @app.get("/memories/{memory_id}/why", response_model=MemoryApiResponse)
    def why_saved(memory_id: str, include_sensitive: bool = False) -> MemoryApiResponse:
        data, backend, error = repository.why_saved(
            memory_id,
            include_sensitive=include_sensitive,
        )
        if data is None:
            raise HTTPException(status_code=404, detail=f"memory not found: {memory_id}")
        return _response(data, backend, error)

    @app.get("/memories/{memory_id}/usage", response_model=MemoryApiResponse)
    def usage(memory_id: str, include_sensitive: bool = False) -> MemoryApiResponse:
        data, backend, error = repository.usage(
            memory_id,
            include_sensitive=include_sensitive,
        )
        return _response(data, backend, error)

    @app.get("/memories/{memory_id}/feedback", response_model=MemoryApiResponse)
    def memory_feedback(
        memory_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> MemoryApiResponse:
        data, backend, error = repository.feedback(memory_id, limit=limit)
        return _response(data, backend, error)

    @app.get("/feedback", response_model=MemoryApiResponse)
    def all_feedback(
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> MemoryApiResponse:
        data, backend, error = repository.feedback(None, limit=limit)
        return _response(data, backend, error)

    @app.get("/eval-cases", response_model=MemoryApiResponse)
    def eval_cases(limit: int = Query(default=100, ge=1, le=1000)) -> MemoryApiResponse:
        data, backend, error = repository.eval_cases(limit=limit)
        return _response(data, backend, error)

    @app.get("/rerank/telemetry", response_model=MemoryApiResponse)
    def rerank_telemetry(
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> MemoryApiResponse:
        data, backend, error = repository.rerank_telemetry(limit=limit)
        return _response(data, backend, error)

    @app.get("/audit", response_model=MemoryApiResponse)
    def audit(
        limit: int = Query(default=100, ge=1, le=1000),
        memory_id: str | None = None,
        run_id: str | None = None,
        action: str | None = None,
        include_sensitive: bool = False,
    ) -> MemoryApiResponse:
        events, backend, error = repository.list_audit_events(
            AuditQuery(
                limit=limit,
                memory_id=memory_id,
                run_id=run_id,
                action=action,
                include_sensitive=include_sensitive,
            )
        )
        return _response(events, backend, error)

    @app.get("/runs/{run_id}/lineage", response_model=MemoryApiResponse)
    def run_lineage(run_id: str) -> MemoryApiResponse:
        data, backend, error = repository.run_lineage(run_id)
        return _response(data, backend, error)

    @app.get("/contradictions/{contradiction_key}", response_model=MemoryApiResponse)
    def contradiction_evolution(
        contradiction_key: str,
        include_sensitive: bool = False,
    ) -> MemoryApiResponse:
        records, backend, error = repository.contradiction_evolution(
            contradiction_key,
            include_sensitive=include_sensitive,
        )
        return _response(records, backend, error)

    @app.get("/pinned", response_model=MemoryApiResponse)
    def pinned_memories(limit: int = Query(default=100, ge=1, le=500)) -> MemoryApiResponse:
        records, backend, error = repository.list_memories(
            MemoryQuery(limit=limit, pinned=True)
        )
        return _response(records, backend, error)

    @app.get("/failures/patterns", response_model=MemoryApiResponse)
    def failure_patterns(
        limit: int = Query(default=100, ge=1, le=500),
        category: str | None = None,
        pattern_id: str | None = None,
    ) -> MemoryApiResponse:
        patterns, backend, error = repository.failure_patterns(
            limit=limit,
            category=category,
            pattern_id=pattern_id,
        )
        return _response(patterns, backend, error)

    @app.get("/files/{file_path:path}/memories", response_model=MemoryApiResponse)
    def file_memories(
        file_path: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> MemoryApiResponse:
        records, backend, error = repository.list_memories(
            MemoryQuery(limit=limit, file_path=file_path)
        )
        return _response(records, backend, error)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BEGINNER_AGENT_MEMORY_API_HOST", "127.0.0.1")
    port = int(os.getenv("BEGINNER_AGENT_MEMORY_API_PORT", "8770"))
    uvicorn.run("beginner_agent.memory_api.app:app", host=host, port=port, reload=False)
