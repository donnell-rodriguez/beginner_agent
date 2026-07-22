from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .models import AuditQuery, MemoryApiResponse, MemoryQuery, PageInfo
from .repository import MemoryQueryRepository, RepositoryResult
from .security import (
    ApiRequestContext,
    RequestGovernanceMiddleware,
    api_context,
    require_role,
    require_sensitive_access,
)


def _response(
    data: Any,
    backend: str,
    *,
    request_id: str,
    error: str = "",
    page: PageInfo | None = None,
) -> MemoryApiResponse:
    count = len(data) if isinstance(data, list) else (1 if data else 0)
    return MemoryApiResponse(
        ok=not bool(error),
        request_id=request_id,
        backend=backend,
        count=count,
        page=page,
        data=data,
        error=error,
    )


def _repository_response(
    result: RepositoryResult,
    context: ApiRequestContext,
) -> MemoryApiResponse:
    return _response(
        result.data,
        result.backend,
        request_id=context.request_id,
        error=result.error,
        page=result.page,
    )


def create_app() -> FastAPI:
    """创建 Memory Query API。

    中文注释：
    这是只读 Admin API，但已经不再是“裸查询接口”：
    - auth：支持 Bearer token / X-API-Key。
    - RBAC：区分 memory_reader、audit_reader、sensitive_reader、admin。
    - request_id：每个请求都有追踪 ID。
    - rate limit：基础进程内限流。
    - tenant isolation：仓库层按 tenant/workspace/project/user 过滤。
    - sensitive approval：敏感内容必须有角色或审批 token。
    """

    app = FastAPI(
        title="Beginner Agent Memory Query API",
        version="0.1.0",
        description="Read-only API for memory, audit, failure patterns, and preferences.",
    )
    app.add_middleware(RequestGovernanceMiddleware)
    repository = MemoryQueryRepository()

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", ""))
        return JSONResponse(
            status_code=exc.status_code,
            content=MemoryApiResponse(
                ok=False,
                request_id=request_id,
                backend="memory-api",
                count=0,
                page=None,
                data=None,
                error=str(exc.detail),
            ).model_dump(mode="json"),
        )

    @app.get("/health", response_model=MemoryApiResponse)
    def health(request: Request) -> MemoryApiResponse:
        request_id = str(getattr(request.state, "request_id", ""))
        return _response({"status": "ok"}, "memory-api", request_id=request_id)

    @app.get("/memories", response_model=MemoryApiResponse)
    def list_memories(
        limit: int = Query(default=50, ge=1, le=500),
        cursor: str | None = None,
        kind: str | None = None,
        task_id: str | None = None,
        tool_name: str | None = None,
        contradiction_key: str | None = None,
        file_path: str | None = None,
        pinned: bool | None = None,
        failure_category: str | None = None,
        failure_pattern_id: str | None = None,
        include_sensitive: bool = False,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "memory_reader")
        include_sensitive = require_sensitive_access(context, include_sensitive)
        query = MemoryQuery(
            limit=limit,
            cursor=cursor,
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
        return _repository_response(repository.list_memories(query, context), context)

    @app.get("/memories/{memory_id}", response_model=MemoryApiResponse)
    def get_memory(
        memory_id: str,
        include_sensitive: bool = False,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "memory_reader")
        include_sensitive = require_sensitive_access(context, include_sensitive)
        result = repository.get_memory(
            memory_id,
            include_sensitive=include_sensitive,
            context=context,
        )
        if result.data is None:
            raise HTTPException(status_code=404, detail=f"memory not found: {memory_id}")
        return _repository_response(result, context)

    @app.get("/memories/{memory_id}/why", response_model=MemoryApiResponse)
    def why_saved(
        memory_id: str,
        include_sensitive: bool = False,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        include_sensitive = require_sensitive_access(context, include_sensitive)
        result = repository.why_saved(
            memory_id,
            include_sensitive=include_sensitive,
            context=context,
        )
        if result.data is None:
            raise HTTPException(status_code=404, detail=f"memory not found: {memory_id}")
        return _repository_response(result, context)

    @app.get("/memories/{memory_id}/usage", response_model=MemoryApiResponse)
    def usage(
        memory_id: str,
        include_sensitive: bool = False,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        include_sensitive = require_sensitive_access(context, include_sensitive)
        result = repository.usage(
            memory_id,
            include_sensitive=include_sensitive,
            context=context,
        )
        return _repository_response(result, context)

    @app.get("/memories/{memory_id}/feedback", response_model=MemoryApiResponse)
    def memory_feedback(
        memory_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(repository.feedback(memory_id, limit=limit), context)

    @app.get("/feedback", response_model=MemoryApiResponse)
    def all_feedback(
        limit: int = Query(default=100, ge=1, le=1000),
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(repository.feedback(None, limit=limit), context)

    @app.get("/eval-cases", response_model=MemoryApiResponse)
    def eval_cases(
        limit: int = Query(default=100, ge=1, le=1000),
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(repository.eval_cases(limit=limit), context)

    @app.get("/rerank/telemetry", response_model=MemoryApiResponse)
    def rerank_telemetry(
        limit: int = Query(default=100, ge=1, le=1000),
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(repository.rerank_telemetry(limit=limit), context)

    @app.get("/usage/effectiveness", response_model=MemoryApiResponse)
    def usage_effectiveness(
        memory_id: str = "",
        limit: int = Query(default=100, ge=1, le=1000),
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(
            repository.usage_effectiveness(memory_id=memory_id, limit=limit),
            context,
        )

    @app.get("/eval/online", response_model=MemoryApiResponse)
    def online_eval(
        limit: int = Query(default=100, ge=1, le=1000),
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(repository.online_eval(limit=limit), context)

    @app.get("/observability/events", response_model=MemoryApiResponse)
    def memory_observability(
        limit: int = Query(default=100, ge=1, le=1000),
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(repository.memory_observability(limit=limit), context)

    @app.get("/postgres/governance", response_model=MemoryApiResponse)
    def postgres_governance(
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "admin")
        return _repository_response(repository.postgres_governance(), context)

    @app.get("/audit", response_model=MemoryApiResponse)
    def audit(
        limit: int = Query(default=100, ge=1, le=1000),
        cursor: str | None = None,
        memory_id: str | None = None,
        run_id: str | None = None,
        action: str | None = None,
        include_sensitive: bool = False,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        include_sensitive = require_sensitive_access(context, include_sensitive)
        result = repository.list_audit_events(
            AuditQuery(
                limit=limit,
                cursor=cursor,
                memory_id=memory_id,
                run_id=run_id,
                action=action,
                include_sensitive=include_sensitive,
            ),
            context,
        )
        return _repository_response(result, context)

    @app.get("/runs/{run_id}/lineage", response_model=MemoryApiResponse)
    def run_lineage(
        run_id: str,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "audit_reader")
        return _repository_response(repository.run_lineage(run_id), context)

    @app.get("/contradictions/{contradiction_key}", response_model=MemoryApiResponse)
    def contradiction_evolution(
        contradiction_key: str,
        include_sensitive: bool = False,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "memory_reader")
        include_sensitive = require_sensitive_access(context, include_sensitive)
        result = repository.contradiction_evolution(
            contradiction_key,
            include_sensitive=include_sensitive,
            context=context,
        )
        return _repository_response(result, context)

    @app.get("/pinned", response_model=MemoryApiResponse)
    def pinned_memories(
        limit: int = Query(default=100, ge=1, le=500),
        cursor: str | None = None,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "memory_reader")
        return _repository_response(
            repository.list_memories(
                MemoryQuery(limit=limit, cursor=cursor, pinned=True),
                context,
            ),
            context,
        )

    @app.get("/failures/patterns", response_model=MemoryApiResponse)
    def failure_patterns(
        limit: int = Query(default=100, ge=1, le=500),
        category: str | None = None,
        pattern_id: str | None = None,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "memory_reader")
        return _repository_response(
            repository.failure_patterns(
                limit=limit,
                category=category,
                pattern_id=pattern_id,
                context=context,
            ),
            context,
        )

    @app.get("/files/{file_path:path}/memories", response_model=MemoryApiResponse)
    def file_memories(
        file_path: str,
        limit: int = Query(default=100, ge=1, le=500),
        cursor: str | None = None,
        context: ApiRequestContext = Depends(api_context),
    ) -> MemoryApiResponse:
        require_role(context, "memory_reader")
        return _repository_response(
            repository.list_memories(
                MemoryQuery(limit=limit, cursor=cursor, file_path=file_path),
                context,
            ),
            context,
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BEGINNER_AGENT_MEMORY_API_HOST", "127.0.0.1")
    port = int(os.getenv("BEGINNER_AGENT_MEMORY_API_PORT", "8770"))
    uvicorn.run("beginner_agent.memory_api.app:app", host=host, port=port, reload=False)
