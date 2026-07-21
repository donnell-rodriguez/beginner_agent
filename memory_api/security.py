from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from beginner_agent.memory_settings import (
    DEFAULT_PROJECT_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_USER_ID,
    DEFAULT_WORKSPACE_ID,
)


ApiRole = Literal[
    "memory_reader",
    "audit_reader",
    "sensitive_reader",
    "admin",
]


@dataclass(frozen=True)
class ApiPrincipal:
    """调用 Memory API 的身份。

    中文注释：
    这不是 LangGraph 的 State，而是 HTTP API 这一层的“人是谁”。
    大厂系统里，任何 Admin API 都要知道：
    - actor_id：谁在调用。
    - roles：他有哪些权限。
    - tenant/workspace/project/user：他能看哪个范围的数据。
    """

    actor_id: str
    roles: frozenset[str]
    tenant_id: str
    workspace_id: str
    project_id: str
    user_id: str


@dataclass(frozen=True)
class ApiRequestContext:
    """单次 API 请求的安全上下文。"""

    request_id: str
    principal: ApiPrincipal
    sensitive_approval_id: str = ""


class InMemoryRateLimiter:
    """非常轻量的内存限流器。

    中文注释：
    生产环境一般会用 Redis / API Gateway / Envoy 做分布式限流。
    当前项目先做进程内限流，至少具备“同一个 token/IP 不能无限打 API”的治理入口。
    """

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}

    def check(self, key: str) -> None:
        enabled = _env_bool("BEGINNER_AGENT_MEMORY_API_RATE_LIMIT_ENABLED", True)
        if not enabled:
            return

        limit = int(os.getenv("BEGINNER_AGENT_MEMORY_API_RATE_LIMIT_PER_MINUTE", "120"))
        now = time.time()
        window_start = now - 60
        hits = [hit for hit in self._hits.get(key, []) if hit >= window_start]
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="Memory API rate limit exceeded.")
        hits.append(now)
        self._hits[key] = hits


class RequestGovernanceMiddleware(BaseHTTPMiddleware):
    """给每个请求补 request_id，并做基础限流。

    中文注释：
    request_id 是排查线上问题的入口。
    有了它，API response、日志、audit event 可以串起来。
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._rate_limiter = InMemoryRateLimiter()

    async def dispatch(self, request: Request, call_next: Any):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        rate_key = _rate_limit_key(request)
        try:
            self._rate_limiter.check(rate_key)
            response = await call_next(request)
        except HTTPException as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={
                    "ok": False,
                    "request_id": request_id,
                    "backend": "memory-api",
                    "count": 0,
                    "page": None,
                    "data": None,
                    "error": str(exc.detail),
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rate_limit_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    api_key = request.headers.get("x-api-key", "")
    if auth or api_key:
        return f"token:{auth or api_key}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def _token_from_request(request: Request) -> str:
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _default_principal() -> ApiPrincipal:
    return ApiPrincipal(
        actor_id=os.getenv("BEGINNER_AGENT_MEMORY_API_ACTOR_ID", "local-admin"),
        roles=frozenset({"admin", "memory_reader", "audit_reader", "sensitive_reader"}),
        tenant_id=os.getenv("BEGINNER_AGENT_TENANT_ID", DEFAULT_TENANT_ID),
        workspace_id=os.getenv("BEGINNER_AGENT_WORKSPACE_ID", DEFAULT_WORKSPACE_ID),
        project_id=os.getenv("BEGINNER_AGENT_PROJECT_ID", DEFAULT_PROJECT_ID),
        user_id=os.getenv("BEGINNER_AGENT_USER_ID", DEFAULT_USER_ID),
    )


def _token_principals() -> dict[str, ApiPrincipal]:
    """从 env 读取 token -> principal 映射。

    中文注释：
    本地可以用简单 env token。
    多人系统后续应该接 OAuth/OIDC，并把 token introspection 放到网关层。
    """

    principals: dict[str, ApiPrincipal] = {}
    raw_json = os.getenv("BEGINNER_AGENT_MEMORY_API_TOKENS", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("BEGINNER_AGENT_MEMORY_API_TOKENS 不是合法 JSON。") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("BEGINNER_AGENT_MEMORY_API_TOKENS 必须是 token -> principal dict。")
        for token, raw_principal in parsed.items():
            if not isinstance(raw_principal, dict):
                continue
            principals[str(token)] = ApiPrincipal(
                actor_id=str(raw_principal.get("actor_id", "api-user")),
                roles=frozenset(str(role) for role in raw_principal.get("roles", [])),
                tenant_id=str(raw_principal.get("tenant_id", DEFAULT_TENANT_ID)),
                workspace_id=str(raw_principal.get("workspace_id", DEFAULT_WORKSPACE_ID)),
                project_id=str(raw_principal.get("project_id", DEFAULT_PROJECT_ID)),
                user_id=str(raw_principal.get("user_id", DEFAULT_USER_ID)),
            )

    admin_token = os.getenv("BEGINNER_AGENT_MEMORY_API_ADMIN_TOKEN", "").strip()
    if admin_token:
        principals[admin_token] = _default_principal()

    reader_token = os.getenv("BEGINNER_AGENT_MEMORY_API_READER_TOKEN", "").strip()
    if reader_token:
        principals[reader_token] = ApiPrincipal(
            actor_id="memory-reader",
            roles=frozenset({"memory_reader"}),
            tenant_id=os.getenv("BEGINNER_AGENT_TENANT_ID", DEFAULT_TENANT_ID),
            workspace_id=os.getenv("BEGINNER_AGENT_WORKSPACE_ID", DEFAULT_WORKSPACE_ID),
            project_id=os.getenv("BEGINNER_AGENT_PROJECT_ID", DEFAULT_PROJECT_ID),
            user_id=os.getenv("BEGINNER_AGENT_USER_ID", DEFAULT_USER_ID),
        )

    auditor_token = os.getenv("BEGINNER_AGENT_MEMORY_API_AUDITOR_TOKEN", "").strip()
    if auditor_token:
        principals[auditor_token] = ApiPrincipal(
            actor_id="memory-auditor",
            roles=frozenset({"memory_reader", "audit_reader"}),
            tenant_id=os.getenv("BEGINNER_AGENT_TENANT_ID", DEFAULT_TENANT_ID),
            workspace_id=os.getenv("BEGINNER_AGENT_WORKSPACE_ID", DEFAULT_WORKSPACE_ID),
            project_id=os.getenv("BEGINNER_AGENT_PROJECT_ID", DEFAULT_PROJECT_ID),
            user_id=os.getenv("BEGINNER_AGENT_USER_ID", DEFAULT_USER_ID),
        )
    return principals


def api_context(request: Request) -> ApiRequestContext:
    """解析并校验当前请求身份。"""

    request_id = str(getattr(request.state, "request_id", "") or uuid.uuid4().hex)
    require_auth = _env_bool("BEGINNER_AGENT_MEMORY_API_REQUIRE_AUTH", False)
    token = _token_from_request(request)
    principals = _token_principals()

    principal = principals.get(token)
    if principal is None:
        if require_auth:
            raise HTTPException(status_code=401, detail="Memory API authentication required.")
        principal = _default_principal()

    return ApiRequestContext(
        request_id=request_id,
        principal=principal,
        sensitive_approval_id=request.headers.get("x-memory-sensitive-approval", "").strip(),
    )


def require_role(context: ApiRequestContext, *roles: str) -> None:
    if "admin" in context.principal.roles:
        return
    if any(role in context.principal.roles for role in roles):
        return
    raise HTTPException(status_code=403, detail=f"Missing required role: {', '.join(roles)}")


def require_sensitive_access(context: ApiRequestContext, include_sensitive: bool) -> bool:
    """判断 include_sensitive=true 是否允许。

    中文注释：
    敏感内容不能只靠 query param 打开。
    允许方式：
    - principal 有 admin/sensitive_reader。
    - 或请求带一次性/人工审批码 X-Memory-Sensitive-Approval。
    """

    if not include_sensitive:
        return False
    if {"admin", "sensitive_reader"}.intersection(context.principal.roles):
        return True

    expected = os.getenv("BEGINNER_AGENT_MEMORY_API_SENSITIVE_APPROVAL_TOKEN", "").strip()
    if expected and context.sensitive_approval_id == expected:
        return True
    raise HTTPException(status_code=403, detail="Sensitive memory access requires approval.")


def context_metadata(context: ApiRequestContext) -> dict[str, Any]:
    """把 API 安全上下文转成可写入 audit metadata 的 dict。"""

    return {
        "request_id": context.request_id,
        "actor_id": context.principal.actor_id,
        "roles": sorted(context.principal.roles),
        "tenant_id": context.principal.tenant_id,
        "workspace_id": context.principal.workspace_id,
        "project_id": context.principal.project_id,
        "user_id": context.principal.user_id,
    }
