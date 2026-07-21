from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryApiResponse(BaseModel):
    """API 通用响应模型。

    中文注释：
    对外 API 不直接返回裸 list / dict。
    统一包一层 response，方便后续加入 request_id、分页、权限信息。
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    backend: str
    count: int = 0
    data: Any
    error: str = ""


class MemoryQuery(BaseModel):
    """Memory 查询条件。"""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, ge=1, le=500)
    kind: str | None = None
    task_id: str | None = None
    tool_name: str | None = None
    contradiction_key: str | None = None
    file_path: str | None = None
    pinned: bool | None = None
    failure_category: str | None = None
    failure_pattern_id: str | None = None
    include_sensitive: bool = False


class AuditQuery(BaseModel):
    """Memory audit 查询条件。"""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=100, ge=1, le=1000)
    memory_id: str | None = None
    run_id: str | None = None
    action: str | None = None
    include_sensitive: bool = False
