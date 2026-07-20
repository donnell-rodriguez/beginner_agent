from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ToolResultStatus = Literal["success", "failed", "blocked", "empty", "partial"]


class ToolValidation(BaseModel):
    """工具参数校验结果。

    中文注释：
    生产级工具调用不能只知道“执行结果”。
    还要知道执行前参数有没有通过 schema / validator。
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    reason: str


class ToolResult(BaseModel):
    """结构化工具执行结果。

    中文注释：
    这是 Executor / Evaluator / Audit / Memory 之间共享的工具结果合同。

    为什么需要它？
    - 字符串只适合人看，不适合机器判断。
    - Evaluator 需要 status / retryable / diagnostics。
    - Audit 需要 duration_ms / metadata。
    - Memory 需要 tool_name / changed_files / artifact_paths。
    - UI 或 API 可以直接消费 JSON Schema。
    """

    model_config = ConfigDict(extra="forbid")

    status: ToolResultStatus
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    normalized_args: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    validation: ToolValidation
    metadata: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    changed_files: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: int = 0
    error_type: str = ""
    retryable: bool = False

    @field_validator("duration_ms")
    @classmethod
    def _duration_must_be_non_negative(cls, value: int) -> int:
        """duration_ms 不能是负数。"""

        if value < 0:
            raise ValueError("duration_ms 不能小于 0。")
        return value


def tool_result_json_schema() -> dict[str, Any]:
    """导出 ToolResult 的 JSON Schema。"""

    return ToolResult.model_json_schema()


def classify_tool_output(output: str) -> ToolResultStatus:
    """把旧工具字符串输出归一成 ToolResultStatus。

    中文注释：
    工具函数当前大多还是返回字符串。
    在工具全部升级成结构化返回之前，这里负责做兼容归类。
    """

    if not output.strip():
        return "empty"
    failed_prefixes = (
        "路径不存在",
        "不允许",
        "未知工具",
        "不是文件",
        "不是目录",
        "search_code 需要",
        "inspect_symbol 需要",
        "git ",
        "静态检查失败",
        "lint/typecheck 发现问题",
        "测试失败",
        "apply_patch 失败",
        "rollback 需要",
        "PatchPlan 不存在",
        "preview_patch 失败",
        "apply_patch_dry_run 失败",
    )
    if output.startswith(("工具安全检查失败", "工具参数不安全", "工具 ")) or output.startswith(
        failed_prefixes
    ):
        return "failed"
    if "内容过长，已截断" in output:
        return "partial"
    return "success"


def retryable_from_status(status: ToolResultStatus) -> bool:
    """根据状态给出默认重试建议。"""

    return status in {"failed", "empty", "partial"}
