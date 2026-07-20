from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .audit_tools import audit_patch_tool, audit_tool_call_tool, read_audit_log_tool
from .check_tools import (
    format_check_tool,
    get_diagnostics_tool,
    lint_typecheck_tool,
    parse_test_failure_tool,
    run_build_tool,
    run_targeted_tests_tool,
    run_tests_tool,
    run_typecheck_tool,
    static_check_tool,
)
from .command_tools import (
    ALLOWED_COMMANDS,
    list_allowed_commands_tool,
    run_allowed_command_tool,
    run_cargo_check_tool,
    run_cargo_clippy_tool,
    run_cargo_fmt_check_tool,
    run_cargo_test_tool,
    run_mypy_tool,
    run_package_script_tool,
    run_pytest_tool,
    run_ruff_format_check_tool,
    run_ruff_tool,
    run_uv_import_graph_tool,
    safe_path_exists_tool,
)
from .core import WORKSPACE_ROOT, json_dumps, list_project_roots, safe_resolve, safe_text_file
from .environment_tools import (
    detect_build_system_tool,
    detect_package_manager_tool,
    detect_project_stack_tool,
    detect_test_framework_tool,
)
from .failure_tools import (
    classify_failure_tool,
    compare_failure_before_after_tool,
    extract_stack_trace_tool,
)
from .git_tools import git_diff_file_tool, git_diff_tool, git_status_tool
from .index_tools import (
    build_project_index_tool,
    inspect_class_hierarchy_tool,
    inspect_function_signature_tool,
    query_project_index_tool,
)
from .inspect_tools import (
    inspect_call_graph_tool,
    inspect_import_graph_tool,
    inspect_references_tool,
    inspect_symbol_tool,
)
from .memory_tools import checkpoint_load_tool, checkpoint_save_tool
from .patch_tools import (
    apply_patch_dry_run_tool,
    apply_patch_plan_tool,
    apply_patch_tool,
    format_apply_tool,
    patch_plan_tool,
    preview_patch_tool,
    revert_file_patch_tool,
    rollback_tool,
    validate_patch_plan_tool,
    validate_patch_scope_tool,
)
from .platform_tools import (
    get_active_project_tool,
    list_project_roots_tool,
    register_project_root_tool,
    set_active_project_tool,
)
from .project_tools import dependency_inspect_tool, summarize_file_tool
from .read_tools import list_files_tool, list_tree_tool, read_file_slice_tool, read_file_tool
from .results import (
    ToolResult,
    ToolValidation,
    classify_tool_output,
    retryable_from_status,
    tool_result_json_schema,
)
from .rust_tools import (
    detect_rust_project_tool,
    inspect_rust_references_tool,
    inspect_rust_symbols_tool,
    map_changed_rust_files_to_tests_tool,
    parse_cargo_test_failure_tool,
    parse_rust_errors_tool,
)
from .search_tools import grep_regex_tool, search_code_tool
from .security_tools import secret_scan_tool
from .test_selection_tools import (
    map_changed_files_to_tests_tool,
    run_impacted_tests_tool,
    select_relevant_tests_tool,
)


# 中文注释：
# registry.py 现在是 ToolSpec 驱动的工具注册中心。
#
# 旧版本里有三份信息要手动同步：
# - READ_ONLY_TOOLS / WRITE_TOOLS
# - validate_tool_request(...)
# - dispatch 字典
#
# 现在每个工具先登记成 ToolSpec：
#
#     ToolSpec(name="read_file", access="read", handler=...)
#
# 然后 READ_ONLY_TOOLS / WRITE_TOOLS / ALL_TOOLS 都从 ToolSpec 自动生成。
# run_tool(...) 也从 ToolSpec.handler 执行工具。
#
# 这更接近大厂工具平台的做法：
# 工具不是一堆散落的函数，而是一组带元数据、权限和执行入口的规范对象。

Access = Literal["read", "write"]
Risk = Literal["low", "medium", "high"]
Language = Literal["python", "rust", "platform", "generic"]
Category = Literal["observe", "understand", "verify", "edit", "audit", "platform", "memory", "security"]
ToolRunStatus = Literal["success", "failed", "blocked"]
Validator = Callable[[dict[str, Any]], tuple[bool, str]]
ToolArgsSchema = type[BaseModel]


class ToolArgsBase(BaseModel):
    """所有工具参数模型的基础类。

    中文注释：
    现在工具参数 schema 升级成 Pydantic。

    Pydantic 带来三件重要能力：
    - 运行时强校验：model_validate(...) 会检查参数类型、必填字段。
    - 自动默认值：没有传的可选参数会自动补默认值。
    - JSON Schema 导出：model_json_schema() 可以给 LLM / UI / MCP 使用。

    extra="forbid" 表示不允许 LLM 传入未声明参数。
    这比“随便一个 dict 都接受”更接近生产级工具平台。
    """

    model_config = ConfigDict(extra="forbid")


def _allow_any_args(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """默认 validator：工具没有必填参数，或者 handler 内部已经有安全边界。"""

    return True, "参数安全。"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """单个工具的注册规格。

    中文注释：
    ToolSpec 是工具平台的核心数据结构。
    它告诉系统：
    - 工具叫什么。
    - 是读工具还是写工具。
    - 属于哪个语言/类别。
    - 风险等级是多少。
    - 参数结构是什么：Pydantic args_schema。
    - 参数如何校验。
    - 真正执行时调用哪个 handler。
    """

    name: str
    access: Access
    category: Category
    language: Language
    risk: Risk
    handler: Callable[[dict[str, Any]], str]
    validator: Validator
    args_schema: ToolArgsSchema
    description: str = ""

    @property
    def requires_approval(self) -> bool:
        """写工具默认需要审批。"""

        return self.access == "write"

    def to_dict(self) -> dict[str, Any]:
        """输出工具元数据。

        中文注释：
        这份元数据是工具平台的“目录信息”。
        未来接 MCP / UI / 审计系统时，都应该优先读这里。
        """

        return {
            "name": self.name,
            "access": self.access,
            "category": self.category,
            "language": self.language,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
            "description": self.description,
            "args_model": self.args_schema.__name__,
            "args_schema": self.args_schema.model_json_schema(),
        }


def _string_arg(args: dict[str, Any], name: str, default: str = "") -> str:
    """从 tool_args 里取字符串参数。"""

    return str(args.get(name, default))


def _int_arg(args: dict[str, Any], name: str, default: int) -> int:
    """从 tool_args 里取整数参数，失败时使用默认值。"""

    try:
        return int(args.get(name, default))
    except (TypeError, ValueError):
        return default


def _schema(model_name: str, **fields: tuple[Any, Any]) -> ToolArgsSchema:
    """创建 Pydantic 参数模型。

    中文注释：
    create_model(...) 会动态创建一个 BaseModel 子类。
    例如：

        _schema("ReadFile", path=(str, Field(...)))

    大致等价于手写：

        class ReadFileArgs(ToolArgsBase):
            path: str
    """

    return create_model(f"{model_name}Args", __base__=ToolArgsBase, **fields)


def _field(
    annotation: Any,
    *,
    required: bool = False,
    description: str = "",
    default: Any = None,
    default_factory: Callable[[], Any] | None = None,
) -> tuple[Any, Any]:
    """创建 Pydantic 字段定义。

    中文注释：
    required=True 时用 Field(...)，表示这个参数必填。
    required=False 时使用 default，表示这个参数可选。
    """

    if required:
        return annotation, Field(..., description=description)
    if default_factory is not None:
        return annotation, Field(default_factory=default_factory, description=description)
    return annotation, Field(default, description=description)


NO_ARGS_SCHEMA = _schema("No")
PATH_ARGS_SCHEMA = _schema(
    "Path",
    path=_field(str, description="当前 active project 内的相对路径。", default="."),
)
TEXT_FILE_ARGS_SCHEMA = _schema(
    "TextFile",
    path=_field(str, required=True, description="当前 active project 内允许处理的文本文件路径。"),
)
SYMBOL_ARGS_SCHEMA = _schema(
    "Symbol",
    symbol=_field(str, required=True, description="要检查的函数、类、变量或符号名。"),
)
OUTPUT_ARGS_SCHEMA = _schema(
    "Output",
    output=_field(str, required=True, description="测试、编译或命令输出文本。"),
)
PATCH_ARGS_SCHEMA = _schema(
    "Patch",
    path=_field(str, required=True, description="要修改的文本文件路径。"),
    old_text=_field(str, required=True, description="目标文件中必须唯一出现的原始文本。"),
    new_text=_field(str, required=True, description="替换后的新文本。"),
)
RESTORE_ARGS_SCHEMA = _schema(
    "Restore",
    path=_field(str, required=True, description="要恢复的文本文件路径。"),
    content=_field(str, required=True, description="恢复后的完整文件内容。"),
)


def _validate_path_dir(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认 path 指向当前 active project 内的目录。"""

    resolved = safe_resolve(_string_arg(tool_args, "path", "."))
    if not resolved.exists():
        return False, f"路径不存在：{tool_args.get('path', '.')}"
    if not resolved.is_dir():
        return False, f"不是目录：{tool_args.get('path', '.')}"
    return True, "目录路径安全。"


def _validate_text_file(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认 path 指向当前 active project 内允许处理的文本文件。"""

    safe_text_file(_string_arg(tool_args, "path"))
    return True, "文本文件路径安全。"


def _validate_safe_path(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """只检查 path 没有越过 active project 边界，不要求文件必须存在。"""

    safe_resolve(_string_arg(tool_args, "path", "."))
    return True, "路径在 active project 内。"


def _min_length_arg(name: str, label: str, minimum: int = 2) -> Validator:
    """生成“某个字符串参数至少 N 个字符”的 validator。"""

    def validate(tool_args: dict[str, Any]) -> tuple[bool, str]:
        if len(_string_arg(tool_args, name).strip()) < minimum:
            return False, f"{label} 需要至少 {minimum} 个字符的 {name}。"
        return True, f"{label} 的 {name} 参数安全。"

    return validate


def _required_arg(name: str, label: str) -> Validator:
    """生成“某个参数不能为空”的 validator。"""

    def validate(tool_args: dict[str, Any]) -> tuple[bool, str]:
        if not _string_arg(tool_args, name).strip():
            return False, f"{label} 需要 {name}。"
        return True, f"{label} 的 {name} 参数安全。"

    return validate


def _any_required_arg(names: tuple[str, ...], label: str) -> Validator:
    """生成“多个参数中至少有一个不能为空”的 validator。"""

    def validate(tool_args: dict[str, Any]) -> tuple[bool, str]:
        if not any(_string_arg(tool_args, name).strip() for name in names):
            return False, f"{label} 需要 {' 或 '.join(names)}。"
        return True, f"{label} 的输入参数安全。"

    return validate


def _choice_arg(name: str, choices: tuple[str, ...], label: str) -> Validator:
    """生成“参数必须在白名单中”的 validator。"""

    def validate(tool_args: dict[str, Any]) -> tuple[bool, str]:
        value = _string_arg(tool_args, name)
        if value not in choices:
            return False, f"{label} 的 {name} 不在白名单中。"
        return True, f"{label} 的 {name} 参数安全。"

    return validate


def _validate_run_allowed_command(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认命令 profile 来自 ALLOWED_COMMANDS 白名单。"""

    profile = _string_arg(tool_args, "profile")
    if profile not in ALLOWED_COMMANDS:
        return False, "run_allowed_command 的 profile 不在 ALLOWED_COMMANDS 白名单中。"
    return True, "命令 profile 安全。"


def _validate_rust_reference(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认 Rust 引用查询有 symbol，并且 path 没有越界。"""

    if len(_string_arg(tool_args, "symbol").strip()) < 2:
        return False, "inspect_rust_references 需要至少 2 个字符的 symbol。"
    safe_resolve(_string_arg(tool_args, "path", "."))
    return True, "Rust 引用查询参数安全。"


def _validate_compare_failure(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认前后两次失败信息都存在，才能比较修复效果。"""

    if not _string_arg(tool_args, "before") or not _string_arg(tool_args, "after"):
        return False, "compare_failure_before_after 需要 before 和 after。"
    return True, "失败对比参数安全。"


def _validate_patch_text(tool_name: str) -> Validator:
    """生成 patch 类工具的 validator。

    中文注释：
    这里做的是生产级 code agent 很重要的一层防护：
    - 文件必须在 active project 内。
    - old_text 不能为空。
    - old_text 和 new_text 不能一样。
    - old_text 必须只出现一次，避免误改多个位置。
    """

    def validate(tool_args: dict[str, Any]) -> tuple[bool, str]:
        path = _string_arg(tool_args, "path")
        old_text = _string_arg(tool_args, "old_text")
        new_text = _string_arg(tool_args, "new_text")
        file_path = safe_text_file(path)
        if not old_text:
            return False, f"{tool_name} 需要 old_text。"
        if old_text == new_text:
            return False, f"{tool_name} 的 old_text 和 new_text 不能相同。"
        occurrences = file_path.read_text(encoding="utf-8", errors="replace").count(old_text)
        if occurrences == 0:
            return False, f"{tool_name} 失败：old_text 在目标文件中不存在。"
        if occurrences > 1:
            return False, f"{tool_name} 失败：old_text 出现多次，修改不够精确。"
        return True, f"{tool_name} 的 patch 参数安全。"

    return validate


def _validate_restore_content(tool_name: str) -> Validator:
    """生成 rollback/revert 类工具的 validator。"""

    def validate(tool_args: dict[str, Any]) -> tuple[bool, str]:
        safe_text_file(_string_arg(tool_args, "path"))
        if not _string_arg(tool_args, "content"):
            return False, f"{tool_name} 需要 content。"
        return True, f"{tool_name} 的恢复参数安全。"

    return validate


def _validate_patch_plan(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认 PatchPlan 至少有目标文件和修改目标。"""

    safe_text_file(_string_arg(tool_args, "path"))
    if not _string_arg(tool_args, "goal").strip():
        return False, "patch_plan 需要 goal，说明这次修改想达成什么目标。"
    return True, "patch_plan 参数安全。"


def _validate_audit_patch(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认审计 patch 时至少知道文件和原因。"""

    safe_text_file(_string_arg(tool_args, "path"))
    if not _string_arg(tool_args, "reason").strip():
        return False, "audit_patch 需要 reason。"
    return True, "audit_patch 参数安全。"


def _validate_register_project_root(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认注册项目根不会越过 aios 工作区。"""

    project_id = _string_arg(tool_args, "project_id").strip()
    path = _string_arg(tool_args, "path").strip()
    if not project_id or not path:
        return False, "register_project_root 需要 project_id 和 path。"
    if not project_id.replace("_", "-").replace("-", "").isalnum():
        return False, "project_id 只能包含字母、数字、下划线和中划线。"
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return False, f"项目根不存在或不是目录：{path}"
    try:
        root.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return False, "项目根必须位于 /Users/christophermanning/Downloads/aios 工作区内。"
    return True, "项目根注册参数安全。"


def _validate_set_active_project(tool_args: dict[str, Any]) -> tuple[bool, str]:
    """确认要切换的 active project 已经注册。"""

    project_id = _string_arg(tool_args, "project_id").strip()
    if not project_id:
        return False, "set_active_project 需要 project_id。"
    if project_id not in list_project_roots():
        return False, f"未注册项目：{project_id}"
    return True, "active project 参数安全。"


_TOOL_VALIDATORS: dict[str, Validator] = {
    "list_files": _validate_path_dir,
    "list_tree": _validate_path_dir,
    "detect_rust_project": _validate_path_dir,
    "inspect_rust_symbols": _validate_path_dir,
    "read_file": _validate_text_file,
    "read_file_slice": _validate_text_file,
    "git_diff_file": _validate_text_file,
    "summarize_file": _validate_text_file,
    "format_apply": _validate_text_file,
    "validate_patch_scope": _validate_text_file,
    "search_code": _min_length_arg("query", "search_code"),
    "grep_regex": _min_length_arg("pattern", "grep_regex"),
    "inspect_symbol": _min_length_arg("symbol", "inspect_symbol"),
    "inspect_references": _min_length_arg("symbol", "inspect_references"),
    "inspect_function_signature": _min_length_arg("symbol", "inspect_function_signature"),
    "query_project_index": _min_length_arg("query", "query_project_index"),
    "inspect_class_hierarchy": _min_length_arg("class_name", "inspect_class_hierarchy"),
    "run_allowed_command": _validate_run_allowed_command,
    "run_package_script": _choice_arg(
        "script",
        (
            "test",
            "lint",
            "format_check",
            "typecheck",
            "build",
            "cargo_check",
            "cargo_test",
            "cargo_clippy",
            "cargo_fmt_check",
        ),
        "run_package_script",
    ),
    "run_pytest": _choice_arg("target", ("beginner_agent", ".", ""), "run_pytest"),
    "safe_path_exists": _validate_safe_path,
    "inspect_rust_references": _validate_rust_reference,
    "parse_rust_errors": _required_arg("output", "parse_rust_errors"),
    "parse_cargo_test_failure": _required_arg("output", "parse_cargo_test_failure"),
    "parse_test_failure": _any_required_arg(("output", "test_output"), "parse_test_failure"),
    "extract_stack_trace": _any_required_arg(("output", "test_output"), "extract_stack_trace"),
    "classify_failure": _any_required_arg(("output", "test_output"), "classify_failure"),
    "compare_failure_before_after": _validate_compare_failure,
    "preview_patch": _validate_patch_text("preview_patch"),
    "apply_patch_dry_run": _validate_patch_text("apply_patch_dry_run"),
    "apply_patch": _validate_patch_text("apply_patch"),
    "rollback": _validate_restore_content("rollback"),
    "revert_file_patch": _validate_restore_content("revert_file_patch"),
    "patch_plan": _validate_patch_plan,
    "validate_patch_plan": _required_arg("patch_plan_id", "validate_patch_plan"),
    "apply_patch_plan": _required_arg("patch_plan_id", "apply_patch_plan"),
    "checkpoint_save": _required_arg("name", "checkpoint_save"),
    "checkpoint_load": _required_arg("name", "checkpoint_load"),
    "audit_tool_call": _required_arg("tool_name", "audit_tool_call"),
    "describe_tool": _required_arg("tool_name", "describe_tool"),
    "audit_patch": _validate_audit_patch,
    "register_project_root": _validate_register_project_root,
    "set_active_project": _validate_set_active_project,
}


_TOOL_ARGS_SCHEMAS: dict[str, ToolArgsSchema] = {
    "list_files": PATH_ARGS_SCHEMA,
    "list_tree": _schema(
        "ListTree",
        path=_field(str, description="当前 active project 内的相对目录。", default="."),
        max_depth=_field(int, description="目录树最大展开层数。", default=2),
    ),
    "detect_rust_project": PATH_ARGS_SCHEMA,
    "inspect_rust_symbols": PATH_ARGS_SCHEMA,
    "read_file": TEXT_FILE_ARGS_SCHEMA,
    "read_file_slice": _schema(
        "ReadFileSlice",
        path=_field(str, required=True, description="要读取的文本文件路径。"),
        start=_field(int, description="起始行号，从 1 开始。", default=1),
        end=_field(int, description="结束行号。", default=120),
    ),
    "git_diff_file": TEXT_FILE_ARGS_SCHEMA,
    "summarize_file": TEXT_FILE_ARGS_SCHEMA,
    "format_apply": TEXT_FILE_ARGS_SCHEMA,
    "validate_patch_scope": _schema(
        "ValidatePatchScope",
        path=_field(str, required=True, description="准备修改的文本文件路径。"),
        goal=_field(str, description="这次修改的目标。", default=""),
    ),
    "search_code": _schema("SearchCode", query=_field(str, required=True, description="要搜索的代码关键词。")),
    "grep_regex": _schema(
        "GrepRegex",
        pattern=_field(str, required=True, description="正则表达式。"),
        path=_field(str, description="搜索范围目录。", default="."),
    ),
    "inspect_symbol": SYMBOL_ARGS_SCHEMA,
    "inspect_references": SYMBOL_ARGS_SCHEMA,
    "inspect_function_signature": SYMBOL_ARGS_SCHEMA,
    "inspect_call_graph": _schema("InspectCallGraph", function=_field(str, description="要分析调用关系的函数名。", default="")),
    "query_project_index": _schema("QueryProjectIndex", query=_field(str, required=True, description="项目索引查询关键词。")),
    "inspect_class_hierarchy": _schema("InspectClassHierarchy", class_name=_field(str, required=True, description="要检查继承关系的类名。")),
    "run_allowed_command": _schema("RunAllowedCommand", profile=_field(str, required=True, description="ALLOWED_COMMANDS 中的命令 profile。")),
    "run_package_script": _schema("RunPackageScript", script=_field(str, required=True, description="预定义包脚本名，例如 test / lint / cargo_check。")),
    "run_pytest": _schema("RunPytest", target=_field(str, description="pytest 白名单目标。", default="beginner_agent")),
    "safe_path_exists": PATH_ARGS_SCHEMA,
    "inspect_rust_references": _schema(
        "InspectRustReferences",
        symbol=_field(str, required=True, description="Rust 符号名。"),
        path=_field(str, description="Rust 项目内相对目录。", default="."),
    ),
    "parse_rust_errors": OUTPUT_ARGS_SCHEMA,
    "parse_cargo_test_failure": OUTPUT_ARGS_SCHEMA,
    "map_changed_rust_files_to_tests": _schema("MapChangedRustFilesToTests", changed_files=_field(Any, description="已变更 Rust 文件列表。", default="")),
    "parse_test_failure": _schema(
        "ParseTestFailure",
        output=_field(str, description="测试输出。", default=""),
        test_output=_field(str, description="测试输出的别名。", default=""),
    ),
    "extract_stack_trace": _schema(
        "ExtractStackTrace",
        output=_field(str, description="错误输出。", default=""),
        test_output=_field(str, description="测试输出的别名。", default=""),
    ),
    "classify_failure": _schema(
        "ClassifyFailure",
        output=_field(str, description="错误输出。", default=""),
        test_output=_field(str, description="测试输出的别名。", default=""),
    ),
    "compare_failure_before_after": _schema(
        "CompareFailureBeforeAfter",
        before=_field(str, required=True, description="修复前失败输出。"),
        after=_field(str, required=True, description="修复后失败输出。"),
    ),
    "select_relevant_tests": _schema(
        "SelectRelevantTests",
        query=_field(str, required=True, description="测试选择目标。"),
        changed_files=_field(Any, description="已变更文件列表。", default=""),
    ),
    "run_targeted_tests": _schema("RunTargetedTests", target=_field(str, description="白名单测试目标。", default="beginner_agent")),
    "map_changed_files_to_tests": _schema("MapChangedFilesToTests", changed_files=_field(Any, description="已变更 Python 文件列表。", default="")),
    "run_impacted_tests": _schema("RunImpactedTests", changed_files=_field(Any, description="已变更文件列表。", default="")),
    "secret_scan": PATH_ARGS_SCHEMA,
    "preview_patch": PATCH_ARGS_SCHEMA,
    "apply_patch_dry_run": PATCH_ARGS_SCHEMA,
    "apply_patch": PATCH_ARGS_SCHEMA,
    "rollback": RESTORE_ARGS_SCHEMA,
    "revert_file_patch": RESTORE_ARGS_SCHEMA,
    "patch_plan": _schema(
        "PatchPlan",
        path=_field(str, required=True, description="计划修改的文本文件路径。"),
        goal=_field(str, required=True, description="修改目标。"),
        old_text=_field(str, description="可选：计划替换的旧文本。", default=""),
        new_text=_field(str, description="可选：计划替换的新文本。", default=""),
    ),
    "validate_patch_plan": _schema("ValidatePatchPlan", patch_plan_id=_field(str, required=True, description="PatchPlan ID。")),
    "apply_patch_plan": _schema("ApplyPatchPlan", patch_plan_id=_field(str, required=True, description="PatchPlan ID。")),
    "checkpoint_save": _schema(
        "CheckpointSave",
        name=_field(str, required=True, description="checkpoint 名称。"),
        data=_field(Any, description="要保存的数据。", default=""),
    ),
    "checkpoint_load": _schema("CheckpointLoad", name=_field(str, required=True, description="checkpoint 名称。")),
    "audit_tool_call": _schema(
        "AuditToolCall",
        tool_name=_field(str, required=True, description="被审计的工具名。"),
        tool_args=_field(dict[str, Any], description="被审计的工具参数。", default_factory=dict),
        decision=_field(str, description="审批或执行决策。", default="record"),
    ),
    "audit_patch": _schema(
        "AuditPatch",
        path=_field(str, required=True, description="被审计的 patch 文件路径。"),
        reason=_field(str, required=True, description="审计原因。"),
        patch_plan_id=_field(str, description="关联 PatchPlan ID。", default=""),
    ),
    "describe_tool": _schema("DescribeTool", tool_name=_field(str, required=True, description="要查看的工具名。")),
    "read_audit_log": _schema("ReadAuditLog", limit=_field(int, description="读取最近多少条审计日志。", default=20)),
    "register_project_root": _schema(
        "RegisterProjectRoot",
        project_id=_field(str, required=True, description="项目 ID。"),
        path=_field(str, required=True, description="aios 工作区内的项目绝对路径。"),
    ),
    "set_active_project": _schema("SetActiveProject", project_id=_field(str, required=True, description="已注册的项目 ID。")),
}


def _validator_for_tool(name: str) -> Validator:
    """根据工具名返回 validator。

    中文注释：
    这个函数只在创建 ToolSpec 时使用一次。
    创建完成后，真正执行校验的是 spec.validator。
    这样可以避免 validate_tool_request 变成越来越长的 if/elif。
    """

    return _TOOL_VALIDATORS.get(name, _allow_any_args)


def _args_schema_for_tool(name: str) -> ToolArgsSchema:
    """根据工具名返回 Pydantic 参数模型。"""

    return _TOOL_ARGS_SCHEMAS.get(name, NO_ARGS_SCHEMA)


def _spec(
    name: str,
    *,
    access: Access = "read",
    category: Category,
    language: Language = "generic",
    risk: Risk = "low",
    handler: Callable[[dict[str, Any]], str],
    validator: Validator | None = None,
    args_schema: ToolArgsSchema | None = None,
    description: str = "",
) -> ToolSpec:
    """创建 ToolSpec 的小助手，减少重复字段噪音。"""

    return ToolSpec(
        name=name,
        access=access,
        category=category,
        language=language,
        risk=risk,
        handler=handler,
        validator=validator or _validator_for_tool(name),
        args_schema=args_schema if args_schema is not None else _args_schema_for_tool(name),
        description=description,
    )


def _build_tool_specs() -> dict[str, ToolSpec]:
    """集中登记所有工具。"""

    specs = [
        _spec("list_files", category="observe", handler=lambda args: list_files_tool(_string_arg(args, "path", "."))),
        _spec(
            "list_tree",
            category="observe",
            handler=lambda args: list_tree_tool(_string_arg(args, "path", "."), _int_arg(args, "max_depth", 2)),
        ),
        _spec("read_file", category="observe", handler=lambda args: read_file_tool(_string_arg(args, "path"))),
        _spec(
            "read_file_slice",
            category="observe",
            handler=lambda args: read_file_slice_tool(
                _string_arg(args, "path"),
                _int_arg(args, "start", 1),
                _int_arg(args, "end", 120),
            ),
        ),
        _spec("search_code", category="observe", handler=lambda args: search_code_tool(_string_arg(args, "query"))),
        _spec(
            "grep_regex",
            category="observe",
            handler=lambda args: grep_regex_tool(_string_arg(args, "pattern"), _string_arg(args, "path", ".")),
        ),
        _spec("inspect_symbol", category="understand", language="python", handler=lambda args: inspect_symbol_tool(_string_arg(args, "symbol"))),
        _spec("inspect_references", category="understand", language="python", handler=lambda args: inspect_references_tool(_string_arg(args, "symbol"))),
        _spec("inspect_import_graph", category="understand", language="python", handler=lambda args: inspect_import_graph_tool()),
        _spec("inspect_call_graph", category="understand", language="python", handler=lambda args: inspect_call_graph_tool(_string_arg(args, "function"))),
        _spec("build_project_index", category="understand", handler=lambda args: build_project_index_tool()),
        _spec("query_project_index", category="understand", handler=lambda args: query_project_index_tool(_string_arg(args, "query"))),
        _spec("inspect_function_signature", category="understand", language="python", handler=lambda args: inspect_function_signature_tool(_string_arg(args, "symbol"))),
        _spec("inspect_class_hierarchy", category="understand", language="python", handler=lambda args: inspect_class_hierarchy_tool(_string_arg(args, "class_name"))),
        _spec("detect_project_stack", category="understand", handler=lambda args: detect_project_stack_tool()),
        _spec("detect_test_framework", category="understand", handler=lambda args: detect_test_framework_tool()),
        _spec("detect_package_manager", category="understand", handler=lambda args: detect_package_manager_tool()),
        _spec("detect_build_system", category="understand", handler=lambda args: detect_build_system_tool()),
        _spec("list_allowed_commands", category="platform", language="platform", handler=lambda args: list_allowed_commands_tool()),
        _spec("run_allowed_command", category="verify", risk="medium", handler=lambda args: run_allowed_command_tool(_string_arg(args, "profile"))),
        _spec("run_package_script", category="verify", risk="medium", handler=lambda args: run_package_script_tool(_string_arg(args, "script"))),
        _spec("run_pytest", category="verify", language="python", risk="medium", handler=lambda args: run_pytest_tool(_string_arg(args, "target", "beginner_agent"))),
        _spec("run_ruff", category="verify", language="python", risk="medium", handler=lambda args: run_ruff_tool()),
        _spec("run_ruff_format_check", category="verify", language="python", risk="medium", handler=lambda args: run_ruff_format_check_tool()),
        _spec("run_mypy", category="verify", language="python", risk="medium", handler=lambda args: run_mypy_tool()),
        _spec("run_uv_import_graph", category="verify", language="python", risk="medium", handler=lambda args: run_uv_import_graph_tool()),
        _spec("run_cargo_check", category="verify", language="rust", risk="medium", handler=lambda args: run_cargo_check_tool()),
        _spec("run_cargo_test", category="verify", language="rust", risk="medium", handler=lambda args: run_cargo_test_tool()),
        _spec("run_cargo_clippy", category="verify", language="rust", risk="medium", handler=lambda args: run_cargo_clippy_tool()),
        _spec("run_cargo_fmt_check", category="verify", language="rust", risk="medium", handler=lambda args: run_cargo_fmt_check_tool()),
        _spec("detect_rust_project", category="understand", language="rust", handler=lambda args: detect_rust_project_tool(_string_arg(args, "path", "."))),
        _spec("inspect_rust_symbols", category="understand", language="rust", handler=lambda args: inspect_rust_symbols_tool(_string_arg(args, "path", "."))),
        _spec(
            "inspect_rust_references",
            category="understand",
            language="rust",
            handler=lambda args: inspect_rust_references_tool(_string_arg(args, "symbol"), _string_arg(args, "path", ".")),
        ),
        _spec("parse_rust_errors", category="verify", language="rust", handler=lambda args: parse_rust_errors_tool(_string_arg(args, "output"))),
        _spec("parse_cargo_test_failure", category="verify", language="rust", handler=lambda args: parse_cargo_test_failure_tool(_string_arg(args, "output"))),
        _spec("map_changed_rust_files_to_tests", category="verify", language="rust", handler=lambda args: map_changed_rust_files_to_tests_tool(args.get("changed_files", ""))),
        _spec("safe_path_exists", category="security", handler=lambda args: safe_path_exists_tool(_string_arg(args, "path", "."))),
        _spec("static_check", category="verify", language="python", handler=lambda args: static_check_tool()),
        _spec("lint_typecheck", category="verify", language="python", risk="medium", handler=lambda args: lint_typecheck_tool()),
        _spec("run_tests", category="verify", risk="medium", handler=lambda args: run_tests_tool()),
        _spec("run_targeted_tests", category="verify", risk="medium", handler=lambda args: run_targeted_tests_tool(_string_arg(args, "target", "beginner_agent"))),
        _spec("parse_test_failure", category="verify", language="python", handler=lambda args: parse_test_failure_tool(_string_arg(args, "test_output") or _string_arg(args, "output"))),
        _spec("extract_stack_trace", category="verify", handler=lambda args: extract_stack_trace_tool(_string_arg(args, "output") or _string_arg(args, "test_output"))),
        _spec("classify_failure", category="verify", handler=lambda args: classify_failure_tool(_string_arg(args, "output") or _string_arg(args, "test_output"))),
        _spec("compare_failure_before_after", category="verify", handler=lambda args: compare_failure_before_after_tool(_string_arg(args, "before"), _string_arg(args, "after"))),
        _spec("run_typecheck", category="verify", language="python", risk="medium", handler=lambda args: run_typecheck_tool()),
        _spec("run_build", category="verify", risk="medium", handler=lambda args: run_build_tool()),
        _spec("get_diagnostics", category="verify", risk="medium", handler=lambda args: get_diagnostics_tool()),
        _spec("format_check", category="verify", risk="medium", handler=lambda args: format_check_tool()),
        _spec("git_status", category="observe", handler=lambda args: git_status_tool()),
        _spec("git_diff", category="observe", handler=lambda args: git_diff_tool()),
        _spec("git_diff_file", category="observe", handler=lambda args: git_diff_file_tool(_string_arg(args, "path"))),
        _spec("map_changed_files_to_tests", category="verify", language="python", handler=lambda args: map_changed_files_to_tests_tool(args.get("changed_files", ""))),
        _spec("select_relevant_tests", category="verify", language="python", handler=lambda args: select_relevant_tests_tool(_string_arg(args, "query"), args.get("changed_files", ""))),
        _spec("run_impacted_tests", category="verify", language="python", risk="medium", handler=lambda args: run_impacted_tests_tool(args.get("changed_files", ""))),
        _spec("checkpoint_load", category="memory", handler=lambda args: checkpoint_load_tool(_string_arg(args, "name"))),
        _spec("secret_scan", category="security", handler=lambda args: secret_scan_tool(_string_arg(args, "path", "."))),
        _spec("dependency_inspect", category="understand", handler=lambda args: dependency_inspect_tool()),
        _spec("summarize_file", category="understand", handler=lambda args: summarize_file_tool(_string_arg(args, "path"))),
        _spec("validate_patch_plan", category="edit", risk="medium", handler=lambda args: validate_patch_plan_tool(_string_arg(args, "patch_plan_id"))),
        _spec("preview_patch", category="edit", risk="medium", handler=lambda args: preview_patch_tool(_string_arg(args, "path"), _string_arg(args, "old_text"), _string_arg(args, "new_text"))),
        _spec("apply_patch_dry_run", category="edit", risk="medium", handler=lambda args: apply_patch_dry_run_tool(_string_arg(args, "path"), _string_arg(args, "old_text"), _string_arg(args, "new_text"))),
        _spec("validate_patch_scope", category="edit", risk="medium", handler=lambda args: validate_patch_scope_tool(_string_arg(args, "path"), _string_arg(args, "goal"))),
        _spec("read_audit_log", category="audit", handler=lambda args: read_audit_log_tool(_int_arg(args, "limit", 20))),
        _spec("list_tool_catalog", category="platform", language="platform", handler=lambda args: list_tool_catalog_from_specs()),
        _spec("describe_tool", category="platform", language="platform", handler=lambda args: describe_tool_from_specs(_string_arg(args, "tool_name"))),
        _spec("tool_policy_report", category="platform", language="platform", handler=lambda args: tool_policy_report_from_specs()),
        _spec("list_project_roots", category="platform", language="platform", handler=lambda args: list_project_roots_tool()),
        _spec("get_active_project", category="platform", language="platform", handler=lambda args: get_active_project_tool()),
        _spec("apply_patch", access="write", category="edit", risk="high", handler=lambda args: apply_patch_tool(_string_arg(args, "path"), _string_arg(args, "old_text"), _string_arg(args, "new_text"))),
        _spec("rollback", access="write", category="edit", risk="high", handler=lambda args: rollback_tool(_string_arg(args, "path"), _string_arg(args, "content"))),
        _spec("format_apply", access="write", category="edit", risk="medium", handler=lambda args: format_apply_tool(_string_arg(args, "path"))),
        _spec("patch_plan", access="write", category="edit", risk="medium", handler=lambda args: patch_plan_tool(_string_arg(args, "path"), _string_arg(args, "goal"), _string_arg(args, "old_text"), _string_arg(args, "new_text"))),
        _spec("apply_patch_plan", access="write", category="edit", risk="high", handler=lambda args: apply_patch_plan_tool(_string_arg(args, "patch_plan_id"))),
        _spec("revert_file_patch", access="write", category="edit", risk="high", handler=lambda args: revert_file_patch_tool(_string_arg(args, "path"), _string_arg(args, "content"))),
        _spec("checkpoint_save", access="write", category="memory", risk="medium", handler=lambda args: checkpoint_save_tool(_string_arg(args, "name"), args.get("data", ""))),
        _spec(
            "audit_tool_call",
            access="write",
            category="audit",
            language="platform",
            risk="medium",
            handler=lambda args: audit_tool_call_tool(
                _string_arg(args, "tool_name"),
                dict(args.get("tool_args", {})) if isinstance(args.get("tool_args", {}), dict) else {},
                _string_arg(args, "decision", "record"),
            ),
        ),
        _spec("audit_patch", access="write", category="audit", language="platform", risk="medium", handler=lambda args: audit_patch_tool(_string_arg(args, "path"), _string_arg(args, "reason"), _string_arg(args, "patch_plan_id"))),
        _spec("register_project_root", access="write", category="platform", language="platform", risk="medium", handler=lambda args: register_project_root_tool(_string_arg(args, "project_id"), _string_arg(args, "path"))),
        _spec("set_active_project", access="write", category="platform", language="platform", risk="medium", handler=lambda args: set_active_project_tool(_string_arg(args, "project_id"))),
    ]
    return {spec.name: spec for spec in specs}


TOOL_SPECS = _build_tool_specs()
READ_ONLY_TOOLS = tuple(name for name, spec in TOOL_SPECS.items() if spec.access == "read")
WRITE_TOOLS = tuple(name for name, spec in TOOL_SPECS.items() if spec.access == "write")
ALL_TOOLS = tuple(TOOL_SPECS)


def list_tool_catalog_from_specs() -> str:
    """从 ToolSpec 输出工具目录。

    中文注释：
    这避免了“目录里写一份元数据，registry 里又写一份元数据”的问题。
    真实系统里应尽量保持单一事实来源。
    """

    return json_dumps(
        {
            "tool_count": len(TOOL_SPECS),
            "read_only_count": len(READ_ONLY_TOOLS),
            "write_count": len(WRITE_TOOLS),
            "tools": [spec.to_dict() for spec in TOOL_SPECS.values()],
        }
    )


def describe_tool_from_specs(tool_name: str) -> str:
    """从 ToolSpec 查看单个工具。"""

    spec = TOOL_SPECS.get(tool_name)
    if spec is None:
        return json_dumps({"status": "not_found", "tool_name": tool_name})
    return json_dumps({"status": "found", "tool": spec.to_dict()})


def tool_policy_report_from_specs() -> str:
    """从 ToolSpec 输出工具权限策略报告。"""

    return json_dumps(
        {
            "read_only_tools": list(READ_ONLY_TOOLS),
            "write_tools": list(WRITE_TOOLS),
            "policy": {
                "read_only_default": "allow",
                "write_default": "ask",
                "unknown_tool": "deny",
                "absolute_path": "deny",
                "path_traversal": "deny",
                "write_requires_approval": True,
                "source_of_truth": "TOOL_SPECS",
            },
            "tools": {
                name: {
                    "access": spec.access,
                    "risk": spec.risk,
                    "category": spec.category,
                    "language": spec.language,
                    "requires_approval": spec.requires_approval,
                }
                for name, spec in TOOL_SPECS.items()
            },
        }
    )

FAILED_TOOL_RESULT_PREFIXES = (
    "路径不存在",
    "文件不存在",
    "不允许",
    "工具拒绝",
    "工具安全检查失败",
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


def _tool_output_status(output: str) -> ToolRunStatus:
    """把旧的字符串输出粗略归一成结构化状态。"""

    if output.startswith(FAILED_TOOL_RESULT_PREFIXES):
        return "failed"
    if output.startswith(("工具 ", "不允许", "未知工具")):
        return "blocked"
    return "success"


def _format_schema_error(tool_name: str, exc: ValidationError) -> str:
    """把 Pydantic 校验错误转成适合给用户和 LLM 看的短消息。"""

    first_error = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(part) for part in first_error.get("loc", ())) or "参数"
    message = first_error.get("msg", "参数不符合 schema")
    return f"{tool_name} 参数结构不符合 Pydantic schema：{loc} - {message}。"


def _validate_tool_request_internal(
    tool_name: str,
    tool_args: dict[str, Any],
) -> tuple[bool, str, dict[str, Any], ToolSpec | None]:
    """工具参数校验的内部入口。

    中文注释：
    这里比 validate_tool_request(...) 多返回一个 normalized_args。

    流程是：

        原始 tool_args
          -> Pydantic args_schema.model_validate(...)
          -> normalized_args
          -> spec.validator(normalized_args)

    Pydantic 负责“结构和类型”。
    validator 负责“业务安全规则”，例如路径不能越界、命令必须白名单。
    """

    spec = TOOL_SPECS.get(tool_name)
    if spec is None:
        return False, f"工具 {tool_name} 不在工具白名单中。", tool_args, None

    try:
        normalized_args = spec.args_schema.model_validate(tool_args).model_dump()
    except ValidationError as exc:
        return False, _format_schema_error(tool_name, exc), tool_args, spec

    try:
        ok, reason = spec.validator(normalized_args)
    except ValueError as exc:
        return False, str(exc), normalized_args, spec

    if not ok:
        return False, reason, normalized_args, spec
    return True, reason or f"{tool_name} 参数安全。", normalized_args, spec


def validate_tool_request(tool_name: str, tool_args: dict[str, Any]) -> tuple[bool, str]:
    """校验工具参数。

    中文注释：
    现在校验逻辑已经由 ToolSpec.validator 驱动。

    你可以这样理解：

        tool_name
          -> 找到 ToolSpec
          -> 调用 spec.validator(tool_args)
          -> 通过后再执行 spec.handler(tool_args)

    这就是更接近大厂工具平台的写法：
    工具的“说明、风险、权限、参数校验、执行入口”都挂在同一个规格对象上。
    """

    ok, reason, _normalized_args, _spec = _validate_tool_request_internal(tool_name, tool_args)
    return ok, reason


def run_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """统一工具执行入口。

    中文注释：
    这是兼容旧代码的入口，仍然返回字符串。
    新代码如果需要更稳定的机器可读结果，优先使用 run_tool_result(...)。
    """

    return str(run_tool_result(tool_name, tool_args)["output"])


def _changed_files_from_tool(tool_name: str, normalized_args: dict[str, Any]) -> list[str]:
    """从工具名和参数中提取可能被修改的文件。

    中文注释：
    这只是保守估计。
    真正生产级可以让每个写工具明确返回 changed_files。
    """

    if tool_name not in {
        "apply_patch",
        "apply_patch_plan",
        "format_apply",
        "rollback",
        "revert_file_patch",
    }:
        return []
    path = str(normalized_args.get("path", ""))
    return [path] if path else []


def run_tool_model(tool_name: str, tool_args: dict[str, Any]) -> ToolResult:
    """结构化工具执行入口，返回 Pydantic ToolResult。

    中文注释：
    生产级 agent 不应该只靠字符串判断工具是否成功。
    这里返回 ToolResult model，Executor / Evaluator / Audit / Memory 都可以直接消费。
    """

    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    is_valid, reason, normalized_args, spec = _validate_tool_request_internal(tool_name, tool_args)
    if not is_valid:
        duration_ms = int((perf_counter() - started) * 1000)
        return ToolResult(
            status="blocked",
            tool_name=tool_name,
            tool_args=tool_args,
            normalized_args=normalized_args,
            output=reason,
            validation=ToolValidation(ok=False, reason=reason),
            metadata=spec.to_dict() if spec else {},
            diagnostics={"tool_result_schema": tool_result_json_schema()},
            started_at=started_at,
            duration_ms=duration_ms,
            error_type="validation_error",
            retryable=False,
        )

    if spec is None:
        output = f"未知工具：{tool_name}"
        duration_ms = int((perf_counter() - started) * 1000)
        return ToolResult(
            status="blocked",
            tool_name=tool_name,
            tool_args=tool_args,
            normalized_args=normalized_args,
            output=output,
            validation=ToolValidation(ok=False, reason=output),
            metadata={},
            diagnostics={"tool_result_schema": tool_result_json_schema()},
            started_at=started_at,
            duration_ms=duration_ms,
            error_type="unknown_tool",
            retryable=False,
        )

    try:
        output = spec.handler(normalized_args)
    except ValueError as exc:
        output = f"工具安全检查失败：{exc}"
        duration_ms = int((perf_counter() - started) * 1000)
        return ToolResult(
            status="blocked",
            tool_name=tool_name,
            tool_args=tool_args,
            normalized_args=normalized_args,
            output=output,
            validation=ToolValidation(ok=True, reason=reason),
            metadata=spec.to_dict(),
            diagnostics={"tool_result_schema": tool_result_json_schema()},
            started_at=started_at,
            duration_ms=duration_ms,
            error_type="safety_error",
            retryable=False,
        )

    status = classify_tool_output(output)
    duration_ms = int((perf_counter() - started) * 1000)
    return ToolResult(
        status=status,
        tool_name=tool_name,
        tool_args=tool_args,
        normalized_args=normalized_args,
        output=output,
        validation=ToolValidation(ok=True, reason=reason),
        metadata=spec.to_dict(),
        diagnostics={"tool_result_schema": tool_result_json_schema()},
        changed_files=_changed_files_from_tool(tool_name, normalized_args),
        started_at=started_at,
        duration_ms=duration_ms,
        retryable=retryable_from_status(status),
    )


def run_tool_result(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """兼容旧代码的结构化工具执行入口，返回普通 dict。"""

    return run_tool_model(tool_name, tool_args).model_dump(mode="json")
