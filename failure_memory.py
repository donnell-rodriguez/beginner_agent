from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


FailureCategory = Literal[
    "test_failure",
    "assertion_failure",
    "syntax_error",
    "type_error",
    "import_error",
    "build_failure",
    "rust_compile_error",
    "permission_blocked",
    "timeout",
    "environment_issue",
    "dependency_issue",
    "unknown",
]

FailureOwner = Literal["code", "environment", "permission", "dependency", "unknown"]
RetryClass = Literal[
    "retryable",
    "repair_required",
    "retry_after_environment_fix",
    "ask_human",
    "non_retryable",
]


@dataclass(frozen=True)
class FailureMemoryProfile:
    """FailureMemoryProfile：一条失败记忆的结构化画像。

    中文注释：
    真正的 code agent 很依赖“失败经验库”。
    它不只是保存失败文本，而是把失败整理成可以复用的知识：
    - 失败属于哪一类。
    - 是代码问题、环境问题、权限问题，还是依赖问题。
    - 以后遇到同类失败应该重试、修代码、修环境，还是问人。
    - 过去是否有从失败走向成功的修复路径。
    """

    pattern_id: str
    category: FailureCategory
    owner: FailureOwner
    retry_class: RetryClass
    failure_stage: str
    stack_signature: str
    locations: tuple[str, ...]
    similar_failure_ids: tuple[str, ...]
    resolved_failure_pattern_ids: tuple[str, ...]
    successful_repair_memory_ids: tuple[str, ...]
    successful_repair_paths: tuple[str, ...]
    recommendation: str
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """转成可写入 MemoryRecord.metadata 的普通 dict。"""

        return {
            "pattern_id": self.pattern_id,
            "category": self.category,
            "owner": self.owner,
            "retry_class": self.retry_class,
            "failure_stage": self.failure_stage,
            "stack_signature": self.stack_signature,
            "locations": list(self.locations),
            "similar_failure_ids": list(self.similar_failure_ids),
            "resolved_failure_pattern_ids": list(self.resolved_failure_pattern_ids),
            "successful_repair_memory_ids": list(self.successful_repair_memory_ids),
            "successful_repair_paths": list(self.successful_repair_paths),
            "recommendation": self.recommendation,
            "evidence": list(self.evidence),
        }


def _text(record: dict[str, Any]) -> str:
    """拼出失败分析使用的文本。"""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    source_memory = metadata.get("source_memory")
    source_memory = source_memory if isinstance(source_memory, dict) else {}
    tool_result_data = metadata.get("tool_result_data")
    tool_result_data = tool_result_data if isinstance(tool_result_data, dict) else {}
    parts = [
        str(record.get("title", "")),
        str(record.get("summary", "")),
        str(record.get("status", "")),
        str(record.get("tool_name", "")),
        str(record.get("tool_result_status", "")),
        str(source_memory.get("reason", "")),
        str(source_memory.get("tool_result", "")),
        str(tool_result_data.get("stdout", "")),
        str(tool_result_data.get("stderr", "")),
        str(tool_result_data.get("output", "")),
        str(tool_result_data.get("summary", "")),
        str(tool_result_data.get("error", "")),
    ]
    return "\n".join(part for part in parts if part).strip()


def _normalized_text(record: dict[str, Any]) -> str:
    """小写化文本，便于规则匹配。"""

    return _text(record).lower()


def _category(record: dict[str, Any]) -> FailureCategory:
    """把失败文本归类成稳定枚举。"""

    text = _normalized_text(record)
    tool_name = str(record.get("tool_name", "")).lower()
    if "syntaxerror" in text or "语法错误" in text:
        return "syntax_error"
    if "borrow checker" in text or "error[e" in text or "cargo check" in text:
        return "rust_compile_error"
    if "modulenotfounderror" in text or "importerror" in text:
        return "import_error"
    if "assertionerror" in text or "assert" in text or "断言" in text:
        return "assertion_failure"
    if "mypy" in text or "pyright" in text or "typeerror" in text:
        return "type_error"
    if "permission" in text or "operation not permitted" in text or "权限" in text:
        return "permission_blocked"
    if "timeout" in text or "timed out" in text or "超时" in text:
        return "timeout"
    if "connection refused" in text or "no such file or directory" in text:
        return "environment_issue"
    if "dependency" in text or "package not found" in text or "lockfile" in text:
        return "dependency_issue"
    if "build" in text or "compile" in text or tool_name in {"run_build", "run_cargo_build"}:
        return "build_failure"
    if (
        "failed" in text
        or "测试失败" in text
        or tool_name in {"run_tests", "run_targeted_tests"}
    ):
        return "test_failure"
    return "unknown"


def _owner(category: FailureCategory) -> FailureOwner:
    """判断失败更像归属于哪一层。"""

    if category in {
        "test_failure",
        "assertion_failure",
        "syntax_error",
        "type_error",
        "build_failure",
        "rust_compile_error",
    }:
        return "code"
    if category in {"environment_issue", "timeout"}:
        return "environment"
    if category == "permission_blocked":
        return "permission"
    if category in {"import_error", "dependency_issue"}:
        return "dependency"
    return "unknown"


def _retry_class(category: FailureCategory, record: dict[str, Any]) -> RetryClass:
    """给恢复循环一个明确动作方向。"""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    tool_data = metadata.get("tool_result_data")
    tool_data = tool_data if isinstance(tool_data, dict) else {}
    if bool(tool_data.get("retryable")):
        return "retryable"
    if category in {"permission_blocked"}:
        return "ask_human"
    if category in {"environment_issue", "dependency_issue", "timeout"}:
        return "retry_after_environment_fix"
    if category in {
        "test_failure",
        "assertion_failure",
        "syntax_error",
        "type_error",
        "build_failure",
        "rust_compile_error",
        "import_error",
    }:
        return "repair_required"
    return "non_retryable"


def _failure_stage(record: dict[str, Any]) -> str:
    """根据工具名粗略判断失败阶段。"""

    tool = str(record.get("tool_name", "none"))
    if tool in {"run_tests", "run_targeted_tests", "run_cargo_test"}:
        return "test"
    if tool in {"run_build", "run_cargo_build"}:
        return "build"
    if tool in {"run_typecheck", "lint_typecheck", "run_cargo_check"}:
        return "static_verification"
    if tool in {"apply_patch", "apply_patch_plan"}:
        return "patch"
    if tool in {"list_files", "read_file", "search_code", "inspect_symbol"}:
        return "inspection"
    return "execution"


def _locations(record: dict[str, Any]) -> tuple[str, ...]:
    """从失败文本和 paths 中提取位置线索。"""

    found = [
        f"{path}:{line}"
        for path, line in re.findall(
            r"([\w./-]+\.(?:py|rs|toml|md|json|yaml|yml|txt)):(\d+)",
            _text(record),
        )
    ]
    paths = record.get("paths", [])
    if isinstance(paths, list):
        found.extend(str(path) for path in paths if path)
    return tuple(sorted(dict.fromkeys(found))[:20])


def _stack_signature(record: dict[str, Any]) -> str:
    """提取稳定的错误签名，用于识别同类失败。"""

    text = _text(record)
    patterns = [
        r"([A-Za-z_][\w.]*Error):\s*([^\n]+)",
        r"([A-Za-z_][\w.]*Exception):\s*([^\n]+)",
        r"error(?:\[[A-Z0-9]+\])?:\s*([^\n]+)",
        r"FAILED\s+([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return " ".join(part.strip() for part in match.groups())[:240]
    first_meaningful = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_meaningful[:240] or "unknown"


def _pattern_id(
    category: FailureCategory,
    stage: str,
    signature: str,
    locations: tuple[str, ...],
) -> str:
    """生成失败模式 ID。"""

    path_names = [Path(location.split(":")[0]).name for location in locations[:5]]
    raw = "|".join([category, stage, signature, ",".join(path_names)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _failure_profile(record: dict[str, Any]) -> dict[str, Any]:
    """从旧记录 metadata 中读取失败画像。"""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    profile = metadata.get("failure_memory")
    return profile if isinstance(profile, dict) else {}


def _record_pattern_id(record: dict[str, Any]) -> str:
    """拿到一条记录的失败模式 ID。"""

    profile = _failure_profile(record)
    if profile.get("pattern_id"):
        return str(profile["pattern_id"])
    category = _category(record)
    stage = _failure_stage(record)
    locations = _locations(record)
    return _pattern_id(category, stage, _stack_signature(record), locations)


def _similar_failures(
    pattern_id: str,
    existing_records: list[dict[str, Any]],
) -> tuple[str, ...]:
    """查找同类失败历史。"""

    ids: list[str] = []
    for item in existing_records:
        if str(item.get("kind", "")) != "failure":
            continue
        if _record_pattern_id(item) == pattern_id:
            ids.append(str(item.get("id", "")))
    return tuple(id_ for id_ in ids if id_)[:20]


def _successful_repairs(
    pattern_id: str,
    locations: tuple[str, ...],
    existing_records: list[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """查找失败后成功的修复路径。"""

    location_files = {location.split(":")[0] for location in locations}
    memory_ids: list[str] = []
    repair_paths: list[str] = []
    for item in existing_records:
        status = str(item.get("tool_result_status", ""))
        if status != "success":
            continue
        item_profile = _failure_profile(item)
        linked_patterns = item_profile.get("resolved_failure_pattern_ids", [])
        if isinstance(linked_patterns, str):
            linked_patterns = [linked_patterns]
        item_paths = {str(path) for path in item.get("paths", [])}
        pattern_match = pattern_id in {str(value) for value in linked_patterns}
        path_match = bool(location_files and location_files.intersection(item_paths))
        if pattern_match or path_match:
            memory_id = str(item.get("id", ""))
            if memory_id:
                memory_ids.append(memory_id)
            repair_paths.extend(sorted(item_paths))
    return (
        tuple(dict.fromkeys(memory_ids).keys())[:20],
        tuple(dict.fromkeys(repair_paths).keys())[:20],
    )


def _recommendation(category: FailureCategory, retry_class: RetryClass) -> str:
    """给恢复/规划阶段的建议。"""

    if retry_class == "ask_human":
        return "不要自动重试，先请求人工确认权限、路径或安全边界。"
    if retry_class == "retry_after_environment_fix":
        return "先检查运行环境、依赖、服务状态或超时预算，再重试。"
    if category in {"syntax_error", "type_error", "rust_compile_error"}:
        return (
            "优先读取编译器/类型检查定位，生成最小补丁，"
            "再运行定向验证。"
        )
    if category in {"test_failure", "assertion_failure"}:
        return (
            "优先解析失败测试、定位断言差异，"
            "再做最小修复并运行目标测试。"
        )
    if category == "import_error":
        return (
            "先检查模块路径、依赖声明和运行入口，"
            "再决定修改代码或环境。"
        )
    return "先保留失败证据，避免盲目循环；必要时重新规划。"


def _evidence(record: dict[str, Any]) -> tuple[str, ...]:
    """提取这条失败记忆保留了哪些证据。"""

    evidence: list[str] = []
    text = _text(record)
    if "traceback" in text.lower():
        evidence.append("stack_trace")
    if _locations(record):
        evidence.append("locations")
    if str(record.get("tool_name", "none")) != "none":
        evidence.append("tool_name")
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("tool_result_data"):
        evidence.append("tool_result_data")
    if metadata.get("source_memory"):
        evidence.append("source_memory")
    return tuple(dict.fromkeys(evidence).keys())


def build_failure_memory_profile(
    record: dict[str, Any],
    existing_records: list[dict[str, Any]],
) -> FailureMemoryProfile | None:
    """构建失败经验库画像。

    中文注释：
    只有失败、阻塞、空结果、部分结果，或者明确来自验证工具的记录，
    才需要进入 Failure Memory Library。
    成功记录也可能携带 resolved_failure_pattern_ids，
    这样后续能知道“哪个失败模式最终被哪条成功修复解决”。
    """

    status = str(record.get("tool_result_status", "none"))
    kind = str(record.get("kind", "task"))
    tool = str(record.get("tool_name", "none"))
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    source_memory = metadata.get("source_memory")
    source_memory = source_memory if isinstance(source_memory, dict) else {}
    has_resolved_patterns = bool(
        source_memory.get("resolved_failure_pattern_ids")
        or metadata.get("resolved_failure_pattern_ids")
    )
    should_profile = (
        kind == "failure"
        or status in {"failed", "blocked", "empty", "partial"}
        or has_resolved_patterns
        or tool in {"run_tests", "run_targeted_tests", "run_build", "run_typecheck"}
        or tool in {"run_cargo_test", "run_cargo_check", "run_cargo_build"}
    )
    if not should_profile:
        return None

    category = _category(record)
    stage = _failure_stage(record)
    locations = _locations(record)
    signature = _stack_signature(record)
    pattern_id = _pattern_id(category, stage, signature, locations)
    retry_class = _retry_class(category, record)
    repair_ids, repair_paths = _successful_repairs(
        pattern_id,
        locations,
        existing_records,
    )
    resolved_patterns = source_memory.get(
        "resolved_failure_pattern_ids",
        metadata.get("resolved_failure_pattern_ids", []),
    )
    if isinstance(resolved_patterns, str):
        resolved_patterns = [resolved_patterns]
    if not isinstance(resolved_patterns, list):
        resolved_patterns = []
    evidence = list(_evidence(record))
    if resolved_patterns:
        evidence.append("resolved_failure_pattern_ids")
    return FailureMemoryProfile(
        pattern_id=pattern_id,
        category=category,
        owner=_owner(category),
        retry_class=retry_class,
        failure_stage=stage,
        stack_signature=signature,
        locations=locations,
        similar_failure_ids=_similar_failures(pattern_id, existing_records),
        resolved_failure_pattern_ids=tuple(str(value) for value in resolved_patterns),
        successful_repair_memory_ids=repair_ids,
        successful_repair_paths=repair_paths,
        recommendation=_recommendation(category, retry_class),
        evidence=tuple(dict.fromkeys(evidence).keys()),
    )


def failure_rerank_signal(record: dict[str, Any]) -> dict[str, Any]:
    """给 MemoryReranker 使用的失败库信号。"""

    profile = _failure_profile(record)
    if not profile:
        return {"has_failure_profile": False, "failure_weight": 0.0}
    retry_class = str(profile.get("retry_class", "non_retryable"))
    owner = str(profile.get("owner", "unknown"))
    has_success_path = bool(profile.get("successful_repair_memory_ids"))
    weight = 0.35
    if retry_class == "repair_required":
        weight += 0.25
    if owner == "code":
        weight += 0.15
    if has_success_path:
        weight += 0.25
    if retry_class in {"ask_human", "non_retryable"}:
        weight -= 0.2
    return {
        "has_failure_profile": True,
        "failure_weight": max(0.0, min(1.0, round(weight, 4))),
        "failure_category": profile.get("category", "unknown"),
        "failure_owner": owner,
        "retry_class": retry_class,
        "pattern_id": profile.get("pattern_id", ""),
        "has_successful_repair_path": has_success_path,
    }
