from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from .failure import build_failure_memory_profile
from .models import MemoryKind, MemoryPolicyDecision, MemoryRecord, MemoryScope, SensitivityLevel
from .quality import MemoryEvaluator, adjusted_memory_fields
from .settings import (
    DEFAULT_MEMORY_TTL_DAYS,
    DEFAULT_PROJECT_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_USER_ID,
    DEFAULT_WORKSPACE_ID,
    MAX_MEMORY_TEXT_CHARS,
)
from .preference import default_preference_payloads, is_preference_record, preference_metadata_from_pending
from ..privacy_governance import (
    memory_prompt_allowed_by_privacy,
    privacy_metadata,
    redact_text_for_memory,
    redact_value_for_memory,
    scan_value_for_privacy,
    storage_summary_for_sensitive_memory,
    stronger_sensitivity,
)
from ..state import State

def _memory_ttl_days() -> int:
    """读取 TTL 记忆默认保留天数。"""

    raw = os.getenv("BEGINNER_AGENT_MEMORY_TTL_DAYS", str(DEFAULT_MEMORY_TTL_DAYS))
    try:
        return max(1, int(raw.strip()))
    except ValueError:
        return DEFAULT_MEMORY_TTL_DAYS


def _expires_at_for_policy(retention_policy: RetentionPolicy) -> str | None:
    """根据 retention_policy 计算 expires_at。"""

    if retention_policy in {"none", "session", "long_term", "pinned"}:
        return None
    return (datetime.now(timezone.utc) + timedelta(days=_memory_ttl_days())).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    """安全解析 ISO datetime。"""

    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _record_is_active(record: dict[str, Any]) -> bool:
    """判断记忆是否仍可被默认检索使用。"""

    if str(record.get("validity_status", "active")) != "active":
        return False
    if bool(record.get("pinned", False)):
        return True
    expires_at = _parse_datetime(record.get("expires_at"))
    return expires_at is None or expires_at > datetime.now(timezone.utc)


def _record_should_be_deleted(record: dict[str, Any]) -> bool:
    """判断过期记忆是否应该被物理清理。"""

    if bool(record.get("pinned", False)):
        return False
    expires_at = _parse_datetime(record.get("expires_at"))
    return expires_at is not None and expires_at <= datetime.now(timezone.utc)


def _record_created_at(record: dict[str, Any]) -> datetime:
    """读取记录创建时间，解析失败时使用很早的时间。"""

    return _parse_datetime(record.get("created_at")) or datetime.min.replace(
        tzinfo=timezone.utc
    )


def _access_value_from_state_or_env(
    state: State,
    state_key: str,
    env_key: str,
    default: str,
) -> str:
    """从 State 或环境变量读取访问控制身份。"""

    value = state.get(state_key)
    if value:
        return str(value)
    return os.getenv(env_key, default).strip() or default


def _memory_access_context(state: State) -> dict[str, str]:
    """构造当前请求的 memory access context。

    中文注释：
    大厂级 memory 不会只问“语义相似吗”，还会先问：
    - 当前用户是谁？
    - 当前项目是谁？
    - 当前 workspace / tenant 是谁？

    这些身份字段用于隔离不同用户、不同项目、不同组织的记忆。
    """

    return {
        "tenant_id": _access_value_from_state_or_env(
            state, "tenant_id", "BEGINNER_AGENT_TENANT_ID", DEFAULT_TENANT_ID
        ),
        "workspace_id": _access_value_from_state_or_env(
            state,
            "workspace_id",
            "BEGINNER_AGENT_WORKSPACE_ID",
            DEFAULT_WORKSPACE_ID,
        ),
        "project_id": _access_value_from_state_or_env(
            state, "project_id", "BEGINNER_AGENT_PROJECT_ID", DEFAULT_PROJECT_ID
        ),
        "user_id": _access_value_from_state_or_env(
            state, "user_id", "BEGINNER_AGENT_USER_ID", DEFAULT_USER_ID
        ),
    }


def _record_acl_identity(record: dict[str, Any]) -> dict[str, str]:
    """读取记录里的 ACL 身份字段，兼容旧记录默认值。"""

    return {
        "tenant_id": str(record.get("tenant_id") or DEFAULT_TENANT_ID),
        "workspace_id": str(record.get("workspace_id") or DEFAULT_WORKSPACE_ID),
        "project_id": str(record.get("project_id") or DEFAULT_PROJECT_ID),
        "user_id": str(record.get("user_id") or DEFAULT_USER_ID),
    }


def _preference_visibility(scope: str) -> str:
    """根据 preference scope 选择默认可见性。"""

    if scope == "user":
        return "private"
    if scope == "workspace":
        return "workspace"
    if scope == "global":
        return "tenant"
    return "project"


def _preference_memory_records_for_context(state: State) -> list[MemoryRecord]:
    """把默认偏好种子转成 MemoryRecord。

    中文注释：
    这些是用户已经反复明确过的长期偏好。
    它们被建模成 MemoryRecord，而不是只写在 prompt / skill 里。
    这样后续 memory retrieval、审计、覆盖、失效都能统一处理。
    """

    identity = _memory_access_context(state)
    records: list[MemoryRecord] = []
    for payload in default_preference_payloads(identity):
        preference = payload["preference"]
        scope = str(preference.get("scope", "project"))
        visibility = _preference_visibility(scope)
        records.append(
            MemoryRecord(
                id=str(payload["id"]),
                kind="user" if scope == "user" else "project",
                task_id="preference-seed",
                title=f"Preference: {preference['key']}",
                summary=str(preference["value"]),
                status="active",
                tool_name="none",
                tool_result_status="success",
                tags=[
                    "preference",
                    str(preference.get("category", "")),
                    str(preference.get("scope", "")),
                ],
                confidence=0.95,
                importance=min(1.0, float(preference.get("priority", 80)) / 100),
                quality_score=0.9,
                trust_score=0.95,
                decay_score=0.0,
                scope="user" if scope == "user" else "project",
                visibility=visibility,  # type: ignore[arg-type]
                sensitivity_level="internal",
                tenant_id=identity["tenant_id"],
                workspace_id=identity["workspace_id"],
                project_id=identity["project_id"],
                user_id=identity["user_id"],
                retention_policy="pinned",
                validity_status="active",
                pinned=True,
                expires_at=None,
                contradiction_key=f"preference:{scope}:{preference['key']}",
                metadata={
                    "preference_memory": preference,
                    "memory_policy": {
                        "action": "store",
                        "reason": "默认长期偏好种子。",
                    },
                },
            )
        )
    return records

def _preference_records_for_state(state: State) -> tuple[list[dict[str, Any]], str, str]:
    """读取当前 user/project 可用的偏好记忆。"""

    # 中文注释：
    # _list_memory_records 在 retrieval 层。
    # 这里放在函数内部导入，是为了避免模块加载时互相 import。
    from .retrieval import _list_memory_records

    records, backend, error = _list_memory_records()
    preferences = [
        {
            **record,
            "access_control": _record_access_control(record, state),
        }
        for record in records
        if is_preference_record(record)
        and _record_visible_to_context(record, state)
        and _record_allowed_in_prompt(record)
    ]
    return preferences, backend, error


def _dedupe_contradiction_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同 contradiction_key 只保留最新 active 记忆。

    中文注释：
    contradiction_key 表示“这些记忆在同一个问题上可能互相修正”。
    例如：
    - 旧记忆：embedding 默认模型是 A。
    - 新记忆：embedding 默认模型已经改成 B。

    这时不应该把 A 和 B 同时喂给 Planner。
    当前策略是：同 key 只返回最新 active 记录。
    """

    without_key: list[dict[str, Any]] = []
    latest_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("contradiction_key") or "").strip()
        if not key:
            without_key.append(record)
            continue
        current = latest_by_key.get(key)
        if current is None or _record_created_at(record) > _record_created_at(current):
            latest_by_key[key] = record
    return [*without_key, *latest_by_key.values()]


def _scope_matches_state(record: dict[str, Any], state: State) -> bool:
    """判断记忆 scope 是否适合当前任务。

    中文注释：
    scope 只回答“这条记忆在任务语义边界上是否适合”。
    真正的用户/项目/租户权限由 _record_visible_to_context(...) 判断。
    """

    scope = str(record.get("scope", "project"))
    if scope in {"global", "user", "project"}:
        return True
    if scope == "task":
        return str(record.get("task_id", "")) == state["current_task_id"]
    if scope == "tool":
        return str(record.get("tool_name", "")) in {state.get("tool_name", ""), "none"}
    if scope == "file":
        current_task = state["task_tree"].get(state["current_task_id"], {})
        args = current_task.get("args", {}) if isinstance(current_task, dict) else {}
        current_path = str(args.get("path", "")) if isinstance(args, dict) else ""
        return bool(current_path and current_path in record.get("paths", []))
    if scope == "thread":
        return True
    return False


def _record_visible_to_context(record: dict[str, Any], state: State) -> bool:
    """判断当前请求是否有权检索这条 memory。

    中文注释：
    这是 Memory Access Control 的核心硬规则。
    它不使用 LLM，因为权限边界必须稳定、可解释、可审计。
    """

    visibility = str(record.get("visibility", "project"))
    context = _memory_access_context(state)
    identity = _record_acl_identity(record)

    if visibility == "public":
        return True
    if visibility == "tenant":
        return identity["tenant_id"] == context["tenant_id"]
    if visibility == "workspace":
        return (
            identity["tenant_id"] == context["tenant_id"]
            and identity["workspace_id"] == context["workspace_id"]
        )
    if visibility in {"project", "retrieval_only"}:
        return (
            identity["tenant_id"] == context["tenant_id"]
            and identity["workspace_id"] == context["workspace_id"]
            and identity["project_id"] == context["project_id"]
        )
    if visibility == "private":
        return (
            identity["tenant_id"] == context["tenant_id"]
            and identity["workspace_id"] == context["workspace_id"]
            and identity["project_id"] == context["project_id"]
            and identity["user_id"] == context["user_id"]
        )
    return False


def _record_allowed_in_prompt(record: dict[str, Any]) -> bool:
    """判断 memory 是否允许进入 prompt / memory_context。

    中文注释：
    有些记忆可以用于检索和审计，但不应该直接进入 prompt。
    例如：
    - visibility=retrieval_only
    - sensitivity_level=confidential / secret

    这样可以避免敏感经验、隐私内容或只供索引用的记录泄露给模型。
    """

    return memory_prompt_allowed_by_privacy(record)


def _record_access_control(record: dict[str, Any], state: State) -> dict[str, Any]:
    """返回一条 memory 的访问控制判断结果。"""

    visible = _record_visible_to_context(record, state)
    prompt_allowed = visible and _record_allowed_in_prompt(record)
    return {
        "visible": visible,
        "prompt_allowed": prompt_allowed,
        "visibility": str(record.get("visibility", "project")),
        "sensitivity_level": str(record.get("sensitivity_level", "internal")),
        **_record_acl_identity(record),
    }

def _redact_sensitive_text(text: str) -> str:
    """对常见敏感片段做轻量脱敏和截断。

    中文注释：
    记忆系统会跨轮次保存内容，所以不能把 api key、token、password、
    email 这类内容原样写进去。
    这里统一调用 privacy_governance.py：
    - secret 会被替换为 redacted fingerprint。
    - PII 会被替换为 redacted fingerprint。
    - 原始值不会写入 memory metadata。
    """

    redacted = redact_text_for_memory(text)
    if len(redacted) > MAX_MEMORY_TEXT_CHARS:
        return redacted[:MAX_MEMORY_TEXT_CHARS] + "...[TRUNCATED]"
    return redacted


def _safe_memory_value(value: Any, *, key: str = "") -> Any:
    """把要写入 memory metadata 的值裁剪成安全版本。"""

    return redact_value_for_memory(value, key=key)


def _memory_has_success_evidence(pending_memory: dict[str, Any]) -> bool:
    """判断 pending_memory 是否带有可验证成功证据。

    中文注释：
    生产级 memory 不应该只因为模型说“成功了”就长期保存。
    更可信的证据通常来自：
    - tests_passed
    - build_passed
    - verification_passed
    - lint_passed / typecheck_passed
    - human_confirmed / approval
    """

    data = pending_memory.get("tool_result_data")
    if not isinstance(data, dict):
        data = {}
    evidence_keys = {
        "tests_passed",
        "build_passed",
        "verification_passed",
        "lint_passed",
        "typecheck_passed",
        "human_confirmed",
        "approval",
    }
    if any(bool(data.get(key)) for key in evidence_keys):
        return True
    metadata = pending_memory.get("metadata")
    if isinstance(metadata, dict) and any(bool(metadata.get(key)) for key in evidence_keys):
        return True
    return any(bool(pending_memory.get(key)) for key in evidence_keys)


def _metadata_value(metadata: dict[str, Any], key: str, default: str) -> str:
    """从 metadata 读取字符串配置。"""

    value = metadata.get(key)
    if value:
        return str(value)
    return default


def _memory_acl_for_pending(
    state: State,
    pending_memory: dict[str, Any],
    *,
    tool_name: str,
) -> dict[str, Any]:
    """为新写入的 memory 生成 ACL 字段。

    中文注释：
    写入时就把身份和敏感级别固化到 MemoryRecord。
    这样后续检索时不需要猜“这条记忆属于谁、能不能给当前项目用”。
    """

    context = _memory_access_context(state)
    metadata = pending_memory.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    privacy_report = scan_value_for_privacy(pending_memory)
    sensitivity: SensitivityLevel = stronger_sensitivity(
        "internal",
        privacy_report.sensitivity_level,
    )
    if tool_name == "secret_scan" or pending_memory.get("sensitivity_level") == "secret":
        sensitivity = "secret"
    elif pending_memory.get("sensitivity_level") in {"public", "internal", "confidential"}:
        sensitivity = stronger_sensitivity(sensitivity, pending_memory["sensitivity_level"])
    elif metadata.get("sensitivity_level") in {"public", "internal", "confidential", "secret"}:
        sensitivity = stronger_sensitivity(sensitivity, metadata["sensitivity_level"])

    visibility: MemoryVisibility = "project"
    raw_visibility = pending_memory.get("visibility") or metadata.get("visibility")
    if raw_visibility in {
        "private",
        "project",
        "workspace",
        "tenant",
        "public",
        "retrieval_only",
    }:
        visibility = raw_visibility
    if sensitivity in {"confidential", "secret"}:
        visibility = "retrieval_only"

    return {
        "tenant_id": _metadata_value(metadata, "tenant_id", context["tenant_id"]),
        "workspace_id": _metadata_value(
            metadata, "workspace_id", context["workspace_id"]
        ),
        "project_id": _metadata_value(metadata, "project_id", context["project_id"]),
        "user_id": _metadata_value(metadata, "user_id", context["user_id"]),
        "visibility": visibility,
        "sensitivity_level": sensitivity,
        "privacy_governance": privacy_metadata(privacy_report),
    }


def _memory_policy_for_pending(
    state: State,
    pending_memory: dict[str, Any],
    *,
    tool_name: str,
    tool_result_status: str,
) -> MemoryPolicyDecision:
    """决定 pending_memory 是否应该写入长期记忆系统。

    中文注释：
    这是当前版本的 MemoryPolicy：
    - 明显没有信息量的内容丢弃。
    - 成功的代码修改/项目结构经验保留更久。
    - 失败经验默认 ttl，因为它有价值但可能会过期。
    - secret_scan 相关内容不存原始结果，只允许存摘要。
    """

    # Production Memory Policy
    #
    # 中文注释：
    # 当前 MemoryPolicy 已经不只是简单 if/else。
    # 它会和 _govern_memory_record_before_write(...) 配合，形成：
    # - 规则判断。
    # - 证据保留。
    # - 多次成功 promotion。
    # - supersedes / contradiction_key 治理。
    # - store / discard / retrieve 审计。
    #
    # 已实现：
    #
    # 1. Evidence-based Retention 雏形
    #    - 有测试通过 / build 通过 / 人工确认 -> long_term 或 pinned。
    #    - 只是一次临时失败日志 -> ttl。
    #    - 没有可验证证据 -> 降低 importance 或 discard。
    #
    # 2. Frequency-based Promotion
    #    同类成功经验多次出现后，自动提高 importance。
    #    例如同一个修复模式连续 3 次成功，可以晋升为 pinned 候选。
    #
    # 3. Failure Memory Governance 雏形
    #    失败经验有价值，但也容易过期。
    #    当前失败默认 TTL，避免长期污染记忆库。
    #
    # 4. Human-confirmed Memory
    #    governance 层会读取 approval / reviewer / human_confirmed 信号，
    #    并提升 confidence、importance、pinned。
    #
    # 5. Sensitive Memory Policy
    #    安全相关工具不应该保存原始密钥、token、隐私内容。
    #    当前会做轻量脱敏，并把拒绝保存原因写入 audit log。
    #
    # 后续 TODO：
    #
    # 6. LLM Memory Judge
    #    不要让 LLM 直接替代这层硬规则。
    #    更合理的是：规则先过滤明显情况，再让 LLM / reranker
    #    判断“这条记忆是否值得长期保存、是否会误导后续任务”。
    #
    # 7. Enterprise DLP / Privacy Classifier
    #    当前已经有统一 secret scanner / PII detector / policy-based redaction。
    #    后续可以接企业 DLP、数据分级平台、人工审批和跨租户访问审计。
    title = str(pending_memory.get("title", "")).strip()
    reason = str(pending_memory.get("reason", "")).strip()
    preference = preference_metadata_from_pending(pending_memory)
    if preference:
        priority = int(preference.get("priority", 80))
        return MemoryPolicyDecision(
            "store",
            "用户/项目长期偏好需要长期保留。",
            scope="user" if preference.get("scope") == "user" else "project",
            retention_policy="pinned",
            importance=min(1.0, max(0.5, priority / 100)),
            pinned=True,
        )
    if not title and not reason:
        return MemoryPolicyDecision("discard", "pending_memory 缺少 title/reason。")

    if tool_name == "none" and tool_result_status == "none":
        return MemoryPolicyDecision("discard", "没有工具结果，不写入长期记忆。")

    if tool_name == "secret_scan":
        return MemoryPolicyDecision(
            "store",
            "安全扫描结果只保存摘要。",
            retention_policy="ttl",
            importance=0.7,
        )

    if tool_result_status in {"failed", "blocked", "empty"}:
        return MemoryPolicyDecision(
            "store",
            "失败经验可帮助后续恢复，但默认 TTL。",
            retention_policy="ttl",
            importance=0.75,
        )

    if tool_result_status == "success" and _memory_has_success_evidence(pending_memory):
        return MemoryPolicyDecision(
            "store",
            "有测试/build/人工确认等成功证据，长期保留。",
            retention_policy="long_term",
            importance=0.9,
        )

    if tool_name in {"apply_patch", "apply_patch_plan", "rollback", "revert_file_patch"}:
        return MemoryPolicyDecision(
            "store",
            "代码修改经验需要长期保留。",
            retention_policy="long_term",
            importance=0.85,
        )

    if tool_name in {"read_file", "summarize_file", "inspect_symbol", "inspect_import_graph"}:
        return MemoryPolicyDecision(
            "store",
            "项目理解类记忆按项目 scope 保存。",
            scope="project",
            retention_policy="ttl",
            importance=0.6,
        )

    if state.get("done") and tool_result_status == "success":
        return MemoryPolicyDecision(
            "store",
            "完成目标的成功经验可长期保留。",
            retention_policy="long_term",
            importance=0.8,
        )

    return MemoryPolicyDecision("store", "默认写入 TTL 记忆。")

def _stable_memory_id(record: dict[str, Any]) -> str:
    """基于关键字段生成稳定 ID，用于去重。

    中文注释：
    如果同一个 task/tool/status 重复写入，我们希望覆盖旧记录，
    而不是把 memory.jsonl 写成很多重复噪声。
    """

    raw = json.dumps(
        {
            "kind": record.get("kind", "task"),
            "task_id": record.get("task_id", ""),
            "title": record.get("title", ""),
            "tool_name": record.get("tool_name", "none"),
            "status": record.get("status", ""),
            "paths": record.get("paths", []),
            "scope": record.get("scope", "project"),
            "visibility": record.get("visibility", "project"),
            "tenant_id": record.get("tenant_id", DEFAULT_TENANT_ID),
            "workspace_id": record.get("workspace_id", DEFAULT_WORKSPACE_ID),
            "project_id": record.get("project_id", DEFAULT_PROJECT_ID),
            "user_id": record.get("user_id", DEFAULT_USER_ID),
            "contradiction_key": record.get("contradiction_key"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def _records_share_memory_pattern(record: MemoryRecord, existing: dict[str, Any]) -> bool:
    """判断两条记忆是否属于同一类可复用经验。"""

    if str(existing.get("tool_result_status", "")) != "success":
        return False
    if str(existing.get("kind", "")) != record.kind:
        return False
    if str(existing.get("tool_name", "")) != record.tool_name:
        return False
    if (
        record.contradiction_key
        and str(existing.get("contradiction_key") or "") == record.contradiction_key
    ):
        return True
    existing_paths = {str(path) for path in existing.get("paths", [])}
    return bool(existing_paths.intersection(record.paths))

def _extract_paths(memory: dict[str, Any]) -> list[str]:
    """从 pending_memory 里提取相关文件路径。"""

    paths: list[str] = []
    task_args = memory.get("args")
    if isinstance(task_args, dict) and task_args.get("path"):
        paths.append(str(task_args["path"]))
    metadata = memory.get("metadata")
    if isinstance(metadata, dict):
        path = metadata.get("path")
        if path:
            paths.append(str(path))
    artifact_paths = memory.get("artifact_paths")
    if isinstance(artifact_paths, list):
        paths.extend(str(path) for path in artifact_paths)
    return sorted(set(paths))


def _classify_memory_kind(memory: dict[str, Any]) -> MemoryKind:
    """根据任务结果粗略分类记忆类型。"""

    preference = preference_metadata_from_pending(memory)
    if preference:
        return "user" if preference.get("scope") == "user" else "project"
    decision = str(memory.get("decision", ""))
    tool_result_status = str(memory.get("tool_result_status", "none"))
    tool_name = str(memory.get("tool_name", "none"))
    if decision == "fail" or tool_result_status in ("failed", "blocked", "empty"):
        return "failure"
    if tool_name in ("apply_patch", "apply_patch_plan", "rollback", "revert_file_patch"):
        return "patch"
    if tool_name.startswith("run_") or tool_name in ("static_check", "lint_typecheck", "run_tests"):
        return "eval"
    if tool_name in ("inspect_symbol", "inspect_import_graph", "summarize_file", "read_file"):
        return "project"
    return "task"


def _memory_tags(memory: dict[str, Any]) -> list[str]:
    """生成便于检索的标签。"""

    tags = {
        str(memory.get("decision", "")),
        str(memory.get("tool_result_status", "")),
        str(memory.get("tool_name", "")),
    }
    tags.update(
        Path(path).suffix.lstrip(".")
        for path in _extract_paths(memory)
        if Path(path).suffix
    )
    return sorted(tag for tag in tags if tag and tag != "none")


def _build_memory_record(state: State, pending_memory: dict[str, Any]) -> MemoryRecord:
    """把 Task Committer 产出的 pending_memory 标准化成 MemoryRecord。"""

    task_id = str(pending_memory.get("task_id", state["current_task_id"]))
    task = dict(state["task_tree"].get(task_id, {}))
    tool_result_data = pending_memory.get("tool_result_data")
    if not isinstance(tool_result_data, dict):
        tool_result_data = {}
    tool_name = str(task.get("tool") or state["tool_name"] or "none")
    changed_files = [
        str(path)
        for path in tool_result_data.get("changed_files", [])
        if isinstance(tool_result_data.get("changed_files", []), list)
    ]
    paths = sorted(
        set(_extract_paths({**pending_memory, "args": task.get("args", {})}) + changed_files)
    )
    status = str(task.get("status") or pending_memory.get("decision") or "unknown")
    tool_result_status = str(
        pending_memory.get("tool_result_status")
        or task.get("tool_result_status")
        or state["tool_result_status"]
        or "none"
    )
    policy = _memory_policy_for_pending(
        state,
        pending_memory,
        tool_name=tool_name,
        tool_result_status=tool_result_status,
    )
    acl = _memory_acl_for_pending(state, pending_memory, tool_name=tool_name)
    preference = preference_metadata_from_pending(pending_memory)
    if preference:
        acl["visibility"] = _preference_visibility(str(preference.get("scope", "project")))
    privacy_report = scan_value_for_privacy(pending_memory)
    summary = _redact_sensitive_text(
        str(preference.get("value", ""))
        if preference
        else (
            f"{pending_memory.get('title', task.get('title', ''))} | "
            f"decision={pending_memory.get('decision', 'none')} | "
            f"reason={pending_memory.get('reason', '')}"
        )
    )[:800]
    safe_title, safe_summary = storage_summary_for_sensitive_memory(
        title=str(pending_memory.get("title") or task.get("title", "")),
        summary=summary,
        report=privacy_report,
    )
    summary = safe_summary
    raw_record = {
        "kind": _classify_memory_kind({**pending_memory, "tool_name": tool_name}),
        "task_id": task_id,
        "title": (
            f"Preference: {preference['key']}"
            if preference
            else safe_title
        ),
        "tool_name": tool_name,
        "status": status,
        "paths": paths,
        "scope": policy.scope,
        "visibility": acl["visibility"],
        "tenant_id": acl["tenant_id"],
        "workspace_id": acl["workspace_id"],
        "project_id": acl["project_id"],
        "user_id": acl["user_id"],
        "contradiction_key": (
            f"preference:{preference.get('scope', 'project')}:{preference['key']}"
            if preference
            else pending_memory.get("contradiction_key")
        ),
    }
    record_id = _stable_memory_id(raw_record)
    confidence = 0.9 if tool_result_status == "success" else 0.65
    if policy.pinned:
        confidence = max(confidence, 0.95)
    return MemoryRecord(
        id=record_id,
        kind=raw_record["kind"],
        task_id=task_id,
        title=_redact_sensitive_text(raw_record["title"])[:200],
        summary=summary,
        status=status,
        tool_name=tool_name,
        tool_result_status=tool_result_status,
        paths=paths,
        tags=_memory_tags({**pending_memory, "tool_name": tool_name}),
        confidence=confidence,
        importance=policy.importance,
        scope=policy.scope,
        visibility=acl["visibility"],
        sensitivity_level=acl["sensitivity_level"],
        tenant_id=acl["tenant_id"],
        workspace_id=acl["workspace_id"],
        project_id=acl["project_id"],
        user_id=acl["user_id"],
        retention_policy=policy.retention_policy,
        validity_status=policy.validity_status,
        pinned=policy.pinned or policy.retention_policy == "pinned",
        expires_at=policy.expires_at or _expires_at_for_policy(policy.retention_policy),
        supersedes=str(pending_memory.get("supersedes") or "") or None,
        contradiction_key=(
            str(pending_memory.get("contradiction_key") or "") or None
        ),
        metadata={
            "memory_policy": {
                "action": policy.action,
                "reason": policy.reason,
            },
            "run_id": state["run_id"],
            "memory_access_control": _safe_memory_value(acl),
            "privacy_governance": privacy_metadata(privacy_report),
            "preference_memory": _safe_memory_value(preference) if preference else {},
            "parent_evaluation": _safe_memory_value(
                pending_memory.get("parent_evaluation", {})
            ),
            "goal_progress": _safe_memory_value(pending_memory.get("goal_progress", {})),
            "tool_result_data": _safe_memory_value(tool_result_data),
            "source_memory": _safe_memory_value(pending_memory),
        },
    )
