from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal


PreferenceScope = Literal["user", "project", "workspace", "global"]
PreferenceCategory = Literal[
    "language",
    "configuration",
    "architecture",
    "verification",
    "quality",
    "persistence",
    "tooling",
]


@dataclass(frozen=True)
class PreferenceMemory:
    """PreferenceMemory：用户/项目长期偏好。

    中文注释：
    这类记忆不是“某次工具执行结果”，
    而是 agent 后续做事时应该遵守的习惯。
    例如：
    - 用户喜欢中文注释。
    - 配置应该放进 .env / .env.example。
    - 修改后必须测试。
    - 优先按大厂工程风格设计。

    把这些做成 memory 的好处是：
    - 不需要每次都靠 prompt 重新提醒。
    - 可以区分 user / project / workspace 作用域。
    - 可以被审计、覆盖、失效，而不是散落在代码注释里。
    """

    key: str
    value: str
    scope: PreferenceScope
    category: PreferenceCategory
    priority: int
    rationale: str
    applies_to: tuple[str, ...]
    source: str = "default_preference_seed"

    def as_dict(self) -> dict[str, Any]:
        """转成可保存到 MemoryRecord.metadata 的普通 dict。"""

        return {
            "key": self.key,
            "value": self.value,
            "scope": self.scope,
            "category": self.category,
            "priority": self.priority,
            "rationale": self.rationale,
            "applies_to": list(self.applies_to),
            "source": self.source,
        }


DEFAULT_PREFERENCES: tuple[PreferenceMemory, ...] = (
    PreferenceMemory(
        key="explanation_language",
        value=(
            "优先使用中文解释，并在关键代码处添加小白能理解的中文注释。"
        ),
        scope="user",
        category="language",
        priority=90,
        rationale="用户正在通过阅读源码和注释学习 agent 工程。",
        applies_to=("docs", "comments", "final_answer", "code_review"),
    ),
    PreferenceMemory(
        key="configuration_location",
        value=(
            "环境相关配置必须放在 .env / .env.example，"
            "不要硬编码到 Python 源码。"
        ),
        scope="project",
        category="configuration",
        priority=95,
        rationale="保持部署配置和业务代码分离，符合生产级工程习惯。",
        applies_to=("config", "database", "model", "runtime", "network"),
    ),
    PreferenceMemory(
        key="low_coupling",
        value=(
            "修改前先判断是否需要拆分文件，"
            "避免把不相关逻辑堆进一个模块。"
        ),
        scope="project",
        category="architecture",
        priority=90,
        rationale="项目正在逐步从教学 demo 走向可维护 code agent。",
        applies_to=("architecture", "modules", "nodes", "tools", "memory"),
    ),
    PreferenceMemory(
        key="verify_after_change",
        value="代码修改完成后必须运行相关测试或最接近的可执行验证。",
        scope="project",
        category="verification",
        priority=100,
        rationale="code agent 必须形成修改、验证、反馈的闭环。",
        applies_to=("code_change", "tools", "memory", "graph", "policy"),
    ),
    PreferenceMemory(
        key="production_style",
        value=(
            "设计时优先对齐大厂风格：分层、治理、审计、安全、可恢复。"
        ),
        scope="project",
        category="quality",
        priority=88,
        rationale="用户目标是逐步接近生产级 code agent，而不是停留在原型。",
        applies_to=("planner", "policy", "executor", "memory", "checkpoint", "tools"),
    ),
    PreferenceMemory(
        key="supported_languages",
        value="当前优先支持 Python 和 Rust，其他语言后续再扩展。",
        scope="project",
        category="tooling",
        priority=86,
        rationale="工具平台已经围绕 Python/Rust adapter 设计。",
        applies_to=("tools", "runtime", "test", "build", "diagnostics"),
    ),
    PreferenceMemory(
        key="remove_stale_code",
        value=(
            "功能升级时要删除旧逻辑、旧注释和无用分支，"
            "避免新旧代码混杂。"
        ),
        scope="project",
        category="architecture",
        priority=87,
        rationale="长期 agent 项目最怕冗余逻辑导致行为不可解释。",
        applies_to=("refactor", "cleanup", "graph", "nodes", "tools"),
    ),
)


def stable_preference_id(preference: PreferenceMemory, identity: dict[str, str]) -> str:
    """生成稳定 preference id。

    中文注释：
    同一个 user/project 下，同一个 key 应该覆盖旧值，而不是重复写多条。
    """

    raw = "|".join(
        [
            preference.scope,
            preference.key,
            identity.get("tenant_id", ""),
            identity.get("workspace_id", ""),
            identity.get("project_id", ""),
            identity.get("user_id", ""),
        ]
    )
    return f"pref-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def default_preference_payloads(identity: dict[str, str]) -> list[dict[str, Any]]:
    """生成默认偏好 payload，供 memory.py 转成 MemoryRecord。"""

    payloads: list[dict[str, Any]] = []
    for preference in DEFAULT_PREFERENCES:
        payloads.append(
            {
                "id": stable_preference_id(preference, identity),
                "preference": preference.as_dict(),
                "identity": identity,
            }
        )
    return payloads


def preference_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """从 MemoryRecord dict 中提取 preference_memory。"""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    preference = metadata.get("preference_memory")
    if not isinstance(preference, dict):
        return None
    if not preference.get("key") or not preference.get("value"):
        return None
    return preference


def is_preference_record(record: dict[str, Any]) -> bool:
    """判断一条记忆是否是 user/project preference。"""

    return preference_from_record(record) is not None


def preference_rerank_signal(record: dict[str, Any]) -> dict[str, Any]:
    """给 MemoryReranker 使用的偏好信号。"""

    preference = preference_from_record(record)
    if not preference:
        return {"has_preference": False, "preference_weight": 0.0}
    try:
        priority = int(preference.get("priority", 50))
    except (TypeError, ValueError):
        priority = 50
    scope = str(preference.get("scope", "project"))
    scope_bonus = 0.15 if scope in {"user", "project"} else 0.05
    weight = min(1.0, max(0.0, (priority / 100) + scope_bonus))
    return {
        "has_preference": True,
        "preference_weight": round(weight, 4),
        "preference_key": preference.get("key", ""),
        "preference_scope": scope,
        "preference_category": preference.get("category", ""),
    }


def merged_preference_context(
    default_payloads: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """合并默认偏好和持久化偏好。

    中文注释：
    合并规则：
    - 默认偏好提供基础行为。
    - 持久化记忆中的同 key 偏好可以覆盖默认值。
    - priority 高的排在前面，方便 Planner/Policy 优先读取。
    """

    merged: dict[str, dict[str, Any]] = {}
    for payload in default_payloads:
        preference = payload["preference"]
        merged[str(preference["key"])] = {
            **preference,
            "memory_id": payload["id"],
            "source": preference.get("source", "default_preference_seed"),
        }
    for record in records:
        preference = preference_from_record(record)
        if not preference:
            continue
        key = str(preference["key"])
        merged[key] = {
            **preference,
            "memory_id": record.get("id", ""),
            "source": preference.get("source", record.get("source", "memory")),
        }
    preferences = sorted(
        merged.values(),
        key=lambda item: int(item.get("priority", 50)),
        reverse=True,
    )
    return {
        "source": "default preference seed + persisted preference memory",
        "count": len(preferences),
        "preferences": preferences,
        "keys": [str(item.get("key", "")) for item in preferences],
    }


def preference_metadata_from_pending(
    pending_memory: dict[str, Any],
) -> dict[str, Any] | None:
    """从 pending_memory 中识别用户明确表达的新偏好。

    中文注释：
    当前实现先支持结构化入口：

        pending_memory["metadata"]["preference_memory"] = {...}

    后续如果要更智能，可以新增 LLM Preference Extractor，
    从自然语言中提取“以后都要...”这类偏好。
    """

    metadata = pending_memory.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    preference = metadata.get("preference_memory") or pending_memory.get(
        "preference_memory"
    )
    if not isinstance(preference, dict):
        return None
    if not preference.get("key") or not preference.get("value"):
        return None
    return {
        "key": str(preference["key"]),
        "value": str(preference["value"]),
        "scope": str(preference.get("scope", "user")),
        "category": str(preference.get("category", "quality")),
        "priority": int(preference.get("priority", 80)),
        "rationale": str(preference.get("rationale", "用户明确表达的长期偏好。")),
        "applies_to": [
            str(item)
            for item in preference.get("applies_to", ["agent_behavior"])
            if item
        ],
        "source": "pending_memory",
    }
