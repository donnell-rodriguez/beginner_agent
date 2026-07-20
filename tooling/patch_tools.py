from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from difflib import unified_diff
from hashlib import sha256
from typing import Any

from .core import (
    MAX_PATCH_CHARS,
    PATCH_PLAN_DIR,
    ensure_state_dirs,
    json_dumps,
    project_text_files,
    safe_resolve,
    safe_text_file,
)
from .check_tools import lint_typecheck_tool


# 中文注释：
# patch_tools.py 放“会修改文件”的工具。
#
# 这些工具风险比只读工具高，所以 Policy 层默认要求 Human Approval。
# 当前实现仍然很保守：
#   - 只能改 beginner_agent 内部文本文件。
#   - apply_patch 只能做 old_text -> new_text 的精确替换。
#   - old_text 必须只出现一次。
#
def apply_patch_tool(path: str, old_text: str, new_text: str) -> str:
    """对单个文件做精确字符串替换。"""

    # 中文注释：
    # safe_text_file 会做路径安全检查和文本文件检查。
    file_path = safe_text_file(path)
    current = file_path.read_text(encoding="utf-8", errors="replace")
    if not old_text:
        return "apply_patch 需要 old_text。"
    if old_text == new_text:
        return "apply_patch 的 old_text 和 new_text 不能相同。"
    occurrences = current.count(old_text)

    # 中文注释：
    # old_text 出现 0 次：说明模型给错了修改位置。
    # old_text 出现多次：说明修改不够精确，可能误改多个地方。
    if occurrences == 0:
        return "apply_patch 失败：old_text 在目标文件中不存在。"
    if occurrences > 1:
        return "apply_patch 失败：old_text 出现多次，修改不够精确。"
    if len(old_text) + len(new_text) > MAX_PATCH_CHARS:
        return "apply_patch 失败：单次 patch 内容过大。"
    file_path.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")
    return f"apply_patch 成功：已修改 {path}。"


def rollback_tool(path: str, content: str) -> str:
    """把文件恢复到给定内容。"""

    # 中文注释：
    # rollback 通常由 Executor/Evaluator 根据 patch_history 调用。
    # content 是之前保存的 before_content。
    file_path = safe_text_file(path)
    if not content:
        return "rollback 需要 content。"
    file_path.write_text(content, encoding="utf-8")
    return f"rollback 成功：已恢复 {path}。"


def format_apply_tool(path: str) -> str:
    """教学版格式化：去掉行尾空白并确保文件以换行结尾。"""

    # 中文注释：
    # 这不是 black/ruff 格式化。
    # 它只是做最简单、最安全的文本格式清理。
    file_path = safe_text_file(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    formatted = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    if formatted == text:
        return f"format_apply：{path} 已经符合教学版格式规则。"
    file_path.write_text(formatted, encoding="utf-8")
    return f"format_apply 成功：已格式化 {path}。"


def _patch_plan_path(plan_id: str):
    """把 patch_plan_id 映射到 .agent_state/patch_plans 里的 JSON 文件。"""

    ensure_state_dirs()

    # 中文注释：
    # 防止 plan_id 里带奇怪字符，统一替换成安全文件名。
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", plan_id)
    return PATCH_PLAN_DIR / f"{safe_id}.json"


def _file_hash_text(text: str) -> str:
    """计算文本内容 hash，用来判断文件是否在验证后被别人改过。"""

    return sha256(text.encode("utf-8")).hexdigest()


def _read_patch_plan(patch_plan_id: str) -> dict[str, Any]:
    """读取 PatchPlan JSON。"""

    plan_path = _patch_plan_path(patch_plan_id)
    if not plan_path.exists():
        raise ValueError(f"PatchPlan 不存在：{patch_plan_id}")
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"PatchPlan 格式不正确：{patch_plan_id}")
    return data


def read_patch_plan_metadata(patch_plan_id: str) -> dict[str, Any]:
    """给 Executor / Registry 读取 PatchPlan 关键信息。

    中文注释：
    这不是一个对 LLM 暴露的工具，而是内部辅助函数。
    Executor 需要知道 apply_patch_plan 最终会改哪个 path，
    才能提前拍快照、记录 diff、写入 patch_history。
    """

    plan = _read_patch_plan(patch_plan_id)
    return {
        "id": str(plan.get("id", patch_plan_id)),
        "path": str(plan.get("path", "")),
        "goal": str(plan.get("goal", "")),
        "validated": bool(plan.get("validated", False)),
        "validated_file_hash": str(plan.get("validated_file_hash", "")),
    }


def patch_plan_tool(path: str, goal: str, old_text: str = "", new_text: str = "") -> str:
    """生成并保存 PatchPlan。"""

    # 中文注释：
    # PatchPlan 是“先计划修改，再验证，再审批，再执行”的基础。
    # 比直接 apply_patch 更接近真实 code agent 的做法。
    file_path = safe_text_file(path)
    current = file_path.read_text(encoding="utf-8", errors="replace")
    plan_id = f"patch-{uuid.uuid4().hex[:8]}"
    plan = {
        "id": plan_id,
        "path": path,
        "goal": goal,
        "old_text": old_text,
        "new_text": new_text,
        "risk_level": "medium" if old_text and new_text else "draft",
        "created_file_hash": _file_hash_text(current),
        "validated": False,
        "validated_at": "",
        "validated_file_hash": "",
        "validation_issues": [],
        "preview": "",
    }
    _patch_plan_path(plan_id).write_text(json_dumps(plan), encoding="utf-8")
    return json_dumps(plan)


def validate_patch_plan_tool(patch_plan_id: str) -> str:
    """验证 PatchPlan 是否可应用。"""

    # 中文注释：
    # validate 只检查，不修改文件。
    # 它会确认 old_text 是否存在且唯一、patch 是否过大。
    plan_path = _patch_plan_path(patch_plan_id)
    if not plan_path.exists():
        return f"PatchPlan 不存在：{patch_plan_id}"
    plan = _read_patch_plan(patch_plan_id)
    path = str(plan.get("path", ""))
    old_text = str(plan.get("old_text", ""))
    new_text = str(plan.get("new_text", ""))
    file_path = safe_text_file(path)
    current = file_path.read_text(encoding="utf-8", errors="replace")
    issues: list[str] = []
    if not old_text:
        issues.append("old_text 为空。")
    if old_text == new_text:
        issues.append("old_text 和 new_text 相同。")
    if current.count(old_text) != 1:
        issues.append("old_text 在目标文件中不是恰好出现一次。")
    if len(old_text) + len(new_text) > MAX_PATCH_CHARS:
        issues.append("patch 内容过大。")
    preview = ""
    if not issues:
        preview = preview_patch_tool(path, old_text, new_text)
    plan.update(
        {
            "validated": not issues,
            "validated_at": datetime.now(timezone.utc).isoformat() if not issues else "",
            "validated_file_hash": _file_hash_text(current) if not issues else "",
            "validation_issues": issues,
            "preview": preview,
        }
    )
    plan_path.write_text(json_dumps(plan), encoding="utf-8")
    return json_dumps(
        {
            "patch_plan_id": patch_plan_id,
            "valid": not issues,
            "issues": issues,
            "validated_file_hash": plan["validated_file_hash"],
            "preview": preview,
        }
    )


def preview_patch_tool(path: str, old_text: str, new_text: str) -> str:
    """只预览 diff，不修改文件。

    中文注释：
    生产级 code agent 在真正改文件前，通常会先生成 preview。
    这样 Planner / Policy / Human Approval 可以看到“准备怎么改”。
    """

    file_path = safe_text_file(path)
    current = file_path.read_text(encoding="utf-8", errors="replace")
    if not old_text:
        return "preview_patch 需要 old_text。"
    if current.count(old_text) != 1:
        return "preview_patch 失败：old_text 在目标文件中不是恰好出现一次。"
    updated = current.replace(old_text, new_text, 1)
    diff = unified_diff(
        current.splitlines(),
        updated.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff) or "preview_patch：没有 diff。"


def apply_patch_dry_run_tool(path: str, old_text: str, new_text: str) -> str:
    """干跑 patch：验证能不能改，但不真正写文件。"""

    file_path = safe_text_file(path)
    current = file_path.read_text(encoding="utf-8", errors="replace")
    issues: list[str] = []
    if not old_text:
        issues.append("old_text 为空。")
    if old_text == new_text:
        issues.append("old_text 和 new_text 相同。")
    if current.count(old_text) != 1:
        issues.append("old_text 在目标文件中不是恰好出现一次。")
    if len(old_text) + len(new_text) > MAX_PATCH_CHARS:
        issues.append("patch 内容过大。")
    return json_dumps(
        {
            "path": path,
            "can_apply": not issues,
            "issues": issues,
            "preview": preview_patch_tool(path, old_text, new_text) if not issues else "",
        }
    )


def validate_patch_scope_tool(path: str, goal: str = "") -> str:
    """检查修改范围是否在 beginner_agent 内，并给出风险提示。"""

    file_path = safe_text_file(path)
    warnings: list[str] = []
    if file_path.name in ("state.py", "graph.py", "planner.py", "executor.py", "policy.py"):
        warnings.append("这是 agent 核心流程文件，修改后必须运行测试。")
    if file_path.suffix != ".py":
        warnings.append("目标不是 Python 文件，静态检查可能无法覆盖语义问题。")
    if goal and any(word in goal for word in ("删除", "清空", "密钥", "密码", "token")):
        warnings.append("修改目标包含高风险词，需要人工审批。")
    return json_dumps({"path": path, "allowed": True, "risk_notes": warnings})


def revert_file_patch_tool(path: str, content: str) -> str:
    """回滚指定文件内容。

    中文注释：
    这个工具和 rollback_tool 类似。
    单独保留名字，是为了让 Planner 能表达“回滚某个文件 patch”这个意图。
    """

    return rollback_tool(path, content)


def apply_patch_plan_tool(patch_plan_id: str) -> str:
    """应用已经验证的 PatchPlan。"""

    # 中文注释：
    # 大厂式修改治理要求：
    #   计划必须先 validate，并把 validated 状态写回 PatchPlan。
    #   执行时再次确认当前文件 hash 没变，避免验证后文件被别人改过。
    plan = _read_patch_plan(patch_plan_id)
    if not plan.get("validated"):
        return "apply_patch_plan 失败：PatchPlan 尚未通过 validate_patch_plan。"
    path = str(plan.get("path", ""))
    file_path = safe_text_file(path)
    current = file_path.read_text(encoding="utf-8", errors="replace")
    validated_hash = str(plan.get("validated_file_hash", ""))
    if not validated_hash or _file_hash_text(current) != validated_hash:
        return "apply_patch_plan 失败：目标文件在验证后发生变化，请重新 validate_patch_plan。"
    validation = json.loads(validate_patch_plan_tool(patch_plan_id))
    if not validation.get("valid"):
        return "apply_patch_plan 失败：\n" + json_dumps(validation)
    plan = _read_patch_plan(patch_plan_id)
    return apply_patch_tool(str(plan["path"]), str(plan["old_text"]), str(plan["new_text"]))


def format_check_tool() -> str:
    """复用 lint_typecheck 作为教学版格式检查。"""

    return lint_typecheck_tool()


def validate_edit_path(path: str) -> tuple[bool, str]:
    """检查写入目标是否是项目内允许文本文件。"""

    # 中文注释：
    # 这个函数给未来更复杂的 Policy 使用。
    # 当前 registry 主要直接调用 safe_text_file。
    try:
        safe_text_file(path)
    except ValueError as exc:
        return False, str(exc)
    return True, "写入路径安全。"


def known_patch_plans() -> list[str]:
    ensure_state_dirs()
    return [path.stem for path in sorted(PATCH_PLAN_DIR.glob("*.json"))]
