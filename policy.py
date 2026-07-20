from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .state import State
from .tooling.audit_tools import audit_tool_call_tool
from .tooling.command_tools import ALLOWED_COMMANDS
from .tooling.core import json_dumps
from .tooling.registry import TOOL_SPECS, ToolSpec, validate_tool_request


PolicyRoute = Literal["approval", "execute", "evaluate"]
PolicyAction = Literal["allow", "ask", "deny"]
PolicyRule = Callable[["PolicyContext", "PolicyDecision"], None]


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Tool Policy 的只读上下文。

    中文注释：
    生产级 policy 不应该在每条规则里到处读 state。
    所以这里先把 policy 需要的信息整理成一个上下文对象。
    规则只读这个对象，然后写入 PolicyDecision。
    """

    task_id: str
    task: dict[str, Any]
    tool_name: str
    tool_args: dict[str, Any]
    tool_spec: ToolSpec | None
    allowed_tools: set[str]
    configured_policy: str
    human_approved: bool
    task_risk_level: str
    patch_history: list[Any]


@dataclass(slots=True)
class PolicyDecision:
    """结构化权限决策。

    中文注释：
    旧版本只返回 decision/reason 字符串。
    生产级系统更需要知道：
    - 哪些规则触发了。
    - 工具元数据是什么。
    - 为什么需要审批。
    - 待审批对象是什么。
    - 审计记录应该写什么。
    """

    action: PolicyAction = "allow"
    reason: str = "工具通过策略检查。"
    triggered_rules: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    validation_reason: str = ""
    pending_approval: dict[str, Any] = field(default_factory=dict)

    def apply(self, action: PolicyAction, rule_name: str, reason: str) -> None:
        """应用一条规则结果。

        中文注释：
        权限等级从宽到严是：

            allow < ask < deny

        deny 最严格，一旦出现 deny，就不能被后面的 ask/allow 覆盖。
        """

        self.triggered_rules.append(rule_name)
        self.risk_notes.append(reason)
        if self.action == "deny":
            return
        if action == "deny" or (action == "ask" and self.action == "allow"):
            self.action = action
            self.reason = reason


def _approval_id(task_id: str, tool_name: str) -> str:
    """生成稳定可读的审批 ID。"""

    safe_task = task_id.replace(".", "-")
    return f"approval-{safe_task}-{tool_name}"


def _tool_metadata(tool_spec: ToolSpec | None) -> dict[str, Any]:
    """返回精简工具元数据，便于写入审批和审计。"""

    if tool_spec is None:
        return {}
    return {
        "name": tool_spec.name,
        "access": tool_spec.access,
        "risk": tool_spec.risk,
        "category": tool_spec.category,
        "language": tool_spec.language,
        "requires_approval": tool_spec.requires_approval,
        "args_model": tool_spec.args_schema.__name__,
    }


def _path_like_arg(tool_args: dict[str, Any]) -> str:
    """从参数里取最常见的 path 字段。"""

    return str(tool_args.get("path", ""))


def _patch_size(tool_args: dict[str, Any]) -> int:
    """估算 patch 文本规模。"""

    return len(str(tool_args.get("old_text", ""))) + len(str(tool_args.get("new_text", "")))


def _tool_args_size(tool_args: dict[str, Any]) -> int:
    """估算工具参数 JSON 体积。"""

    return len(json_dumps(tool_args))


def rule_known_tool(context: PolicyContext, decision: PolicyDecision) -> None:
    """拒绝 registry 里不存在的工具。"""

    if context.tool_spec is None:
        decision.apply("deny", "known_tool", f"工具 {context.tool_name} 不在 TOOL_SPECS 注册表中。")


def rule_tool_selected(context: PolicyContext, decision: PolicyDecision) -> None:
    """拒绝空工具。

    中文注释：
    正常情况下，能进入 Tool Policy 的任务应该已经由 Planner 选好了工具。
    如果 tool_name 还是 none，说明 Planner / Scheduler 状态不完整。
    """

    if context.tool_name == "none":
        decision.apply("deny", "tool_selected", "当前任务没有选择可执行工具。")


def rule_allowed_tools(context: PolicyContext, decision: PolicyDecision) -> None:
    """拒绝不在当前运行白名单里的工具。"""

    if context.tool_name not in context.allowed_tools:
        decision.apply("deny", "allowed_tools", f"工具 {context.tool_name} 不在 allowed_tools 白名单中。")


def rule_configured_permission(context: PolicyContext, decision: PolicyDecision) -> None:
    """执行用户/系统配置的 allow / ask / deny。"""

    configured = context.configured_policy
    if configured == "deny":
        decision.apply("deny", "configured_permission", f"工具 {context.tool_name} 的权限策略是 deny。")
    elif configured == "ask" and not context.human_approved:
        decision.apply("ask", "configured_permission", f"工具 {context.tool_name} 的权限策略是 ask。")
    elif configured != "allow":
        decision.apply("deny", "configured_permission", f"工具 {context.tool_name} 的权限策略非法：{configured}。")


def rule_validate_args(context: PolicyContext, decision: PolicyDecision) -> None:
    """用 Pydantic schema + 工具 validator 校验参数。"""

    if decision.action == "deny":
        return
    is_valid, reason = validate_tool_request(context.tool_name, context.tool_args)
    decision.validation_reason = reason
    if not is_valid:
        decision.apply("deny", "validate_args", f"工具参数不安全或不可执行：{reason}")


def rule_tool_args_size(context: PolicyContext, decision: PolicyDecision) -> None:
    """限制工具参数体积。

    中文注释：
    生产级 agent 要防止 LLM 把超大内容塞进 tool_args。
    大文本应该放在文件、patch plan 或 checkpoint 中，而不是直接塞进一次工具调用。
    """

    size = _tool_args_size(context.tool_args)
    if size > 20_000:
        decision.apply("deny", "tool_args_size", f"工具参数过大：{size} 字符，拒绝执行。")
    elif size > 8_000 and not context.human_approved:
        decision.apply("ask", "tool_args_size", f"工具参数较大：{size} 字符，需要人工确认。")


def rule_tool_risk(context: PolicyContext, decision: PolicyDecision) -> None:
    """根据 ToolSpec.risk 做风险升级。"""

    if context.tool_spec is None or decision.action == "deny":
        return
    if context.tool_spec.risk == "high" and not context.human_approved:
        decision.apply("ask", "tool_risk", f"高风险工具 {context.tool_name} 需要人工审批。")
    elif context.tool_spec.risk == "medium" and context.task_risk_level == "high" and not context.human_approved:
        decision.apply("ask", "tool_risk", f"高风险任务调用中风险工具 {context.tool_name}，需要人工审批。")


def rule_write_requires_approval(context: PolicyContext, decision: PolicyDecision) -> None:
    """所有写工具默认需要人工审批。"""

    if context.tool_spec is None or decision.action == "deny":
        return
    if context.tool_spec.access == "write" and not context.human_approved:
        decision.apply("ask", "write_requires_approval", f"写工具 {context.tool_name} 需要人工审批。")


def rule_platform_write_requires_approval(context: PolicyContext, decision: PolicyDecision) -> None:
    """平台管理类写操作必须审批。

    中文注释：
    register_project_root / set_active_project 这类工具不一定改代码，
    但会改变 agent 后续能操作的项目范围，所以属于平台级高风险变更。
    """

    if context.tool_spec is None or decision.action == "deny":
        return
    if context.tool_spec.category == "platform" and context.tool_spec.access == "write" and not context.human_approved:
        decision.apply("ask", "platform_write_requires_approval", f"平台管理工具 {context.tool_name} 需要人工审批。")


def rule_sensitive_path(context: PolicyContext, decision: PolicyDecision) -> None:
    """对敏感路径做升级或拒绝。

    中文注释：
    这里不是路径越界检查；路径越界已经由 validate_tool_request 处理。
    这里处理的是“虽然在项目内，但属于敏感文件”的情况。
    """

    path = _path_like_arg(context.tool_args)
    if not path or decision.action == "deny":
        return
    sensitive_names = (".env", "secret", "token", "credential", "key", "password")
    if any(name in path.lower() for name in sensitive_names):
        decision.apply("deny", "sensitive_path", f"工具参数涉及敏感路径：{path}。")
    elif path.endswith(("pyproject.toml", "Cargo.toml", "Cargo.lock", "uv.lock")) and not context.human_approved:
        decision.apply("ask", "sensitive_path", f"工具将访问或修改关键配置/锁文件：{path}。")


def rule_generated_or_state_path(context: PolicyContext, decision: PolicyDecision) -> None:
    """限制 agent 内部状态目录和生成目录的访问。"""

    path = _path_like_arg(context.tool_args)
    if not path or decision.action == "deny":
        return
    internal_markers = (".agent_state", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
    if any(marker in path for marker in internal_markers):
        decision.apply("deny", "generated_or_state_path", f"工具不应直接操作内部状态或缓存路径：{path}。")


def rule_config_write_requires_approval(context: PolicyContext, decision: PolicyDecision) -> None:
    """写入关键配置文件需要审批。"""

    if context.tool_spec is None or context.tool_spec.access != "write" or decision.action == "deny":
        return
    path = _path_like_arg(context.tool_args)
    protected_suffixes = ("pyproject.toml", "Cargo.toml", "Cargo.lock", "uv.lock", "README.md")
    if path.endswith(protected_suffixes) and not context.human_approved:
        decision.apply("ask", "config_write_requires_approval", f"写入关键项目文件 {path} 需要人工审批。")


def rule_patch_size(context: PolicyContext, decision: PolicyDecision) -> None:
    """大 patch 需要人工审批。"""

    if context.tool_name not in {"apply_patch", "preview_patch", "apply_patch_dry_run", "patch_plan"}:
        return
    size = _patch_size(context.tool_args)
    if size > 4_000 and not context.human_approved:
        decision.apply("ask", "patch_size", f"patch 文本规模较大：{size} 字符，需要人工审批。")


def rule_direct_patch_requires_plan(context: PolicyContext, decision: PolicyDecision) -> None:
    """直接 apply_patch 需要更高门槛。

    中文注释：
    更生产级的 code agent 通常优先走：

        patch_plan -> validate_patch_plan -> preview/dry_run -> apply_patch_plan

    直接 apply_patch 不是绝对禁止，但没有人工审批时必须停下来。
    """

    if context.tool_name != "apply_patch" or decision.action == "deny":
        return
    if not context.human_approved:
        decision.apply("ask", "direct_patch_requires_plan", "直接 apply_patch 需要人工审批；优先使用 patch_plan / validate_patch_plan / apply_patch_plan。")


def rule_rollback_requires_history(context: PolicyContext, decision: PolicyDecision) -> None:
    """rollback 必须有 patch_history 支撑。"""

    if context.tool_name != "rollback" or decision.action == "deny":
        return
    if not context.patch_history:
        decision.apply("deny", "rollback_requires_history", "rollback 需要已有 patch_history，当前没有可回滚记录。")


def rule_command_profile_risk(context: PolicyContext, decision: PolicyDecision) -> None:
    """根据命令 profile 做风险升级。"""

    if decision.action == "deny":
        return
    profile = ""
    if context.tool_name == "run_allowed_command":
        profile = str(context.tool_args.get("profile", ""))
    elif context.tool_name == "run_package_script":
        script = str(context.tool_args.get("script", ""))
        profile = {
            "test": "pytest_beginner_agent",
            "lint": "ruff_check",
            "format_check": "ruff_format_check",
            "typecheck": "mypy_beginner_agent",
            "build": "uv_import_graph",
            "cargo_check": "cargo_check",
            "cargo_test": "cargo_test",
            "cargo_clippy": "cargo_clippy",
            "cargo_fmt_check": "cargo_fmt_check",
        }.get(script, "")
    if not profile:
        return
    command_spec = ALLOWED_COMMANDS.get(profile, {})
    if command_spec.get("writes_cache") and not context.human_approved:
        decision.apply("ask", "command_profile_risk", f"命令 profile {profile} 会写缓存或构建产物，需要人工确认。")
    if command_spec.get("cwd") == "active_project" and context.task_risk_level == "high" and not context.human_approved:
        decision.apply("ask", "command_profile_risk", f"命令 profile {profile} 会在 active project 中运行，高风险任务需要审批。")


def rule_high_risk_task(context: PolicyContext, decision: PolicyDecision) -> None:
    """Router 判断高风险时，对写工具和命令类工具升级为 ask。"""

    if decision.action == "deny" or context.tool_spec is None:
        return
    command_like = context.tool_spec.category == "verify" and context.tool_name.startswith("run_")
    if context.task_risk_level == "high" and (context.tool_spec.access == "write" or command_like) and not context.human_approved:
        decision.apply("ask", "high_risk_task", "Router 判断任务高风险，需要人工确认后才能继续。")


POLICY_RULES: tuple[PolicyRule, ...] = (
    rule_tool_selected,
    rule_known_tool,
    rule_allowed_tools,
    rule_configured_permission,
    rule_validate_args,
    rule_tool_args_size,
    rule_tool_risk,
    rule_write_requires_approval,
    rule_platform_write_requires_approval,
    rule_sensitive_path,
    rule_generated_or_state_path,
    rule_config_write_requires_approval,
    rule_patch_size,
    rule_direct_patch_requires_plan,
    rule_rollback_requires_history,
    rule_command_profile_risk,
    rule_high_risk_task,
)


def _make_context(state: State) -> PolicyContext:
    """从 LangGraph State 构造 PolicyContext。"""

    task_tree = dict(state["task_tree"])
    task_id = state["current_task_id"]
    tool_name = state["tool_name"]
    return PolicyContext(
        task_id=task_id,
        task=dict(task_tree.get(task_id, {})),
        tool_name=tool_name,
        tool_args=dict(state["tool_args"]),
        tool_spec=TOOL_SPECS.get(tool_name),
        allowed_tools=set(state["allowed_tools"]),
        configured_policy=str(state["permission_policy"].get(tool_name, "deny")),
        human_approved=bool(state["human_approvals"].get(task_id, False)),
        task_risk_level=str(state["risk_level"]),
        patch_history=list(state.get("patch_history", [])),
    )


def evaluate_policy(state: State) -> PolicyDecision:
    """执行完整 policy 规则链。"""

    context = _make_context(state)
    decision = PolicyDecision()
    for rule in POLICY_RULES:
        rule(context, decision)
    if decision.action == "ask":
        decision.pending_approval = {
            "approval_id": _approval_id(context.task_id, context.tool_name),
            "task_id": context.task_id,
            "tool_name": context.tool_name,
            "tool_args": context.tool_args,
            "tool_metadata": _tool_metadata(context.tool_spec),
            "risk_level": context.task_risk_level,
            "reason": decision.reason,
            "triggered_rules": list(decision.triggered_rules),
            "risk_notes": list(decision.risk_notes),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    return decision


def _audit_policy_decision(context: PolicyContext, decision: PolicyDecision) -> None:
    """写入工具调用审计记录。

    中文注释：
    这里直接调用 audit_tool_call_tool，而不是通过 run_tool。
    因为 policy 本身就是工具执行前的安全层，不能为了审计再绕回工具执行层。
    """

    audit_tool_call_tool(
        context.tool_name,
        {
            "task_id": context.task_id,
            "tool_args": context.tool_args,
            "tool_metadata": _tool_metadata(context.tool_spec),
            "policy": {
                "decision": decision.action,
                "reason": decision.reason,
                "validation_reason": decision.validation_reason,
                "triggered_rules": decision.triggered_rules,
                "risk_notes": decision.risk_notes,
            },
        },
        decision.action,
    )


def tool_policy_node(state: State) -> dict[str, Any]:
    """4. Tool Policy / Permission Layer：判断工具能不能用、要不要确认。"""

    task_tree = dict(state["task_tree"])
    context = _make_context(state)
    decision = evaluate_policy(state)
    _audit_policy_decision(context, decision)

    task = dict(context.task)
    if decision.action == "ask":
        task["status"] = "waiting_approval"
        task["result"] = decision.reason
        task["policy"] = {
            "decision": decision.action,
            "reason": decision.reason,
            "triggered_rules": decision.triggered_rules,
            "risk_notes": decision.risk_notes,
        }
        task_tree[context.task_id] = task
        return {
            "task_tree": task_tree,
            "policy_decision": decision.action,
            "policy_reason": decision.reason,
            "pending_approval": decision.pending_approval,
            "next_action": "approval",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Tool Policy：需要人工审批，原因：{decision.reason}",
                }
            ],
        }

    if decision.action == "deny":
        task["status"] = "failed"
        task["result"] = decision.reason
        task["tool_result_status"] = "blocked"
        task["policy"] = {
            "decision": decision.action,
            "reason": decision.reason,
            "triggered_rules": decision.triggered_rules,
            "risk_notes": decision.risk_notes,
        }
        task_tree[context.task_id] = task
        return {
            "task_tree": task_tree,
            "policy_decision": decision.action,
            "policy_reason": decision.reason,
            "pending_approval": {},
            "tool_result": decision.reason,
            "tool_result_status": "blocked",
            "next_action": "evaluate",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Tool Policy：拒绝工具调用，原因：{decision.reason}",
                }
            ],
        }

    task["status"] = "approved"
    task["policy"] = {
        "decision": "allow",
        "reason": decision.reason,
        "triggered_rules": decision.triggered_rules,
        "risk_notes": decision.risk_notes,
    }
    task_tree[context.task_id] = task
    return {
        "task_tree": task_tree,
        "policy_decision": "allow",
        "policy_reason": decision.reason,
        "pending_approval": {},
        "next_action": "execute",
        "messages": [
            {
                "role": "assistant",
                "content": f"Tool Policy：允许任务 {context.task_id} 使用工具 {context.tool_name}。",
            }
        ],
    }


def route_after_policy(state: State) -> PolicyRoute:
    """Tool Policy 后的路由。"""

    if state["next_action"] == "approval":
        return "approval"
    if state["next_action"] == "execute":
        return "execute"
    return "evaluate"
