from __future__ import annotations

import json
from typing import Any, Literal

from .llm_client import chat_completion
from .node_utils import (
    PROJECT_TEXT_FILES,
    PROJECT_TOOLS,
    fallback_subtasks,
    json_loads_from_model,
    new_task,
)
from .state import State


PlannerRoute = Literal["validate", "select_tool"]


def _normalize_tool_args(tool: str, args: dict[str, Any], state: State) -> dict[str, Any] | None:
    """把 LLM 输出的工具参数收敛成安全、稳定的格式。

    中文注释：
    LLM 输出的 JSON 不能直接相信。
    Planner 这里先做第一轮清洗，后面 Plan Validator / Policy 还会继续检查。
    """

    if tool == "list_files":
        return {"path": "."}

    if tool == "list_tree":
        return {"path": ".", "max_depth": int(args.get("max_depth", 2) or 2)}

    if tool == "read_file":
        path = str(args.get("path", ""))
        if path in PROJECT_TEXT_FILES:
            return {"path": path}
        return None

    if tool == "read_file_slice":
        path = str(args.get("path", ""))
        if path in PROJECT_TEXT_FILES:
            return {
                "path": path,
                "start": int(args.get("start", 1) or 1),
                "end": int(args.get("end", 120) or 120),
            }
        return None

    if tool == "search_code":
        query = str(args.get("query") or state["user_input"]).strip()
        if len(query) >= 2:
            return {"query": query[:80]}
        return None

    if tool == "grep_regex":
        pattern = str(args.get("pattern") or args.get("query") or "").strip()
        if len(pattern) >= 2:
            return {"pattern": pattern[:120], "path": str(args.get("path", "."))}
        return None

    if tool == "inspect_symbol":
        symbol = str(args.get("symbol", "")).strip()
        if len(symbol) >= 2:
            return {"symbol": symbol[:80]}
        return None

    if tool == "inspect_references":
        symbol = str(args.get("symbol", "")).strip()
        if len(symbol) >= 2:
            return {"symbol": symbol[:80]}
        return None

    if tool == "inspect_call_graph":
        return {"function": str(args.get("function", ""))[:80]}

    if tool == "query_project_index":
        query = str(args.get("query") or state["user_input"]).strip()
        if len(query) >= 2:
            return {"query": query[:80]}
        return None

    if tool == "inspect_function_signature":
        symbol = str(args.get("symbol", "")).strip()
        if len(symbol) >= 2:
            return {"symbol": symbol[:80]}
        return None

    if tool == "inspect_class_hierarchy":
        class_name = str(args.get("class_name", "")).strip()
        if len(class_name) >= 2:
            return {"class_name": class_name[:80]}
        return None

    if tool in (
        "inspect_import_graph",
        "build_project_index",
        "detect_project_stack",
        "detect_test_framework",
        "detect_package_manager",
        "detect_build_system",
        "list_allowed_commands",
        "list_tool_catalog",
        "tool_policy_report",
        "list_project_roots",
        "get_active_project",
        "run_ruff",
        "run_ruff_format_check",
        "run_mypy",
        "run_uv_import_graph",
        "run_cargo_check",
        "run_cargo_test",
        "run_cargo_clippy",
        "run_cargo_fmt_check",
        "static_check",
        "lint_typecheck",
        "run_tests",
        "run_typecheck",
        "run_build",
        "get_diagnostics",
        "format_check",
        "git_status",
        "git_diff",
        "dependency_inspect",
    ):
        return {}

    if tool == "run_allowed_command":
        profile = str(args.get("profile", ""))
        if profile in (
            "python_compileall",
            "python_compile_state",
            "uv_import_graph",
            "ruff_check",
            "ruff_format_check",
            "mypy_beginner_agent",
            "pytest_beginner_agent",
            "cargo_check",
            "cargo_test",
            "cargo_clippy",
            "cargo_fmt_check",
        ):
            return {"profile": profile}
        return None

    if tool == "run_package_script":
        script = str(args.get("script", ""))
        if script in (
            "test",
            "lint",
            "format_check",
            "typecheck",
            "build",
            "cargo_check",
            "cargo_test",
            "cargo_clippy",
            "cargo_fmt_check",
        ):
            return {"script": script}
        return None

    if tool == "run_pytest":
        return {"target": str(args.get("target", "beginner_agent"))}

    if tool == "describe_tool":
        tool_name = str(args.get("tool_name", ""))
        if tool_name in PROJECT_TOOLS:
            return {"tool_name": tool_name}
        return None

    if tool == "register_project_root":
        project_id = str(args.get("project_id", ""))
        path = str(args.get("path", ""))
        if project_id and path:
            return {"project_id": project_id, "path": path}
        return None

    if tool == "set_active_project":
        project_id = str(args.get("project_id", ""))
        if project_id:
            return {"project_id": project_id}
        return None

    if tool in ("detect_rust_project", "inspect_rust_symbols"):
        return {"path": str(args.get("path", "."))}

    if tool == "inspect_rust_references":
        symbol = str(args.get("symbol", "")).strip()
        if len(symbol) >= 2:
            return {"symbol": symbol[:80], "path": str(args.get("path", "."))}
        return None

    if tool in ("parse_rust_errors", "parse_cargo_test_failure"):
        output = str(args.get("output", ""))
        if output:
            return {"output": output[:5000]}
        return None

    if tool == "map_changed_rust_files_to_tests":
        changed_files = args.get("changed_files", [])
        if isinstance(changed_files, str):
            changed_files = [item.strip() for item in changed_files.split(",") if item.strip()]
        if isinstance(changed_files, list):
            return {"changed_files": [str(item) for item in changed_files if str(item).endswith(".rs")]}
        return {"changed_files": []}

    if tool == "safe_path_exists":
        return {"path": str(args.get("path", "."))}

    if tool == "run_targeted_tests":
        return {"target": str(args.get("target", "beginner_agent"))}

    if tool == "parse_test_failure":
        output = str(args.get("test_output") or args.get("output") or "")
        if output:
            return {"test_output": output[:4000]}
        return None

    if tool in ("extract_stack_trace", "classify_failure"):
        output = str(args.get("output") or args.get("test_output") or "")
        if output:
            return {"output": output[:4000]}
        return None

    if tool == "compare_failure_before_after":
        before = str(args.get("before", ""))
        after = str(args.get("after", ""))
        if before and after:
            return {"before": before[:4000], "after": after[:4000]}
        return None

    if tool in ("git_diff_file", "format_apply", "summarize_file", "validate_patch_scope"):
        path = str(args.get("path", ""))
        if path in PROJECT_TEXT_FILES:
            normalized = {"path": path}
            if tool == "validate_patch_scope":
                normalized["goal"] = str(args.get("goal") or state["user_input"])
            return normalized
        return None

    if tool in ("map_changed_files_to_tests", "run_impacted_tests"):
        changed_files = args.get("changed_files", [])
        if isinstance(changed_files, str):
            changed_files = [item.strip() for item in changed_files.split(",") if item.strip()]
        if isinstance(changed_files, list):
            return {"changed_files": [str(item) for item in changed_files if str(item) in PROJECT_TEXT_FILES]}
        return {"changed_files": []}

    if tool == "select_relevant_tests":
        changed_files = args.get("changed_files", [])
        if isinstance(changed_files, str):
            changed_files = [item.strip() for item in changed_files.split(",") if item.strip()]
        if not isinstance(changed_files, list):
            changed_files = []
        return {
            "query": str(args.get("query") or state["user_input"])[:80],
            "changed_files": [str(item) for item in changed_files if str(item) in PROJECT_TEXT_FILES],
        }

    if tool == "secret_scan":
        return {"path": str(args.get("path", "."))}

    if tool == "patch_plan":
        path = str(args.get("path", ""))
        if path in PROJECT_TEXT_FILES:
            return {
                "path": path,
                "goal": str(args.get("goal") or state["user_input"]),
                "old_text": str(args.get("old_text", "")),
                "new_text": str(args.get("new_text", "")),
            }
        return None

    if tool in ("validate_patch_plan", "apply_patch_plan"):
        patch_plan_id = str(args.get("patch_plan_id", ""))
        if patch_plan_id:
            return {"patch_plan_id": patch_plan_id}
        return None

    if tool in ("preview_patch", "apply_patch_dry_run"):
        path = str(args.get("path", ""))
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        if path in PROJECT_TEXT_FILES and old_text and new_text and old_text != new_text:
            return {"path": path, "old_text": old_text, "new_text": new_text}
        return None

    if tool == "checkpoint_save":
        return {"name": str(args.get("name", "checkpoint")), "data": args.get("data", "")}

    if tool == "checkpoint_load":
        return {"name": str(args.get("name", "checkpoint"))}

    if tool == "apply_patch":
        path = str(args.get("path", ""))
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        if path in PROJECT_TEXT_FILES and old_text and new_text and old_text != new_text:
            return {"path": path, "old_text": old_text, "new_text": new_text}
        return None

    if tool == "rollback":
        # 中文注释：
        # rollback 通常不应该由 Planner 直接创造。
        # 它应该由 Evaluator 在发现测试失败后，根据 patch_history 生成。
        return None

    if tool == "revert_file_patch":
        return None

    if tool == "audit_tool_call":
        tool_name = str(args.get("tool_name", ""))
        if tool_name:
            return {
                "tool_name": tool_name,
                "tool_args": args.get("tool_args", {}) if isinstance(args.get("tool_args", {}), dict) else {},
                "decision": str(args.get("decision", "record")),
            }
        return None

    if tool == "audit_patch":
        path = str(args.get("path", ""))
        if path in PROJECT_TEXT_FILES:
            return {
                "path": path,
                "reason": str(args.get("reason") or state["user_input"]),
                "patch_plan_id": str(args.get("patch_plan_id", "")),
            }
        return None

    if tool == "read_audit_log":
        return {"limit": int(args.get("limit", 20) or 20)}

    return None


def _normalize_subtasks(data: Any, parent: dict[str, Any], state: State) -> list[dict[str, Any]]:
    """清洗 LLM 生成的子任务。

    中文注释：
    生产级 Planner 不只是“让模型吐 JSON”。
    它还要做任务建模、去重、工具约束和参数规整。
    这里是教学版实现：保留简单结构，但把工具参数先收敛到安全格式。
    """

    if not isinstance(data, dict):
        return []
    raw_tasks = data.get("subtasks", [])
    if not isinstance(raw_tasks, list):
        return []

    parent_id = str(parent["id"])
    depth = int(parent.get("depth", 0)) + 1
    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(raw_tasks[:4]):
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", "none"))
        if tool not in state["allowed_tools"] or tool not in PROJECT_TOOLS:
            continue
        args = item.get("args", {})
        if not isinstance(args, dict):
            args = {}
        normalized_args = _normalize_tool_args(tool, args, state)
        if normalized_args is None:
            continue
        normalized.append(
            new_task(
                str(item.get("id") or f"{parent_id}.{index + 1}"),
                str(item.get("title") or f"执行 {tool}"),
                parent_id=parent_id,
                depth=depth,
                tool=tool,
                args=normalized_args,
                reason=str(item.get("reason") or "补充观察信息。"),
            )
        )

    return normalized


def _llm_plan(task: dict[str, Any], state: State) -> tuple[str, str, list[dict[str, Any]]]:
    """Planner 让 LLM 判断 expand 还是 execute，并在 expand 时生成子任务。"""

    response = chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你是 Planner / Decomposer。"
                    "你的职责是判断当前任务是否需要拆解。"
                    "如果当前任务没有绑定具体工具，通常 action=expand。"
                    "如果当前任务已经绑定具体工具，通常 action=execute。"
                    "如果 action=expand，请生成 1 到 4 个子任务。"
                    "当前只支持 Python 和 Rust 两种语言；其他语言不要生成专用工具任务。"
                    f"只能使用这些工具：{', '.join(PROJECT_TOOLS)}。"
                    'list_files 的 args 必须是 {"path":"."}。'
                    'list_tree 的 args 是 {"path":".","max_depth":2}。'
                    f"read_file 只能读取这些文件：{', '.join(PROJECT_TEXT_FILES)}。"
                    'read_file_slice 的 args 是 {"path":"...","start":1,"end":120}。'
                    'search_code 的 args 是 {"query":"关键词"}。'
                    'grep_regex 的 args 是 {"pattern":"正则","path":"."}。'
                    'inspect_symbol 的 args 是 {"symbol":"函数或类名"}。'
                    'inspect_references 的 args 是 {"symbol":"函数或类名"}。'
                    'query_project_index 的 args 是 {"query":"关键词"}。'
                    'inspect_function_signature 的 args 是 {"symbol":"函数名"}。'
                    'inspect_class_hierarchy 的 args 是 {"class_name":"类名"}。'
                    "detect_project_stack、detect_test_framework、detect_package_manager、detect_build_system、list_allowed_commands、list_tool_catalog、tool_policy_report、list_project_roots、get_active_project 的 args 必须是空对象。"
                    'describe_tool 的 args 是 {"tool_name":"工具名"}。'
                    'register_project_root 的 args 是 {"project_id":"短名称","path":"/Users/christophermanning/Downloads/aios/项目目录"}，这是控制类写工具。'
                    'set_active_project 的 args 是 {"project_id":"已注册项目名"}，这是控制类写工具。'
                    'run_allowed_command 的 args 是 {"profile":"python_compileall|uv_import_graph|ruff_check|ruff_format_check|mypy_beginner_agent|pytest_beginner_agent|cargo_check|cargo_test|cargo_clippy|cargo_fmt_check"}，只能使用白名单 profile。'
                    'run_package_script 的 args 是 {"script":"test|lint|format_check|typecheck|build|cargo_check|cargo_test|cargo_clippy|cargo_fmt_check"}。'
                    'run_pytest 的 args 是 {"target":"beginner_agent"}。'
                    "run_ruff、run_ruff_format_check、run_mypy、run_uv_import_graph 的 args 必须是空对象。"
                    "run_cargo_check、run_cargo_test、run_cargo_clippy、run_cargo_fmt_check 的 args 必须是空对象。"
                    'detect_rust_project、inspect_rust_symbols 的 args 是 {"path":"."}。'
                    'inspect_rust_references 的 args 是 {"symbol":"符号名","path":"."}。'
                    'parse_rust_errors、parse_cargo_test_failure 的 args 是 {"output":"cargo 输出"}。'
                    'map_changed_rust_files_to_tests 的 args 是 {"changed_files":["src/lib.rs"]}。'
                    'safe_path_exists 的 args 是 {"path":"."} 或 {"path":"文件路径"}。'
                    "inspect_import_graph、build_project_index、static_check、lint_typecheck、run_tests、run_typecheck、run_build、get_diagnostics、format_check、git_status、git_diff、dependency_inspect 的 args 必须是空对象。"
                    'map_changed_files_to_tests、run_impacted_tests 的 args 是 {"changed_files":["文件路径"]}。'
                    'select_relevant_tests 的 args 是 {"query":"目标","changed_files":["文件路径"]}。'
                    'git_diff_file、summarize_file、format_apply 的 args 是 {"path":"..."}。'
                    'preview_patch、apply_patch_dry_run 的 args 必须包含 path、old_text、new_text。'
                    'validate_patch_scope 的 args 是 {"path":"...","goal":"修改目标"}。'
                    "apply_patch 只能在用户明确要求修改代码时使用，args 必须包含 path、old_text、new_text。"
                    "更推荐先生成 patch_plan，再 validate_patch_plan，最后 apply_patch_plan。"
                    "不要主动生成 rollback，rollback 由 Evaluator 在失败恢复时安排。"
                    "优先生成能帮助理解代码结构的任务，避免重复读取同一个文件。"
                    "只返回严格 JSON，不要解释。"
                    '格式：{"action":"expand|execute","reason":"一句话原因",'
                    '"subtasks":[{"id":"root.1","title":"...",'
                    f'"tool":"{"|".join(PROJECT_TOOLS)}",'
                    '"args":{"path":"..."},"reason":"..."}]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户目标：{state['user_input']}\n\n"
                    f"当前任务：{json.dumps(task, ensure_ascii=False)}\n\n"
                    f"任务树：{json.dumps(state['task_tree'], ensure_ascii=False)}\n\n"
                    f"已完成任务：{json.dumps(state['completed_tasks'], ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0,
        max_tokens=900,
    )
    data = json_loads_from_model(response)
    if not isinstance(data, dict):
        raise ValueError("Planner 返回的不是 JSON object。")
    action = str(data.get("action", "")).lower().strip()
    reason = str(data.get("reason") or "Planner 未提供原因。")
    subtasks = _normalize_subtasks(data, task, state)
    if action not in ("expand", "execute"):
        raise ValueError("Planner action 不合法。")
    return action, reason, subtasks


def planner_decomposer_node(state: State) -> dict[str, Any]:
    """2A. Planner / Decomposer：只判断当前任务是否要拆解。

    中文注释：
    这个节点现在只负责一件事：

        大任务 -> 拆成 children

    它不再负责把 tool_name / tool_args 写入 State。
    叶子任务的工具选择交给后面的 tool_selector_node。
    这样拆开后，Planner 的职责更接近生产级 agent 里的 Decomposer。
    """

    task_tree = dict(state["task_tree"])
    agenda = list(state["agenda"])
    task_id = state["current_task_id"]
    task = dict(task_tree[task_id])

    if task.get("children"):
        return {
            "next_action": "validate",
            "planner_reason": "Planner：当前任务已经拆过，回到 Scheduler。",
            "messages": [{"role": "assistant", "content": f"Planner：任务 {task_id} 已经有 children。"}],
        }

    can_expand = (
        task.get("depth", 0) < state["max_depth"]
        and len(task_tree) < state["max_total_tasks"]
        and task.get("tool", "none") == "none"
    )

    if can_expand:
        try:
            action, reason, subtasks = _llm_plan(task, {**state, "task_tree": task_tree})
        except (RuntimeError, ValueError, json.JSONDecodeError):
            action = "expand"
            reason = "Planner 兜底：LLM 失败，使用保底子任务。"
            subtasks = fallback_subtasks(task, {**state, "task_tree": task_tree})

        if action == "expand" and subtasks:
            existing_titles = {str(item.get("title", "")) for item in task_tree.values()}
            subtasks = [subtask for subtask in subtasks if subtask["title"] not in existing_titles]
        if action == "expand" and subtasks:
            child_ids = [subtask["id"] for subtask in subtasks]
            task["status"] = "expanded"
            task["children"] = child_ids
            task_tree[task_id] = task
            for subtask in subtasks:
                task_tree[subtask["id"]] = subtask
            agenda = child_ids + agenda
            return {
                "task_tree": task_tree,
                "agenda": agenda,
                "current_task_id": task_id,
                "tool_name": "none",
                "tool_args": {},
                "planner_reason": reason,
                "next_action": "validate",
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"Planner：拆解任务 {task_id} -> {', '.join(child_ids)}。原因：{reason}",
                    }
                ],
            }

    # 中文注释：
    # 走到这里说明当前任务没有被拆成子任务。
    # 它可能本来就是叶子任务，也可能 LLM 判断不需要继续拆。
    # Planner 只把任务状态标记为 planned，具体选择哪个工具交给 tool_selector_node。
    task["status"] = "planned"
    task_tree[task_id] = task
    return {
        "task_tree": task_tree,
        "tool_name": "none",
        "tool_args": {},
        "planner_reason": "Planner：任务已经足够具体，进入 Tool Selector。",
        "next_action": "validate",
        "messages": [
            {
                "role": "assistant",
                "content": f"Planner：任务 {task_id} 不再继续拆解，进入 Tool Selector。",
            }
        ],
    }


def tool_selector_node(state: State) -> dict[str, Any]:
    """2B. Tool Selector：把叶子任务绑定到具体工具和参数。

    中文注释：
    Planner 负责“任务要不要拆”。
    Tool Selector 负责“这个叶子任务用哪个工具、带什么参数”。

    目前工具主要来自 task_tree 里已经保存的 task["tool"] / task["args"]。
    如果以后要做更强的 agent，可以在这个节点里加入：
    - LLM 重新选择工具。
    - 多工具候选排序。
    - 根据历史失败换工具。
    - 根据语言 Python / Rust 选择不同 adapter。
    """

    task_tree = dict(state["task_tree"])
    task_id = state["current_task_id"]
    task = dict(task_tree.get(task_id, {}))
    tool_name = str(task.get("tool", "none"))
    tool_args = task.get("args", {})
    if not isinstance(tool_args, dict):
        tool_args = {}

    task["status"] = "planned"
    task_tree[task_id] = task

    return {
        "task_tree": task_tree,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "planner_reason": f"Tool Selector：任务 {task_id} 选择工具 {tool_name}。",
        "next_action": "validate",
        "messages": [
            {
                "role": "assistant",
                "content": f"Tool Selector：任务 {task_id} 准备使用工具 {tool_name}。",
            }
        ],
    }


def route_after_planner(state: State) -> PlannerRoute:
    """Planner 后的路由。

    中文注释：
    - 如果当前任务已经有 children，说明刚刚完成拆解，去 Plan Validator 检查子任务结构。
    - 如果没有 children，说明它是叶子任务，先去 Tool Selector 绑定工具。
    """

    task_id = state["current_task_id"]
    task = state["task_tree"].get(task_id, {})
    if task.get("children"):
        return "validate"
    return "select_tool"
