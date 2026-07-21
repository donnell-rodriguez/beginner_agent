from __future__ import annotations

from typing import Any

from .tools import READ_ONLY_TOOLS, WRITE_TOOLS


def create_initial_state(user_input: str) -> dict[str, Any]:
    """创建 beginner_agent 每次运行的初始 State。

    中文注释：
    main.py 和 cli.py 都需要一份完整的初始 State。
    如果两边各写一份，后续新增字段时很容易忘记同步。

    所以这里把“初始状态工厂”单独拆出来：

        用户输入
          -> create_initial_state(...)
          -> graph.invoke(...) / graph.stream(...)

    这样图怎么运行、CLI 怎么交互、State 有哪些默认字段，
    三件事不会互相耦合在一起。
    """

    return {
        "user_input": user_input,
        "task_type": "chat",
        "risk_level": "low",
        "needs_tool": False,
        "route_reason": "",
        "next_action": "schedule",
        "draft": "",
        "final_answer": "",
        "tool_name": "none",
        "tool_args": {},
        "tool_result": "",
        "tool_result_data": {},
        "tool_result_status": "none",
        "execution_monitor_status": "ok",
        "execution_monitor_reason": "",
        "recovery_action": "none",
        "recovery_reason": "",
        "partial_result": "",
        "resume_hint": "",
        "parent_evaluation": {},
        "goal_progress": {},
        "memory_notes": [],
        "memory_context": {},
        "pending_memory": {},
        "checkpoint_report": {},
        "sandbox_report": {},
        "async_job_report": {},
        "artifact_report": {},
        "observability_report": {},
        "root_task_id": "root",
        "task_tree": {},
        "agenda": [],
        "current_task_id": "",
        "completed_tasks": [],
        "patch_history": [],
        "execution_status": "not_started",
        "active_execution": {},
        "execution_attempts": [],
        "max_tool_duration_ms": 30000,
        "human_approvals": {},
        "pending_approval": {},
        "planner_reason": "",
        "plan_validation_status": "none",
        "plan_validation_reason": "",
        "policy_decision": "deny",
        "policy_reason": "",
        "evaluation_decision": "none",
        "evaluation_reason": "",
        "done": False,
        "step_count": 0,
        "max_steps": 12,
        "max_depth": 2,
        "max_total_tasks": 10,
        "max_task_retries": 1,
        "allowed_tools": [*READ_ONLY_TOOLS, *WRITE_TOOLS],
        "permission_policy": {
            **{tool_name: "allow" for tool_name in READ_ONLY_TOOLS},
            # 中文注释：
            # 写工具默认 ask。
            # 真实执行前会进入 Tool Policy，再进入 Approval Interrupt。
            **{tool_name: "ask" for tool_name in WRITE_TOOLS},
        },
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ],
    }
