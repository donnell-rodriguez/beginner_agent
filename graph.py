from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .checkpointing import build_checkpointer
from .nodes import (
    chat_node,
    execution_monitor_node,
    evaluator_verifier_node,
    executor_node,
    human_approval_node,
    memory_retriever_node,
    memory_writer_node,
    plan_validator_node,
    planner_decomposer_node,
    recovery_planner_node,
    route_after_evaluator,
    route_after_execution_monitor,
    route_after_human_approval,
    route_after_memory_writer,
    route_after_plan_validator,
    route_after_planner,
    route_after_policy,
    route_after_recovery_planner,
    route_after_scheduler,
    route_after_task_committer,
    route_by_task,
    router_classifier_node,
    scheduler_node,
    search_node,
    summarize_node,
    task_committer_node,
    tool_policy_node,
    tool_selector_node,
    write_node,
)
from .state import State


# 中文注释：
# graph.py 只负责“编排节点和边”。
#
# 具体智能逻辑不要堆在这里：
# - router.py 负责任务分类和风险判断。
# - memory.py 负责轻量记忆读取和写入。
# - scheduler.py 负责 agenda / task_tree 调度。
# - planner.py 负责拆解任务，tool_selector_node 负责选择工具。
# - plan_validator.py 负责计划质量和工具参数可执行性检查。
# - policy.py 负责 Tool Policy / Permission Layer。
# - approval.py 负责 Human Approval。
# - executor.py 负责真正执行工具。
# - execution_monitor.py 负责观察执行是否失败、超预算、空结果或部分结果。
# - recovery.py 负责决定重试、换方案、重新拆解、人工确认或停止总结。
# - evaluator.py 负责结果验证，task_committer_node 负责重试、继续拆解、失败恢复和 rollback 安排。
#
# 这样 graph.py 保持稳定，其他模块可以独立升级。


def build_graph():
    """组装 beginner_agent 的 LangGraph 主流程。

    中文注释：
    当前图已经不是单纯教学版的“线性流程”，而是一个分层 code agent 主干：

        START
          -> Router / Classifier
          -> 简单任务：search / write / chat
          -> 复杂任务：
                Memory Retriever
                Scheduler
                -> Planner / Decomposer
                -> Tool Selector
                -> Plan Validator
                -> Tool Policy
                -> Human Approval（只在需要审批时）
                -> Executor
                -> Execution Monitor / Watchdog
                -> Recovery Planner（只在失败、超预算、空结果、部分结果时）
                -> Evaluator / Verifier
                -> Task Committer
                -> Memory Writer
                -> Scheduler 或 Summarize
          -> END

    注意：
    - patch planning 现在由 Planner + patch_plan/validate_patch_plan 工具承担。
    - approval gate 现在由 tool_policy_node + human_approval_node 承担。
    - test runner 现在由 run_tests / run_targeted_tests / run_impacted_tests 工具承担。
    - failure analyzer 现在由 evaluator_verifier_node 承担。
    - retry / rollback / memory commit 现在由 task_committer_node 承担。
    - memory retriever/writer 已经是独立节点。
    - checkpoint 仍然属于 LangGraph runtime 层，由 compile(checkpointer=...) 配置。
    """

    builder = StateGraph(State)

    # 1. Router / Classifier
    # 判断用户任务是 search / write / chat / agent，并给出 risk_level。
    builder.add_node("router_classifier", router_classifier_node)

    # 2. 简单任务节点。
    # 这些任务不进入复杂 agent loop，执行后直接汇总。
    builder.add_node("search", search_node)
    builder.add_node("write", write_node)
    builder.add_node("chat", chat_node)

    # 3. 复杂 agent loop 节点。
    # 这是 code agent 的核心闭环。
    builder.add_node("memory_retriever", memory_retriever_node)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("planner_decomposer", planner_decomposer_node)
    builder.add_node("tool_selector", tool_selector_node)
    builder.add_node("plan_validator", plan_validator_node)
    builder.add_node("tool_policy", tool_policy_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("executor", executor_node)
    builder.add_node("execution_monitor", execution_monitor_node)
    builder.add_node("recovery_planner", recovery_planner_node)
    builder.add_node("evaluator_verifier", evaluator_verifier_node)
    builder.add_node("task_committer", task_committer_node)
    builder.add_node("memory_writer", memory_writer_node)

    # 4. 最终汇总节点。
    builder.add_node("summarize", summarize_node)

    # 入口：所有任务先进入 Router。
    builder.add_edge(START, "router_classifier")

    # Router 根据 task_type 分流。
    builder.add_conditional_edges(
        "router_classifier",
        route_by_task,
        {
            "search": "search",
            "write": "write",
            "chat": "chat",
            "agent": "memory_retriever",
        },
    )

    # Memory Retriever
    # 复杂任务开始前先读取轻量记忆，后续 Planner/Evaluator 可以参考。
    builder.add_edge("memory_retriever", "scheduler")

    # 简单任务完成后进入统一汇总。
    builder.add_edge("search", "summarize")
    builder.add_edge("write", "summarize")
    builder.add_edge("chat", "summarize")

    # Scheduler / Agenda Manager
    # 选择下一个 pending task；没有任务或超出 max_steps 时结束。
    builder.add_conditional_edges(
        "scheduler",
        route_after_scheduler,
        {
            "plan": "planner_decomposer",
            "finish": "summarize",
        },
    )

    # Planner / Decomposer
    # 只决定当前任务是否继续拆解。
    builder.add_conditional_edges(
        "planner_decomposer",
        route_after_planner,
        {
            "validate": "plan_validator",
            "select_tool": "tool_selector",
        },
    )

    # Tool Selector
    # 叶子任务在这里绑定 tool_name / tool_args，然后再进入计划验证。
    builder.add_edge("tool_selector", "plan_validator")

    # Plan Validator
    # 检查子任务结构、工具是否存在、Pydantic 参数 schema 和安全 validator。
    builder.add_conditional_edges(
        "plan_validator",
        route_after_plan_validator,
        {
            "schedule": "scheduler",
            "policy": "tool_policy",
            "evaluate": "evaluator_verifier",
        },
    )

    # Tool Policy / Permission Layer
    # 规则引擎判断 allow / ask / deny。
    # ask 和 deny 都不会直接执行工具，而是进入 Evaluator 处理 blocked 状态。
    builder.add_conditional_edges(
        "tool_policy",
        route_after_policy,
        {
            "approval": "human_approval",
            "execute": "executor",
            "evaluate": "evaluator_verifier",
        },
    )

    # Human Approval
    # 需要审批的工具调用在这里触发 LangGraph interrupt。
    # CLI / UI 收到审批请求后，用 Command(resume=...) 恢复图执行。
    builder.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        {
            "execute": "executor",
            "evaluate": "evaluator_verifier",
        },
    )

    # Executor
    # 真正执行工具。写工具执行前后会记录 patch_history，供 rollback 使用。
    builder.add_edge("executor", "execution_monitor")

    # Execution Monitor / Watchdog
    # 先判断执行是否超预算、失败、空结果或部分结果。
    builder.add_conditional_edges(
        "execution_monitor",
        route_after_execution_monitor,
        {
            "evaluate": "evaluator_verifier",
            "recover": "recovery_planner",
        },
    )

    # Recovery Planner
    # 如果执行不理想，先决定恢复策略，再交给 Evaluator/Committer 落地。
    builder.add_conditional_edges(
        "recovery_planner",
        route_after_recovery_planner,
        {
            "evaluate": "evaluator_verifier",
        },
    )

    # Evaluator / Verifier
    # 只检查工具结果，产出 complete / retry / expand / fail。
    builder.add_conditional_edges(
        "evaluator_verifier",
        route_after_evaluator,
        {
            "commit": "task_committer",
        },
    )

    # Task Committer
    # 根据 Evaluator 判断更新 task_tree / agenda / memory。
    # 如果测试失败且存在 patch_history，会安排 rollback 任务回到 Scheduler。
    builder.add_conditional_edges(
        "task_committer",
        route_after_task_committer,
        {
            "memory": "memory_writer",
            "finish": "summarize",
        },
    )

    # Memory Writer
    # 把 Task Committer 生成的 pending_memory 写入 memory_notes。
    builder.add_conditional_edges(
        "memory_writer",
        route_after_memory_writer,
        {
            "schedule": "scheduler",
            "finish": "summarize",
        },
    )

    # 汇总结束。
    builder.add_edge("summarize", END)

    # Memory / Checkpoint
    #
    # 当前通过 checkpointing.build_checkpointer() 选择后端：
    # - memory：适合本地教学和单进程实验。
    # - postgres：适合本地长任务和进程重启后的状态恢复。
    #
    # beginner_agent 另外还有工具级 checkpoint_save / checkpoint_load，
    # 用于保存 agent 自己的中间资料。
    #
    # 注意：
    # checkpoint 不是 memory.py 里的长期经验库。
    # checkpoint 保存 LangGraph runtime 状态；
    # memory.py 保存 agent 可复用的经验和检索记录。
    checkpointer = build_checkpointer()
    return builder.compile(checkpointer=checkpointer)
