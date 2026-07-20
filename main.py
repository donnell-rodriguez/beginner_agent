# 中文注释：
# __future__.annotations 表示延迟解析类型注解。
# 这里不是业务逻辑，只是让类型注解处理更稳定。
from __future__ import annotations

# 中文注释：
# json 用来把 Python dict 格式化成好看的 JSON 字符串输出。
import json

# 中文注释：
# Path 用来处理文件路径。
# 下面会用它找到当前文件所在目录的上级目录。
from pathlib import Path

# 中文注释：
# sys 可以访问 Python 解释器相关能力。
# 这里主要用 sys.path.append(...) 临时添加模块搜索路径。
import sys


# 中文注释：
# __package__ 表示当前文件是不是作为 package 模块运行。
#
# 当你直接运行：
#
#     python beginner_agent/main.py
#
# 这个文件可能没有 package 上下文。
#
# 这时相对导入可能找不到 beginner_agent 包。
#
# 所以下面这段代码会把项目根目录加入 sys.path，
# 让 Python 能找到 beginner_agent.graph。
if __package__ is None or __package__ == "":
    # 中文注释：
    # Path(__file__) 表示当前 main.py 文件路径。
    # resolve() 转成绝对路径。
    # parents[1] 表示 main.py 的上上级目录，也就是 langgraph 仓库根目录。
    #
    # 最后 sys.path.append(...) 把这个目录加入 Python 模块搜索路径。
    sys.path.append(str(Path(__file__).resolve().parents[1]))

# 中文注释：
# 从 graph.py 导入 build_graph。
#
# build_graph() 会组装 LangGraph：
#   StateGraph(State)
#   add_node(...)
#   add_edge(...)
#   add_conditional_edges(...)
#   compile()
from beginner_agent.graph import build_graph
from beginner_agent.tools import READ_ONLY_TOOLS, WRITE_TOOLS


# 中文注释：
# _serialize_for_json 用来把 LangGraph 的最终 State 转成可打印 JSON。
#
# 为什么需要它？
#   messages: Annotated[list, add_messages] 会把普通 dict 消息
#   转成 LangChain 的 HumanMessage / AIMessage 对象。
#
# 这些对象不能直接 json.dumps(...)。
#
# 所以这里做一层转换：
#   HumanMessage(content="你好")
#   ->
#   {"type": "human", "content": "你好"}
def _serialize_for_json(value):
    # 中文注释：
    # 如果 value 是 list，就递归处理列表里的每个元素。
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]

    # 中文注释：
    # 如果 value 是 dict，就递归处理每个 key/value。
    if isinstance(value, dict):
        return {key: _serialize_for_json(item) for key, item in value.items()}

    # 中文注释：
    # LangChain 的消息对象通常有 type 和 content 两个属性。
    #
    # 例如：
    #   HumanMessage.type    -> "human"
    #   HumanMessage.content -> 用户内容
    #
    #   AIMessage.type      -> "ai"
    #   AIMessage.content   -> 模型内容
    if hasattr(value, "type") and hasattr(value, "content"):
        return {
            "type": value.type,
            "content": value.content,
        }

    # 中文注释：
    # 其他普通值，例如字符串、数字、None，直接返回即可。
    return value


# 中文注释：
# run_case 是一个辅助函数，用来运行一次 agent。
#
# 参数：
#   user_input：用户输入的一句话。
#
# 返回：
#   None，表示这个函数不返回值，只负责打印结果。
def run_case(user_input: str) -> None:
    # 中文注释：
    # 调用 build_graph() 创建并编译 LangGraph。
    #
    # graph 是已经 compile 后的可运行图。
    # 它可以调用 invoke(...) 执行。
    graph = build_graph()

    # 中文注释：
    # graph.invoke(...) 表示执行这张图。
    #
    # 传进去的 dict 就是初始 State。
    #
    # LangGraph 会从 START 开始，
    # 按 graph.py 里定义的节点和边执行，
    # 最后返回完整 State。
    initial_state = {
            # 中文注释：
            # user_input 是用户输入。
            # router_classifier_node 会读取它来判断任务类型、风险等级、是否需要工具。
            "user_input": user_input,

            # 中文注释：
            # task_type 是任务类型。
            # 初始值先给 "chat"。
            # 后面 router_classifier_node 会根据 user_input 改成 search/write/chat/agent。
            "task_type": "chat",

            # 中文注释：
            # risk_level 是风险等级。
            # Router / Classifier 会更新它。
            "risk_level": "low",

            # 中文注释：
            # needs_tool 表示当前任务是否需要工具。
            # Router / Classifier 会更新它。
            "needs_tool": False,

            # 中文注释：
            # route_reason 保存 Router / Classifier 的判断原因。
            "route_reason": "",

            # 中文注释：
            # next_action 表示复杂 agent 下一步应该走哪个模块。
            "next_action": "schedule",

            # 中文注释：
            # draft 是中间结果。
            # 初始为空字符串。
            # search/write/chat 节点会写入它。
            "draft": "",

            # 中文注释：
            # final_answer 是最终结果。
            # 初始为空字符串。
            # summarize_node 会写入它。
            "final_answer": "",

            # 中文注释：
            # tool_name 是 LLM 选择的工具名。
            # 初始值给 "none"，表示还没有选择工具。
            "tool_name": "none",

            # 中文注释：
            # tool_args 是工具参数。
            # 初始给空 dict。
            "tool_args": {},

            # 中文注释：
            # tool_result 是工具执行结果。
            # 初始为空字符串。
            "tool_result": "",

            # 中文注释：
            # tool_result_data 是结构化工具结果。
            # Executor 会写入 Pydantic ToolResult 的 dict。
            # 机器判断优先看这个字段，人类阅读可以看 tool_result 字符串。
            "tool_result_data": {},

            # 中文注释：
            # tool_result_status 明确区分工具执行结果：
            # success / failed / blocked / empty / partial / none。
            "tool_result_status": "none",

            # 中文注释：
            # parent_evaluation 保存“子任务完成后，父任务状态如何变化”。
            "parent_evaluation": {},

            # 中文注释：
            # goal_progress 保存“当前结果距离用户最终目标还有多远”。
            "goal_progress": {},

            # 中文注释：
            # memory_notes 是我们放在 State 里的轻量记忆。
            # LangGraph 的真正 checkpoint 在 graph.py 的 MemorySaver 里。
            "memory_notes": [],

            # 中文注释：
            # memory_context 是 Memory Retriever 读取出来的相关历史记忆。
            # 初始为空，复杂 agent 分支开始后由 memory_retriever_node 写入。
            "memory_context": {},

            # 中文注释：
            # pending_memory 是 Task Committer 准备交给 Memory Writer 保存的记忆。
            # 初始为空，每个任务提交后可能会产生一条。
            "pending_memory": {},

            # 中文注释：
            # root_task_id 是任务树的根任务 id。
            # root 代表用户输入的原始大任务。
            "root_task_id": "root",

            # 中文注释：
            # task_tree 是复杂 agent 的任务树数据结构。
            #
            # 初始为空。
            # scheduler_node 第一次运行时会自动创建 root 任务。
            "task_tree": {},

            # 中文注释：
            # agenda 是待处理任务 id 队列。
            #
            # 注意：
            #   queue agent 的 agenda 里直接放任务对象。
            #   当前复杂 agent 的 agenda 里只放任务 id。
            #
            # 真正的任务内容都保存在 task_tree 里。
            "agenda": [],

            # 中文注释：
            # current_task_id 是当前正在处理的任务 id。
            # 初始为空，scheduler_node 会从 agenda 中选择一个 pending 任务。
            "current_task_id": "",

            # 中文注释：
            # completed_tasks 保存已经完成的任务记录。
            # 它在 State 里是 Annotated[list, add]，所以会自动追加。
            "completed_tasks": [],

            # 中文注释：
            # patch_history 保存 apply_patch 的修改记录。
            # rollback 会基于这里的 before_content 恢复文件。
            "patch_history": [],

            # 中文注释：
            # human_approvals 模拟人工审批。
            #
            # 写工具 apply_patch / rollback 默认不会自动执行。
            # 如果你真的要允许某个任务执行写操作，可以把对应 task_id 设置为 True。
            "human_approvals": {},

            # 中文注释：
            # pending_approval 保存当前等待人工确认的工具调用。
            "pending_approval": {},

            # 中文注释：
            # planner_reason 保存 Planner / Decomposer 的判断原因。
            "planner_reason": "",

            # 中文注释：
            # Plan Validator 用来检查 Planner 生成的计划是否可执行、是否重复、是否符合工具边界。
            "plan_validation_status": "none",
            "plan_validation_reason": "",

            # 中文注释：
            # policy_decision 和 policy_reason 保存工具权限判断结果。
            "policy_decision": "deny",
            "policy_reason": "",

            # 中文注释：
            # evaluation_decision 和 evaluation_reason 保存结果验证判断。
            "evaluation_decision": "none",
            "evaluation_reason": "",

            # 中文注释：
            # done 表示复杂 agent loop 是否结束。
            # 初始 False。
            "done": False,

            # 中文注释：
            # step_count 是当前复杂 agent 已经循环了多少轮。
            "step_count": 0,

            # 中文注释：
            # max_steps 是复杂 agent loop 的最大循环次数。
            #
            # 真实 agent 必须有这个上限，避免 LLM 一直决定继续执行，
            # 导致无限循环。
            "max_steps": 12,

            # 中文注释：
            # max_depth 是任务树最多允许拆几层。
            # 例如 2 表示：
            #   root -> root.1 -> root.1.1
            "max_depth": 2,

            # 中文注释：
            # max_total_tasks 是整棵任务树最多允许有多少个任务节点。
            # 这是防止任务树无限膨胀的安全阀。
            "max_total_tasks": 10,

            # 中文注释：
            # max_task_retries 是单个任务最大重试次数。
            "max_task_retries": 1,

            # 中文注释：
            # allowed_tools 是工具白名单。
            # 当前只允许只读 code-agent 工具，避免修改你的 Mac 文件。
            "allowed_tools": [*READ_ONLY_TOOLS, *WRITE_TOOLS],

            # 中文注释：
            # permission_policy 是工具权限策略。
            #
            # allow：允许直接执行。
            # deny：拒绝执行。
            #
            # 以后如果加入写文件、执行命令等危险工具，
            # 可以扩展成 ask，让 agent 先请求用户确认。
            "permission_policy": {
                **{tool_name: "allow" for tool_name in READ_ONLY_TOOLS},
                # 中文注释：
                # 写工具默认 ask。
                # Tool Policy 会要求 human_approvals[task_id] == True 才执行。
                **{tool_name: "ask" for tool_name in WRITE_TOOLS},
            },

            # 中文注释：
            # messages 是消息历史。
            #
            # 重点：
            #   这个字段在 State 里写的是：
            #   messages: Annotated[list, add_messages]
            #
            # 所以它不是普通覆盖字段，而是“自动追加字段”。
            #
            # 这里先放入第一条用户消息。
            # 后面的 router/write/chat/summarize 节点都会返回新的 messages。
            # LangGraph 会用 add_messages 自动把它们合并起来。
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
        }
    # 中文注释：
    # graph.py 里 compile(checkpointer=MemorySaver()) 开启了 checkpoint。
    # 有 checkpoint 时，LangGraph 需要 thread_id 来区分不同会话。
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "beginner-agent-demo"}},
    )

    # 中文注释：
    # 打印用户输入，方便你看到这次运行的输入是什么。
    print("\n用户输入:")
    print(user_input)

    # 中文注释：
    # 打印最终运行结果。
    # result 是一个 dict，也就是最终 State。
    print("\n运行结果:")

    # 中文注释：
    # json.dumps(...) 把 result 格式化成 JSON 字符串。
    #
    # ensure_ascii=False：
    #   保留中文，不把中文转成 \\uXXXX。
    #
    # indent=2：
    #   用 2 个空格缩进，让输出更容易阅读。
    # 中文注释：
    # 由于 messages 里可能包含 HumanMessage / AIMessage 对象，
    # 所以先用 _serialize_for_json(...) 转成普通 dict/list/string。
    print(json.dumps(_serialize_for_json(result), ensure_ascii=False, indent=2))


# 中文注释：
# 这是 Python 常见入口判断。
#
# 当你直接运行这个文件时：
#
#     python beginner_agent/main.py
#
# __name__ 会等于 "__main__"，
# 下面的代码就会执行。
#
# 如果这个文件是被别的文件 import，
# 下面的代码不会自动执行。
if __name__ == "__main__":
    # 中文注释：
    # 运行一个测试用例。
    #
    # 这次版本已经接入本地 OMLX 模型：
    #   1. router_classifier_node 判断任务类型、风险等级、是否需要工具。
    #   2. scheduler_node 选择下一个 pending 任务。
    #   3. planner_decomposer_node 判断是否拆解。
    #   4. tool_policy_node 判断工具是否允许。
    #   5. executor_node 真正执行工具。
    #   6. evaluator_verifier_node 检查结果是否完成、是否要重试。
    #   7. MemorySaver 保存 checkpoint。
    # run_case("帮我写一段介绍 LangGraph 的文字，适合 Python 小白阅读")

    # 中文注释：
    # 下面两个是备用测试用例。
    # 你可以取消注释，观察 search/chat 分支如何调用模型。
    # run_case("帮我搜索 LangGraph 的核心概念")
    # run_case("LangGraph 是什么？")

    # 中文注释：
    # 这个例子会触发 agent 分支。
    #
    # 注意：
    #   当前版本已经不是“一开始生成完整计划”。
    #
    # 而是更接近复杂 agent 的分层 loop：
    #   router_classifier_node 判断为 agent
    #   scheduler_node 创建 root 任务并选择 pending 任务
    #   planner_decomposer_node 判断是否拆成子任务
    #   plan_validator_node 检查计划质量
    #   tool_policy_node 检查工具权限
    #   executor_node 执行只读工具并写入 tool_result_status
    #   evaluator_verifier_node 验证结果、父任务状态和目标进度
    #   执行后回到 scheduler_node
    #   直到没有 pending 任务，或者达到 max_steps
    #   summarize_node 汇总 task_tree、memory_notes 和 completed_tasks
    run_case("帮我理解 beginner_agent 这个项目结构和执行流程")
