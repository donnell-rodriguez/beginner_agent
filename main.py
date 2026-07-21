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
from beginner_agent.serialization import serialize_for_json
from beginner_agent.state_factory import create_initial_state


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
    initial_state = create_initial_state(user_input)

    # 中文注释：
    # graph.py 里 compile(checkpointer=...) 开启了 checkpoint。
    # 有 checkpoint 时，LangGraph 需要 thread_id 来区分不同会话。
    #
    # 注意：
    #   run_case 是非交互 demo。
    #   如果任务触发 Human Approval interrupt，请使用 cli.py。
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
    # 所以先用 serialize_for_json(...) 转成普通 dict/list/string。
    print(json.dumps(serialize_for_json(result), ensure_ascii=False, indent=2))


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
    #   7. checkpointing.py 选择 Memory/Postgres checkpoint 后端。
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
    #   code_agent_summarize_node 汇总完成项、未完成项、修改文件、
    #   验证结果、恢复建议和风险提示
    run_case("帮我理解 beginner_agent 这个项目结构和执行流程")
