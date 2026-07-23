from operator import add
from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# 中文注释：
# 这里的 TaskType / RiskLevel / ToolName / NextAction 都是 `Literal[...]`。
#
# 对 Python 小白来说，可以先把它理解成“只能从固定选项里选择”的类型。
#
# 例如：
#
#     TaskType = Literal["search", "write", "chat", "agent"]
#
# 意思是 task_type 这个字段只能是：
#
#     "search" / "write" / "chat" / "agent"
#
# 这样做的好处：
# - 代码更清楚：你一眼能看到系统支持哪些任务类型。
# - 更不容易写错：比如把 "search" 写成 "serach"，类型检查工具会提醒。
# - Agent 更可控：大模型不能随便发明一个不存在的工具名或动作名。


# 中文注释：
# TaskType 表示 Router / Classifier 对用户任务的第一层分类。
#
# search：偏资料查找、检索、搜索。
# write：偏写作、生成、润色、总结。
# chat：普通问答，不一定需要工具。
# agent：复杂任务，需要进入多步骤 agent loop。
# 例如读代码、拆任务、调用工具、验证结果。
TaskType = Literal["search", "write", "chat", "agent"]


# 中文注释：
# RiskLevel 表示当前任务的风险等级。
#
# low：低风险，通常只是回答问题、读文件、总结内容。
# medium：中风险，可能运行测试、做静态检查、读取较多项目内容。
# high：高风险，可能修改文件、应用 patch、执行会改变项目状态的操作。
#
# 这个字段主要给 Tool Policy / Permission Layer 使用，
# 用来决定是否允许执行、是否需要人工确认。
RiskLevel = Literal["low", "medium", "high"]
ExecutionMonitorStatus = Literal["ok", "over_budget", "failed", "blocked", "empty", "partial"]
RecoveryAction = Literal[
    "none",
    "retry_same",
    "retry_with_new_args",
    "use_alternative_tool",
    "replan",
    "decompose_more",
    "ask_human",
    "stop_with_summary",
]


# 中文注释：
# ToolName 是当前 Agent 支持的工具名清单。
#
# 你可以把它理解成“Agent 的工具箱目录”。
#
# 大模型不是直接碰你的电脑，而是只能从这里列出的工具里选择。
# 真正执行之前，还会经过 Tool Policy / Permission Layer 做安全检查。
#
# 当前语言范围：
# - Python
# - Rust
#
# 其他语言暂时不支持，后续可以继续增加 Node/Go/Java Adapter。
ToolName = Literal[
    # 中文注释：
    # 目录查看类工具。
    # 用来回答“项目里有哪些文件、目录结构是什么”。
    "list_files",
    "list_tree",

    # 中文注释：
    # 文件读取类工具。
    # read_file：读取完整文件。
    # read_file_slice：只读取某个文件的一部分。
    # 适合大文件，避免一次输出太多。
    "read_file",
    "read_file_slice",

    # 中文注释：
    # 代码搜索类工具。
    # search_code：按关键词搜索代码。
    # grep_regex：按正则表达式搜索代码，表达能力更强。
    "search_code",
    "grep_regex",

    # 中文注释：
    # 代码结构理解类工具。
    # inspect_symbol：查看某个函数、类、变量在哪里定义。
    # inspect_references：查看某个名字在哪里被使用。
    # inspect_import_graph：查看文件之间的 import 依赖关系。
    # inspect_call_graph：查看函数调用关系。
    # build_project_index：建立项目索引，形成“项目地图”。
    # query_project_index：查询项目索引，快速定位文件、符号、依赖。
    # inspect_function_signature：查看函数签名，例如参数和返回值。
    # inspect_class_hierarchy：查看类继承和方法列表。
    # detect_project_stack：识别项目语言、包管理器、测试框架和可用命令。
    # detect_test_framework：识别测试框架。
    # detect_package_manager：识别包管理器，例如 uv/npm/cargo。
    # detect_build_system：识别构建系统。
    # list_allowed_commands：列出白名单命令 profile。
    # run_allowed_command：运行白名单命令 profile，不允许任意 shell。
    # run_package_script：运行预定义脚本，例如 test/lint/typecheck/build。
    # run_pytest：运行白名单 pytest 目标。
    # run_ruff：运行 ruff check。
    # run_ruff_format_check：运行 ruff format --check。
    # run_mypy：运行 mypy。
    # run_uv_import_graph：验证 beginner_agent 图能否构建。
    # run_cargo_check：运行 Rust cargo check。
    # run_cargo_test：运行 Rust cargo test。
    # run_cargo_clippy：运行 Rust clippy。
    # run_cargo_fmt_check：运行 Rust cargo fmt --check。
    # detect_rust_project：识别 Rust 项目。
    # inspect_rust_symbols：扫描 Rust fn/struct/enum/trait/impl。
    # inspect_rust_references：查找 Rust 符号引用。
    # parse_rust_errors：解析 rustc/cargo 错误。
    # parse_cargo_test_failure：解析 cargo test 失败。
    # map_changed_rust_files_to_tests：根据 Rust 改动文件推断测试目标。
    # safe_path_exists：安全检查 beginner_agent 内路径是否存在。
    "inspect_symbol",
    "inspect_references",
    "inspect_import_graph",
    "inspect_call_graph",
    "build_project_index",
    "query_project_index",
    "inspect_function_signature",
    "inspect_class_hierarchy",
    "detect_project_stack",
    "detect_test_framework",
    "detect_package_manager",
    "detect_build_system",
    "list_allowed_commands",
    "run_allowed_command",
    "run_package_script",
    "run_pytest",
    "run_ruff",
    "run_ruff_format_check",
    "run_mypy",
    "run_uv_import_graph",
    "run_cargo_check",
    "run_cargo_test",
    "run_cargo_clippy",
    "run_cargo_fmt_check",
    "detect_rust_project",
    "inspect_rust_symbols",
    "inspect_rust_references",
    "parse_rust_errors",
    "parse_cargo_test_failure",
    "map_changed_rust_files_to_tests",
    "safe_path_exists",

    # 中文注释：
    # 检查与测试类工具。
    # static_check：做基础静态检查，例如 Python 语法是否能编译。
    # lint_typecheck：运行 lint / typecheck，检查风格、类型等问题。
    # run_tests：运行项目测试。
    # run_targeted_tests：只运行某个指定测试，适合修复单个问题。
    # parse_test_failure：解析测试失败信息，帮助下一步修复。
    # run_typecheck：单独运行类型检查。
    # run_build：运行构建命令，检查项目是否能构建成功。
    # get_diagnostics：汇总诊断信息。
    # format_check：检查代码格式，但不修改文件。
    # format_apply：应用格式化，可能修改文件，所以风险更高。
    # extract_stack_trace：从失败日志中提取 traceback。
    # classify_failure：把失败分类成语法错误、测试失败、导入错误等。
    # compare_failure_before_after：比较修复前后的失败是否改善。
    "static_check",
    "lint_typecheck",
    "run_tests",
    "run_targeted_tests",
    "parse_test_failure",
    "extract_stack_trace",
    "classify_failure",
    "compare_failure_before_after",
    "run_typecheck",
    "run_build",
    "get_diagnostics",
    "format_check",
    "format_apply",

    # 中文注释：
    # Git 观察类工具。
    # git_status：查看当前工作区状态。
    # git_diff：查看所有改动。
    # git_diff_file：查看某个文件的改动。
    #
    # 这些主要用于“修改前后对比”和“确认 agent 到底改了什么”。
    "git_status",
    "git_diff",
    "git_diff_file",

    # 中文注释：
    # 影响测试选择类工具。
    # map_changed_files_to_tests：根据改动文件推断相关测试。
    # select_relevant_tests：根据任务目标或改动文件选择优先测试。
    # run_impacted_tests：先选择受影响测试，再通过白名单 pytest 入口验证。
    "map_changed_files_to_tests",
    "select_relevant_tests",
    "run_impacted_tests",

    # 中文注释：
    # 补丁计划与代码修改类工具。
    # patch_plan：生成一个修改计划。
    # validate_patch_plan：检查修改计划是否安全、合理。
    # apply_patch_plan：执行经过验证的修改计划。
    # preview_patch：只预览 diff，不修改文件。
    # apply_patch_dry_run：干跑 patch，验证能不能应用，但不写文件。
    # validate_patch_scope：检查修改范围和风险提示。
    # apply_patch：直接应用 patch 修改文件。
    # rollback：回滚最近一次修改。
    # revert_file_patch：按指定内容回滚某个文件。
    #
    # 这一组工具是 code repair agent 的核心，
    # 因为它们让 Agent 从“只读分析”升级为“可以修代码”。
    "patch_plan",
    "validate_patch_plan",
    "preview_patch",
    "apply_patch_dry_run",
    "validate_patch_scope",
    "apply_patch_plan",

    # 中文注释：
    # 记忆 / checkpoint 类工具。
    # checkpoint_save：保存当前中间状态。
    # checkpoint_load：读取之前保存的中间状态。
    #
    # 它们用于长任务恢复，避免任务做到一半丢失上下文。
    "checkpoint_save",
    "checkpoint_load",

    # 中文注释：
    # 安全与项目理解类工具。
    # secret_scan：扫描是否可能包含密钥、token、密码等敏感信息。
    # dependency_inspect：查看项目依赖。
    # summarize_file：总结某个文件的作用。
    # audit_tool_call：记录工具调用审计事件。
    # audit_patch：记录 patch 审计事件。
    # read_audit_log：读取最近审计日志。
    # list_tool_catalog：查看工具目录和工具元数据。
    # describe_tool：查看单个工具的风险、语言、权限等信息。
    # tool_policy_report：查看工具平台权限策略。
    # list_project_roots：查看已注册项目根。
    # get_active_project：查看当前 active project。
    "secret_scan",
    "dependency_inspect",
    "summarize_file",
    "audit_tool_call",
    "audit_patch",
    "read_audit_log",
    "list_tool_catalog",
    "describe_tool",
    "tool_policy_report",
    "list_project_roots",
    "get_active_project",

    # 中文注释：
    # 直接修改 / 回滚工具。
    # apply_patch：直接修改文件，通常需要人工审批。
    # rollback：把最近一次修改恢复回去。
    # register_project_root：注册新的受控项目根，默认需要审批。
    # set_active_project：切换 active project，默认需要审批。
    "apply_patch",
    "rollback",
    "revert_file_patch",
    "register_project_root",
    "set_active_project",

    # 中文注释：
    # none 表示当前不需要工具。
    # 例如普通聊天、最终总结、或者某一步只做判断。
    "none",
]


# 中文注释：
# NextAction 表示当前节点执行完后，复杂 agent loop 下一步去哪里。
#
# 它不是工具名，而是“控制流程”的动作名。
#
# 例如：
#
#     next_action = "plan"
#
# 表示下一步进入 Planner / Decomposer 节点。
#
#     next_action = "execute"
#
# 表示下一步进入 Executor 节点，真正执行工具。
NextAction = Literal[
    # 中文注释：回到 Scheduler / Agenda Manager，选择下一个 pending task。
    "schedule",
    # 中文注释：进入 Planner / Decomposer，判断任务是否要拆成子任务。
    "plan",
    # 中文注释：进入 Plan Validator，检查计划是否合理。
    "validate",
    # 中文注释：进入 Tool Policy / Permission Layer，检查工具是否允许执行。
    "policy",
    # 中文注释：
    # 进入 Approval Interrupt，处理需要人工确认的工具调用。
    "approval",
    # 中文注释：进入 Sandbox Runner，准备受控运行边界。
    "sandbox",
    # 中文注释：进入 Executor，真正执行工具。
    "execute",
    # 中文注释：进入 Async Job Waiter，等待后台任务或确认同步完成。
    "wait",
    # 中文注释：进入 Execution Monitor / Watchdog，检查执行是否超预算或卡住。
    "monitor",
    # 中文注释：
    # 进入 Recovery Planner，决定重试、换方案、重新拆解或停止总结。
    "recover",
    # 中文注释：
    # 进入 Evaluator / Verifier，检查执行结果是否完成、失败、需要重试。
    "evaluate",
    # 中文注释：
    # 进入 Task Committer，把 Evaluator 的判断真正写回 task_tree / agenda / memory。
    #
    # 这样 Evaluator 只负责“判断”，
    # Committer 只负责“落库式更新状态”，职责更清楚。
    "commit",
    # 中文注释：
    # 进入 Memory Writer，把本轮产生的结构化经验写入 memory_notes。
    "memory",
    # 中文注释：
    # 进入 Memory Compaction，把重复/相似长期记忆压缩成更稳定的摘要记忆。
    "compact",
    # 中文注释：进入 Artifact Collector，收集补丁、改动文件和验证产物。
    "artifact",
    # 中文注释：进入 Observability Reporter，汇总运行报告。
    "observability",
    # 中文注释：进入最终汇总，结束 agent loop。
    "finish",
]


class State(TypedDict):
    """复杂 agent 运行过程中共享的状态。"""

    # 中文注释：
    # run_id 表示一次 agent 运行的唯一 id。
    #
    # Artifact Storage / Observability / Async Job / Audit
    # 都可以用它把同一次运行产生的数据串起来。
    run_id: str

    # 中文注释：
    # thread_id 是 LangGraph checkpoint 恢复时常用的线程标识。
    #
    # 真实 LangGraph checkpoint 通常通过 invoke/stream 的 config 传入：
    #
    #     {"configurable": {"thread_id": "..."}}
    #
    # 这里把 thread_id 也放进 State，是为了让 checkpoint_report
    # 可以在 Summary / Observability 里说明当前运行是否具备恢复条件。
    thread_id: str

    # 中文注释：
    # 用户最开始输入的问题或任务。
    user_input: str

    # 中文注释：
    # Router / Classifier 输出的任务类型。
    task_type: TaskType

    # 中文注释：
    # Router / Classifier 输出的风险等级。
    # low：只读、问答、总结类任务。
    # medium：可能需要执行工具，但仍在安全边界内。
    # high：可能修改文件、执行命令、删除数据等高风险任务。
    risk_level: RiskLevel

    # 中文注释：
    # Router / Classifier 判断这个任务是否需要工具。
    needs_tool: bool

    # 中文注释：
    # Router / Classifier 给出的路由原因，方便调试和学习。
    route_reason: str

    # 中文注释：
    # router_report 保存 Router 的结构化观测信息。
    #
    # 它会记录：
    # - 本次决策来自 LLM / fallback / security_override。
    # - Router 花了多久。
    # - 是否命中 prompt injection / 数据外泄 / 高风险动作。
    # - 最终 task_type / risk_level / needs_tool。
    #
    # 这让后续排查“为什么任务走到某个分支”时有证据可看。
    router_report: dict[str, Any]

    # 中文注释：
    # 当前复杂 agent 下一步应该走哪个模块。
    #
    # schedule：回到 Scheduler 选择任务。
    # plan：进入 Planner / Decomposer。
    # validate：进入 Plan Validator。
    # policy：进入 Tool Policy / Permission Layer。
    # approval：进入 Approval Interrupt，等待或检查人工审批。
    # execute：进入 Executor。
    # evaluate：进入 Evaluator / Verifier。
    # commit：进入 Task Committer，把评估结论写回任务树和记忆。
    # memory：进入 Memory Writer，保存本轮任务经验。
    # compact：进入 Memory Compaction，整理长期记忆库。
    # finish：进入最终汇总。
    next_action: NextAction

    # 中文注释：
    # 中间草稿。
    draft: str

    # 中文注释：
    # 最终答案。
    final_answer: str

    # 中文注释：
    # 消息历史。
    # add_messages 会让节点返回的新 messages 自动追加到历史后面。
    messages: Annotated[list, add_messages]

    # 中文注释：
    # Memory / Checkpoint 里的轻量记忆。
    #
    # 这不是 LangGraph 的底层 checkpoint，
    # 而是我们放在 State 里、方便你观察的“任务记忆”。
    memory_notes: Annotated[list, add]

    # 中文注释：
    # memory_context 是 Memory Retriever 读出来的“相关历史经验”。
    #
    # 它不是长期数据库，只是当前运行 State 里的一段可读上下文。
    # Planner / Evaluator 以后可以参考它，避免重复犯错或重复读取文件。
    #
    # 当前 memory_context 里也会包含：
    # - user_preferences：长期用户/项目偏好。
    #   例如中文注释、配置进 .env、修改后必须测试、优先大厂风格等。
    # - relevant_records：和当前任务相关的项目/工具/失败经验。
    memory_context: dict[str, Any]

    # 中文注释：
    # pending_memory 是 Task Committer 生成、等待 Memory Writer 写入的记忆。
    #
    # 为什么不直接在 Committer 里写 memory_notes？
    # 因为现在我们把“状态提交”和“记忆写入”拆成两个节点，
    # 这样更接近生产系统里的职责边界。
    pending_memory: dict[str, Any]

    # 中文注释：
    # memory_compaction_report 保存长期记忆压缩报告。
    #
    # 为什么需要这个字段？
    # 长时间运行后 memory 会越来越多。
    # 如果每次都检索所有碎片记录，速度会变慢，上下文也会变脏。
    #
    # Memory Compaction 会把：
    # - 多条相似经验合并成规则。
    # - 多次失败合并成 failure pattern。
    # - 一个文件的多次修改总结成 file memory。
    # - 一个项目阶段总结成 project memory。
    #
    # 这个 report 用来告诉 Summary / API：
    # 本轮压缩了多少条、旧记忆归档了多少条、使用哪个 backend。
    memory_compaction_report: dict[str, Any]

    # 中文注释：
    # checkpoint_report 记录当前 LangGraph checkpoint 后端。
    #
    # 真正 checkpoint 保存发生在 graph.py 的 compile(checkpointer=...)。
    # 这个字段只是把运行时 checkpoint 信息暴露给 Summary / Observability。
    checkpoint_report: dict[str, Any]

    # 中文注释：
    # sandbox_report 记录 Sandbox Runner 对当前工具调用的运行边界判断。
    #
    # 当前是本地受控工具层；后续可以升级成容器 sandbox 或远程 runner。
    sandbox_report: dict[str, Any]

    # 中文注释：
    # async_job_report 记录 Async Job Waiter 的等待状态。
    #
    # 当前大多数工具同步完成；后续接远程 worker 时可以保存 job_id / 状态。
    async_job_report: dict[str, Any]

    # 中文注释：
    # artifact_report 保存本轮 agent 产物索引。
    #
    # 例如修改文件、patch 数量、验证任务、执行尝试数量。
    artifact_report: dict[str, Any]

    # 中文注释：
    # observability_report 保存可观测性报告。
    #
    # 它让最终 summary 能看到 step_count、task 状态分布、checkpoint、
    # sandbox、async job、artifact、policy、recovery 等信息。
    observability_report: dict[str, Any]

    # 中文注释：
    # run_lineage_report 是某次 run 的完整运行链路。
    #
    # 它把以下内容串起来：
    # - checkpoint：这次运行用哪个 checkpoint 后端。
    # - memory：读取了哪些记忆、生成了哪些记忆。
    # - tool result：执行了哪个工具，结果是否成功。
    # - audit event：产生了哪些治理/隐私/记忆审计事件。
    # - observability：最后是否完成，目标进度如何。
    #
    # 这就是更强系统里的 run-level lineage / trace。
    run_lineage_report: dict[str, Any]

    # 中文注释：
    # root_task_id 是整棵任务树的根任务 id。
    root_task_id: str

    # 中文注释：
    # task_tree 是任务树。
    #
    # key 是 task_id，value 是任务节点。
    #
    # 一个任务节点大致长这样：
    #
    # {
    #   "id": "root.1",
    #   "title": "读取 graph.py",
    #   "status": "pending",
    #   "parent_id": "root",
    #   "children": [],
    #   "depth": 1,
    #   "tool": "read_file",
    #   "args": {"path": "graph.py"},
    #   "retry_count": 0,
    #   "result": "",
    # }
    task_tree: dict[str, dict[str, Any]]

    # 中文注释：
    # Scheduler / Agenda Manager 使用的待处理任务 id 队列。
    agenda: list[str]

    # 中文注释：
    # Scheduler 选中的当前任务 id。
    current_task_id: str

    # 中文注释：
    # Executor 完成后的任务记录。
    completed_tasks: Annotated[list, add]

    # 中文注释：
    # patch_history 保存代码修改记录。
    #
    # apply_patch 成功后会记录：
    # - 修改了哪个文件
    # - 修改前内容
    # - 修改后内容
    #
    # rollback 可以基于它恢复最近一次修改。
    patch_history: Annotated[list, add]

    # 中文注释：
    # execution_status 表示 Executor 这一层的执行状态。
    #
    # 注意它和 tool_result_status 不完全一样：
    # - tool_result_status 关注“工具结果是否成功”。
    # - execution_status 关注“执行过程是否正常、是否超时、是否需要等待”。
    #
    # 例如：
    # - 工具最终成功，但耗时超过预算，可以是 completed_over_budget。
    # - 工具被权限/参数拦截，可以是 blocked。
    # - 未来接入后台任务队列时，可以扩展成 waiting_external。
    execution_status: Literal[
        "not_started",
        "completed",
        "completed_over_budget",
        "failed",
        "blocked",
        "waiting_external",
    ]

    # 中文注释：
    # active_execution 保存当前正在执行或最近一次执行的摘要。
    #
    # 它类似生产系统里的 job/run record：
    # 记录 task_id、tool_name、开始时间、结束时间、耗时、
    # 是否长任务工具等。
    active_execution: dict[str, Any]

    # 中文注释：
    # execution_attempts 保存所有执行尝试记录。
    #
    # 它是 Annotated[list, add]，所以每次 Executor 返回一条 attempt，
    # LangGraph 会自动追加，而不是覆盖旧记录。
    execution_attempts: Annotated[list, add]

    # 中文注释：
    # max_tool_duration_ms 是工具执行预算。
    #
    # 当前工具还是同步执行，所以这个字段不能强行杀掉正在运行的工具。
    # 它的作用是：执行完成后判断是否超过预算，并把结果交给 Evaluator。
    # 后续接后台 worker 时，可以把它变成真正的 timeout / cancellation 策略。
    max_tool_duration_ms: int

    # 中文注释：
    # human_approvals 保存已经完成的人工审批结果。
    #
    # key 是 task_id，value 是 True / False。
    # 写文件工具默认需要审批，Approval Interrupt 会通过 LangGraph interrupt
    # 暂停图执行，等 CLI / UI 用 Command(resume=...) 恢复。
    human_approvals: dict[str, bool]

    # 中文注释：
    # pending_approval 保存当前等待人工确认的操作，
    # CLI 会把这个内容展示给用户。
    pending_approval: dict[str, Any]

    # 中文注释：
    # Planner / Decomposer 给出的是否拆解、是否执行等原因。
    planner_reason: str

    # 中文注释：
    # Plan Validator 对 Planner 生成的计划质量做判断。
    #
    # valid：计划可用。
    # invalid：计划不可用。
    plan_validation_status: Literal["valid", "invalid", "none"]

    # 中文注释：
    # Plan Validator 给出的判断原因。
    plan_validation_reason: str

    # 中文注释：
    # Tool Policy / Permission Layer 给出的权限决策。
    policy_decision: Literal["allow", "deny", "ask"]

    # 中文注释：
    # Tool Policy / Permission Layer 给出的权限原因。
    policy_reason: str

    # 中文注释：
    # Evaluator / Verifier 对最近一次工具结果的判断。
    evaluation_decision: Literal["complete", "retry", "expand", "fail", "none"]

    # 中文注释：
    # Evaluator / Verifier 给出的判断原因。
    evaluation_reason: str

    # 中文注释：
    # 当前任务选择的工具名。
    tool_name: ToolName

    # 中文注释：
    # 当前工具参数。
    tool_args: dict[str, Any]

    # 中文注释：
    # 最近一次工具执行结果。
    tool_result: str

    # 中文注释：
    # 最近一次工具执行的结构化结果。
    #
    # 它来自 Pydantic ToolResult：
    # - status：success / failed / blocked / empty / partial
    # - output：工具输出文本
    # - duration_ms：耗时
    # - retryable：是否建议重试
    # - validation：参数校验结果
    # - changed_files：工具可能修改的文件
    #
    # tool_result 仍然保留字符串形式，方便手机阅读；
    # tool_result_data 用于 Evaluator / Audit / Memory 做机器判断。
    tool_result_data: dict[str, Any]

    # 中文注释：
    # 工具结果状态。
    #
    # success：工具正常返回可用内容。
    # failed：工具报错或返回明确错误。
    # blocked：工具被权限层拦截。
    # empty：工具返回空内容。
    # partial：工具返回部分内容，例如被截断。
    # none：还没有执行工具。
    tool_result_status: Literal["success", "failed", "blocked", "empty", "partial", "none"]

    # 中文注释：
    # execution_monitor_status 是 Watchdog 对最近一次执行的观察结果。
    #
    # ok：执行过程正常。
    # over_budget：执行完成了，但超过 max_tool_duration_ms 预算。
    # failed：执行失败。
    # blocked：执行被权限或参数阻断。
    # empty：执行返回空结果。
    # partial：执行只返回部分结果。
    execution_monitor_status: ExecutionMonitorStatus

    # 中文注释：
    # execution_monitor_reason 是 Watchdog 给出的观察原因。
    execution_monitor_reason: str

    # 中文注释：
    # recovery_action 是 Recovery Planner 给出的恢复动作。
    #
    # 这就是你提到的“长时间没有结果怎么办”的工程化表达：
    # - retry_same：同样方式再试一次。
    # - retry_with_new_args：换参数再试。
    # - use_alternative_tool：换工具。
    # - replan：让后续 Planner 基于失败重新规划。
    # - decompose_more：把任务继续拆小。
    # - ask_human：交给人类确认。
    # - stop_with_summary：停止并如实总结已完成/未完成。
    recovery_action: RecoveryAction

    # 中文注释：
    # recovery_reason 是 Recovery Planner 的决策原因。
    recovery_reason: str

    # 中文注释：
    # partial_result 保存长任务或失败任务已经拿到的部分结果。
    #
    # 这样即使任务没有完全成功，Summarizer / Memory Writer 也能如实说明：
    # 哪些完成了，哪些没完成，下次应该从哪里继续。
    partial_result: str

    # 中文注释：
    # resume_hint 保存下一次继续任务时的建议入口。
    resume_hint: str

    # 中文注释：
    # 父任务评估结果。
    # 例如某个子任务完成后，父任务是 partial / done / needs_more。
    parent_evaluation: dict[str, Any]

    # 中文注释：
    # 目标进度评估。
    # 用来观察当前执行结果距离用户目标还有多远。
    goal_progress: dict[str, Any]

    # 中文注释：
    # done 表示复杂 agent loop 是否结束。
    done: bool

    # 中文注释：
    # step_count 是复杂 agent 已经循环了多少轮。
    step_count: int

    # 中文注释：
    # max_steps 是最大循环轮数。
    max_steps: int

    # 中文注释：
    # max_depth 是任务树最多拆几层。
    max_depth: int

    # 中文注释：
    # max_total_tasks 是整棵任务树最多允许有多少个任务节点。
    max_total_tasks: int

    # 中文注释：
    # 单个任务最多允许重试几次。
    max_task_retries: int

    # 中文注释：
    # allowed_tools 表示工具白名单。
    allowed_tools: list[str]

    # 中文注释：
    # permission_policy 表示工具权限策略。
    #
    # allow：允许执行。
    # ask：需要人工确认，本教学项目会安全拒绝。
    # deny：拒绝执行。
    permission_policy: dict[str, str]
