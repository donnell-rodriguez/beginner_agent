# Beginner LangGraph Layered Agent

这是一个用于学习复杂 Agent 架构的 LangGraph 示例项目。

当前定位：

```text
受控代码修复 Agent 原型
```

它已经不只是读取文件和理解代码，也具备教学版的代码修改闭环：

```text
观察代码 -> 计划修改 -> 人工审批 -> apply_patch -> run_tests/lint -> 失败时 rollback
```

当前版本把 agent 拆成更接近真实工程的七个模块：

```text
1. Router / Classifier
2. Planner / Decomposer
3. Scheduler / Agenda Manager
4. Tool Policy / Permission Layer
5. Executor
6. Evaluator / Verifier
7. Memory / Checkpoint
```

并且额外加入了四个更真实的工程能力：

```text
Plan Validator          检查计划质量
Tool Result Status      区分 success / failed / blocked / empty / partial
Parent Task Evaluation  子任务完成后评估父任务状态
Goal Progress           评估当前结果距离用户目标还有多远
```

当前版本把工具层升级成“受控 code-agent 工具箱”：

```text
文件读取：list_files / list_tree / read_file / read_file_slice
代码搜索：search_code / grep_regex
代码结构：inspect_symbol / inspect_references / inspect_import_graph / inspect_call_graph
诊断测试：static_check / lint_typecheck / run_tests / run_targeted_tests / get_diagnostics
构建检查：run_typecheck / run_build / parse_test_failure
Git 观察：git_status / git_diff / git_diff_file
Patch 闭环：patch_plan / validate_patch_plan / apply_patch_plan / apply_patch / rollback
安全记忆：checkpoint_save / checkpoint_load / secret_scan / dependency_inspect / summarize_file
格式工具：format_check / format_apply
```

## 核心思想

```text
不是把所有逻辑塞进一个 node。

而是让每个 node 承担一个清晰职责：

Router     判断任务类型、风险等级、是否需要工具
MemoryR    复杂任务开始前读取轻量记忆
Scheduler  从 agenda 选择下一个 pending 任务
Planner    判断任务是否需要拆解，并生成子任务
Selector   为叶子任务选择具体工具和参数
Validator  检查 Planner 生成的计划是否可执行
Policy     判断工具是否允许、是否需要人工确认
Approval   处理需要人工确认的工具调用
Executor   真正执行工具
Evaluator  检查结果是否完成、是否重试、是否继续拆解
Committer  把 Evaluator 判断写回 task_tree / agenda
MemoryW    把本轮任务经验写入 memory_notes
Checkpoint 由 LangGraph runtime 保存中间状态
Rollback   测试失败时可基于 patch_history 安排回滚任务
```

## 当前图结构

```text
START
  -> router_classifier
  -> route_by_task
      -> search -> summarize -> END
      -> write  -> summarize -> END
      -> chat   -> summarize -> END
      -> agent
          -> memory_retriever
          -> scheduler
          -> planner_decomposer
          -> tool_selector
          -> plan_validator
          -> tool_policy
          -> human_approval   # 只在 policy_decision=ask 时进入
          -> executor
          -> evaluator_verifier
          -> task_committer
          -> memory_writer
          -> scheduler
          -> ...
          -> summarize
          -> END
```

## 七层模块说明

### 1. Router / Classifier

位置：

```text
router.py -> router_classifier_node(...)
```

职责：

```text
判断 task_type
判断 risk_level
判断 needs_tool
写入 route_reason
```

输出示例：

```python
{
    "task_type": "agent",
    "risk_level": "low",
    "needs_tool": True,
    "route_reason": "用户需要读取项目文件并理解执行流程",
}
```

### 2. Planner / Decomposer

位置：

```text
planner.py -> planner_decomposer_node(...)
```

职责：

```text
判断当前任务是否太大
如果太大，拆成子任务
如果足够具体，进入工具策略检查
清洗 LLM 输出的工具参数
避免重复生成相同任务
用户明确要求修复代码时，可以生成 apply_patch / run_tests / git_diff 等任务
```

它会生成这样的子任务：

```python
{
    "id": "root.1",
    "title": "读取 graph.py",
    "status": "pending",
    "parent_id": "root",
    "children": [],
    "depth": 1,
    "tool": "read_file",
    "args": {"path": "graph.py"},
    "retry_count": 0,
    "result": "",
}
```

### 2.5. Plan Validator

位置：

```text
plan_validator.py -> plan_validator_node(...)
```

职责：

```text
检查 Planner 生成的计划质量
判断子任务是否存在
判断子任务是否重复
判断子任务是否绑定允许的工具
判断工具参数是否安全
判断叶子任务是否可以进入工具策略检查
```

为什么需要它：

```text
Planner 生成计划，不代表计划一定好。
Plan Validator 负责在执行前拦住无效计划。
```

### Plan Validator 后续升级 TODO

当前版本的 Plan Validator 是第一层本地规则验证器。
它主要检查计划结构、安全边界和基本可执行性。

后续可以继续增加这些层：

```text
TODO 1. 语义质量检查
判断计划是否真的覆盖用户目标。
例如用户要求“理解整个项目”，计划却只读取一个无关文件，就应该判为不充分。
这一层可以引入 LLM Plan Critic / Semantic Validator。

TODO 2. 成本和风险检查
判断计划是否过大、过贵、过深、步骤过多、是否触碰高风险工具。
例如一次性生成 50 个 read_file 子任务，就应该被压缩或拒绝。

TODO 3. 可恢复性检查
判断每一步是否有清晰状态记录，失败后能否重试或跳过。
例如任务没有 id、没有 parent_id、没有 retry_count，就不利于恢复。

TODO 4. 覆盖度检查
判断子任务是否覆盖父任务的关键部分。
例如分析项目时，是否至少看了入口文件、图结构文件、节点文件、状态文件。

TODO 5. 计划改写机制
当计划无效时，不只是标记 failed，而是把原因发回 Planner，让 Planner 重新生成计划。
对应流程可以是：
Planner -> Plan Validator -> Planner retry -> Plan Validator
```

### 3. Scheduler / Agenda Manager

位置：

```text
scheduler.py -> scheduler_node(...)
```

职责：

```text
维护 agenda
从 agenda 中选择下一个 pending task_id
没有 pending 任务时结束 loop
```

注意：

```text
agenda 只保存 task_id。
真正的任务内容保存在 task_tree。
```

### 4. Tool Policy / Permission Layer

位置：

```text
policy.py -> tool_policy_node(...)
```

职责：

```text
检查工具是否在 allowed_tools 白名单里
检查 permission_policy 是 allow / ask / deny
如果 risk_level 是 high，会升级为 ask
```

当前教学项目为了安全：

```text
ask 不会自动执行。
deny 不会执行。
只读工具可以自动执行。
写工具 apply_patch / rollback 默认 ask。
写工具必须有 human_approvals[task_id] == True 才会执行。
即使工具在白名单里，也会再次检查工具参数。
```

### 5. Executor

位置：

```text
executor.py -> executor_node(...)
```

职责：

```text
真正调用 tools.py 里的 run_tool_model(...)
得到 Pydantic ToolResult
把工具结果写回 task_tree[current_task_id]["result"]
把结构化结果写回 task_tree[current_task_id]["tool_result_data"]
写入 tool_result_status
区分 success / failed / blocked / empty / partial
记录 duration_ms / retryable / validation / metadata / changed_files
生成 execution_attempts，记录每次执行尝试
写入 execution_status，区分执行过程是否正常、失败、blocked、超预算
```

ToolResult 的作用：

```text
tool_result       给人看的字符串输出
tool_result_data  给机器看的结构化结果

Evaluator / Memory / Audit 后续应该优先读取 tool_result_data。
```

Execution Attempt 的作用：

```text
executor.py 现在不只是“调用工具然后返回字符串”。
它会把每一次工具调用记录成一次 execution attempt：

attempt_id           本次执行尝试 id
task_id              对应哪个任务
tool_name            执行哪个工具
tool_args            工具参数
execution_mode       sync / long_running_sync
execution_status     completed / completed_over_budget / failed / blocked
duration_ms          实际耗时
budget_ms            本次执行预算
over_budget          是否超过预算
retryable            是否建议重试
tool_result_data     工具结构化结果
```

为什么需要这一层：

```text
真实 code agent 里的执行不是只有一种情况：

1. 立即完成
   例如 read_file、list_files。

2. 需要较长时间
   例如 run_tests、run_build、cargo test。

3. 执行失败
   例如参数错误、测试失败、命令失败。

4. 被权限层拦截
   例如 apply_patch 没有人类审批。

5. 未来可能进入后台等待
   例如把长任务提交给 worker，然后轮询 job_id。

当前 beginner_agent 仍然是同步工具执行。
所以它不会真的启动后台 worker，也不会真正异步等待。
但是 active_execution / execution_attempts / execution_status 已经把接口预留出来。

后续如果升级成大厂式执行平台，可以把：

sync / long_running_sync

替换成：

async_job + job_id + polling + cancel + resume
```

### 5A. 修改治理流程

现在项目已经不鼓励直接改文件，而是走一条治理链路：

```text
patch_plan
  -> validate_patch_plan
  -> preview_patch / apply_patch_dry_run
  -> human approval
  -> apply_patch_plan
  -> automatic verification tasks
  -> evaluator / recovery
```

为什么这么做：

```text
LLM 不能直接碰文件。
LLM 只能提出修改计划。
修改计划必须被验证。
真正写入必须经过审批。
写入后必须记录 before/after/hash/diff。
写入后必须自动安排验证任务。
验证失败时进入 Recovery Planner，必要时 rollback 或重新规划。
```

当前关键保护：

```text
1. 普通流程禁止直接 apply_patch
   policy.py 会拒绝直接 apply_patch。
   除非任务显式设置 allow_direct_patch=True。

2. PatchPlan 保存验证状态
   patch_plan_tool 会记录 created_file_hash。
   validate_patch_plan_tool 会写入：
     validated
     validated_at
     validated_file_hash
     validation_issues
     preview

3. apply_patch_plan 必须使用已验证计划
   如果 PatchPlan 没有 validated=True，拒绝执行。
   如果目标文件在验证后发生变化，拒绝执行。

4. Executor 写前还有 preflight
   即使 Policy 通过了，Executor 还会再次检查：
     是否通过审批
     是否直接 apply_patch
     PatchPlan 是否验证
     文件 hash 是否匹配
     rollback 是否有 patch_history

5. Executor 写后记录证据
   patch_history 记录：
     before_content
     after_content
     before_hash
     after_hash
     diff
     changed_line_count
     patch_plan_id

6. 写成功后自动安排验证任务
   Task Committer 会插入：
     git_diff_file
     secret_scan
     static_check
     run_targeted_tests

所以写任务不会立刻算真正完成。
它会先进入 pending_verification，等验证任务跑完后再由 Evaluator / Parent Evaluation 判断。
```

### 5B. Execution Monitor / Watchdog

位置：

```text
execution_monitor.py -> execution_monitor_node(...)
```

职责：

```text
Executor 执行完后，不直接进入 Evaluator。
先由 Watchdog 检查这次执行过程是否正常。

它会观察：
  - 是否 blocked
  - 是否 failed
  - 是否 empty
  - 是否 partial
  - 是否 completed_over_budget

如果一切正常：
  进入 Evaluator。

如果执行不理想：
  进入 Recovery Planner。
```

为什么要这样设计：

```text
长任务不能只靠“等它跑完”。

大厂式 agent 通常会把“执行”和“监控”拆开：

Executor 负责做事。
Watchdog 负责观察做事过程是否健康。

这样以后可以升级成：
  - 查询后台 job 状态
  - 轮询 worker
  - 判断是否卡住
  - 取消超时任务
  - 把部分结果保存下来
```

### 5C. Recovery Planner

位置：

```text
recovery.py -> recovery_planner_node(...)
```

职责：

```text
当 Watchdog 发现执行失败、超预算、空结果、部分结果时，
Recovery Planner 决定下一步恢复动作。

当前支持：

retry_same            同样方式重试
retry_with_new_args   调整参数后重试
use_alternative_tool  换工具
replan                重新规划
decompose_more        继续拆小任务
ask_human             请求人工确认
stop_with_summary     停止并如实总结
```

这里的关键思想：

```text
不是每次失败都问 LLM。

简单明确的问题先用本地规则：
  - blocked -> ask_human
  - failed/empty 且有重试额度 -> retry
  - over_budget -> decompose_more 或 stop_with_summary
  - partial -> stop_with_summary

复杂或连续失败时，再请求 LLM 给恢复建议。

LLM 只能从固定 action 里选择，不能直接绕过 Tool Policy。
```

这正是你提到的长任务处理方式：

```text
长时间没有结果
  -> 不继续盲等
  -> 判断是否换方法
  -> 必要时请求 LLM 给新方案
  -> 如果继续不划算，就如实总结完成/未完成
  -> 写入 memory，下一次可以接着做
```

当前工具：

```text
list_files(path=".")
list_tree(path=".", max_depth=2)
read_file(path="state.py")
read_file_slice(path="state.py", start=1, end=80)
search_code(query="planner")
grep_regex(pattern="def .*node", path=".")
inspect_symbol(symbol="State")
inspect_references(symbol="State")
inspect_import_graph()
inspect_call_graph(function="build_graph")
static_check()
lint_typecheck()
run_tests()
run_targeted_tests(target="beginner_agent")
run_typecheck()
run_build()
get_diagnostics()
format_check()
git_status()
git_diff()
git_diff_file(path="tools.py")
patch_plan(path="tools.py", goal="说明修改目标")
validate_patch_plan(patch_plan_id="patch-xxx")
apply_patch_plan(patch_plan_id="patch-xxx")
apply_patch(path="...", old_text="...", new_text="...")
rollback(path="...", content="...")
checkpoint_save(name="before-change", data={...})
checkpoint_load(name="before-change")
secret_scan(path=".")
dependency_inspect()
summarize_file(path="graph.py")
```

### 6. Evaluator / Verifier

位置：

```text
evaluator.py -> evaluator_verifier_node(...)
```

职责：

```text
检查工具结果是否可用
判断 complete / retry / expand / fail
对 blocked / failed / empty 优先用本地规则判断
对 success / partial 再让 LLM 做语义补充
只输出评估结论，不直接写回任务树
```

### Task Committer

位置：

```text
evaluator.py -> task_committer_node(...)
```

职责：

```text
根据 Evaluator 的判断更新 task_tree / agenda
失败时可以重试
需要更多上下文时可以继续拆任务
测试或 lint 失败时，如果存在 patch_history，可以安排 rollback 任务
完成时写入 completed_tasks
生成 pending_memory，交给 Memory Writer 保存
更新 parent_evaluation 和 goal_progress
```

### Parent Task Evaluation

位置：

```text
evaluator.py -> _evaluate_parent_task(...)
```

职责：

```text
子任务完成后，检查父任务下面所有 children 的状态。

如果还有 pending 子任务，父任务是 partial。
如果所有子任务 done，父任务是 done。
如果有失败子任务，父任务可能是 needs_more 或 failed。
```

### Goal Progress

位置：

```text
node_utils.py -> goal_progress_snapshot(...)
```

职责：

```text
根据 task_tree 估算当前目标完成度。
输出 completion_ratio、pending_tasks、missing 等信息。
```

### 7. Memory / Checkpoint

位置：

```text
graph.py -> MemorySaver()
main.py  -> config={"configurable": {"thread_id": "..."}}
```

为什么需要 Memory 节点：

```text
对小白来说，可以先把 State、Checkpoint、Memory 分成三层理解：

1. State
   当前这一次图运行里的工作表。
   节点读取 State，返回局部更新，LangGraph 自动合并。
   它适合保存“本轮正在发生什么”。

2. Checkpoint
   当前这一次运行过程的快照。
   它解决的是“运行到一半能不能恢复”的问题。
   例如长任务中断后，可以从上次状态继续。

3. Memory
   跨多次任务复用的经验库。
   它解决的是“下次遇到类似问题，agent 能不能想起过去经验”的问题。
```

为什么不能只靠 prompt 或 State：

```text
1. LLM 上下文有限
   不能把所有历史任务、所有文件、所有失败记录都塞进 prompt。

2. 长任务会产生很多中间结果
   如果全部放在 State 里，State 会越来越大，后面的节点越来越难读。

3. Code agent 最有价值的是失败经验
   例如：
     这个测试以前失败过，原因是什么？
     这个文件以前修改过，风险在哪里？
     这个工具以前执行失败，是权限问题还是参数问题？

   这些经验不应该随着一次 graph.invoke(...) 结束就消失。

4. 真实工程需要可查询、可审计
   大厂 agent 不只是“会回答”，还要能追踪：
     它记住了什么？
     它为什么选择这个任务？
     它为什么避开某个工具？
     哪次失败影响了这次决策？
```

为什么拆成 Memory Retriever 和 Memory Writer：

```text
Memory Retriever = 做事前读取记忆

它通常放在复杂任务开始前。
作用是把和当前任务相关的历史经验读出来，放进 memory_context。
后面的 Planner、Scheduler、Evaluator 可以参考这些记忆。

Memory Writer = 做事后写入记忆

它通常放在任务执行、评估、提交之后。
作用是把这次任务产生的新经验写成 MemoryRecord。
例如工具执行失败、测试结果、重要文件、风险提醒、修复经验。

为什么要分开？

因为“读历史经验”和“沉淀新经验”是两个不同职责：
  - Retriever 影响下一步怎么做。
  - Writer 负责把这次做事的结果保存下来。

分开以后，图结构更清楚，也更容易替换实现。
例如以后 Retriever 可以升级成向量检索，Writer 可以升级成审计写入。
```

为什么使用结构化 MemoryRecord：

```text
MemoryRecord 不是随便存一段文本，而是把记忆拆成字段：

kind        记忆类型，例如 task_result / tool_failure / note
task_id     哪个任务产生的记忆
title       记忆标题
summary     记忆摘要
tool_name   相关工具
status      success / failed / blocked / partial
paths       相关文件路径
tags        标签
confidence 可信度
metadata    额外结构化信息

这样做的好处是：
  - 可以按 task_id 查
  - 可以按 tool_name 查
  - 可以按 status 查失败经验
  - 可以按 paths 查某个文件的历史记录
  - 可以导出 JSON Schema 给前端、数据库、审计系统使用
```

为什么使用 Postgres + pgvector：

```text
Postgres 负责持久化：
  - agent 重启后记忆还在
  - 可以做索引
  - 可以做审计查询
  - 可以和真实业务数据库集成

pgvector 负责语义相似检索：
  - 当前任务和哪条历史经验语义相似？
  - 当前报错和过去哪次失败相似？
  - 当前文件和哪些历史修改相关？

本项目现在采用 hybrid retrieval：
  - 结构化字段检索：更稳定，适合查精确条件。
  - 向量相似检索：更灵活，适合找语义相近经验。

一句话：
Memory 节点让 agent 从“每次重新开始”升级为“会积累经验的工程系统”。
```

项目里当前包含这些记忆相关组件：

```text
1. LangGraph checkpoint
   MemorySaver 保存图执行过程中的状态。

2. State 里的 memory_notes
   方便你直接在输出 JSON 里观察 agent 记住了哪些任务结果。

3. Memory Retriever / Writer 节点
   memory_retriever_node 负责把短期记忆和持久记忆读成 memory_context。
   memory_writer_node 负责把 pending_memory 标准化成结构化 MemoryRecord。

4. Postgres + pgvector 主记忆库
   Postgres 保存结构化 MemoryRecord。
   pgvector 保存 embedding 向量。
   Memory Retriever 会做规则分数 + 向量相似度的 hybrid retrieval。

5. 本地 fallback memory.jsonl
   位置：beginner_agent/.agent_state/memory/memory.jsonl
   格式：一行一个 JSON。
   只有 Postgres / pgvector 不可用时才兜底写入。
   它不是当前架构的主记忆库。

6. Pydantic MemoryRecord
   memory.py 使用 Pydantic model 校验记忆结构。
   可以导出 JSON Schema，方便前端、数据库、审计系统对齐字段。

7. Memory backend 配置
   默认 backend=postgres。
   正常使用：
   BEGINNER_AGENT_MEMORY_BACKEND=postgres
   DATABASE_URL=postgresql://user:password@host:port/dbname
   如果 Postgres / pgvector 连接失败，会安全回退到 JSONL，
   并在 memory_context.backend_error 里记录原因。
```

### Local Docker Postgres

本项目已经提供本地 Docker Postgres 配置：

```text
docker-compose.yml
.env.example
scripts/check_postgres_memory.py
```

启动 Postgres：

```bash
docker compose up -d postgres
```

默认连接信息：

```text
DATABASE_URL=postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent
BEGINNER_AGENT_MEMORY_BACKEND=postgres
```

验证 memory 后端是否真的写入 Postgres：

```bash
DATABASE_URL=postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent \
uv run python scripts/check_postgres_memory.py
```

PostgresMemoryStore 会自动创建：

```text
beginner_agent_memory 表
idx_beginner_agent_memory_kind
idx_beginner_agent_memory_task_id
idx_beginner_agent_memory_tool_status
idx_beginner_agent_memory_created_at
idx_beginner_agent_memory_tags
```

### Local Vector Memory

本项目现在使用 Postgres + pgvector 作为本地向量数据库：

```text
docker-compose.yml 使用 pgvector/pgvector:pg16
memory.py 自动创建 vector extension
memory.py 按维度自动创建 beginner_agent_memory_embeddings_<dimension> 表
embeddings.py 提供 EmbeddingProvider 抽象
```

重要说明：

```text
Qwen3-ASR-1.7B-bf16 是 ASR 语音识别模型，不是向量数据库，也不是 embedding 模型。

向量数据库：Postgres + pgvector
向量生成：EmbeddingProvider
默认本地模型 provider：OmlxEmbeddingProvider
```

验证 pgvector 是否真的可写可查：

```bash
docker compose up -d postgres
uv run python scripts/check_vector_memory.py
```

默认 embedding 配置：

```text
BEGINNER_AGENT_EMBEDDING_PROVIDER=omlx
BEGINNER_AGENT_EMBEDDING_DIM=1024
```

如果你的 OMLX 提供真正的 embedding 模型和 `/v1/embeddings` 接口，
可以使用本机 Qwen3 embedding 模型：

```text
BEGINNER_AGENT_EMBEDDING_PROVIDER=omlx
BEGINNER_AGENT_EMBEDDING_DIM=1024
OMLX_BASE_URL=http://127.0.0.1:8000/v1
OMLX_API_KEY=local-omlx-key
OMLX_EMBEDDING_MODEL=Qwen3-Embedding-8B-4bit-DWQ
OMLX_EMBEDDING_SEND_DIMENSIONS=true
```

为什么这里推荐 1024 维，而不是直接使用 4096 维：

```text
Qwen3-Embedding-8B 的能力很强，可以输出高维向量。
但是本项目当前使用 Postgres + pgvector 做本地向量库。

对这个阶段来说：
  - 1024 维已经适合 memory 检索和代码片段检索。
  - 向量越短，写入、索引、查询越轻。
  - pgvector 常规 vector 索引更适合 2000 维以内。

所以推荐：
  Qwen3-Embedding-8B-4bit-DWQ 模型
  + BEGINNER_AGENT_EMBEDDING_DIM=1024

这样不是浪费 8B 模型，而是让 8B 模型输出更适合本地工程系统使用的向量。
```

模型下载完成后，先验证 OMLX embedding 接口：

```bash
BEGINNER_AGENT_EMBEDDING_PROVIDER=omlx \
BEGINNER_AGENT_EMBEDDING_DIM=1024 \
OMLX_BASE_URL=http://127.0.0.1:8000/v1 \
OMLX_API_KEY=local-omlx-key \
OMLX_EMBEDDING_MODEL=Qwen3-Embedding-8B-4bit-DWQ \
uv run python scripts/check_omlx_embedding.py
```

再验证 Postgres + pgvector + Qwen embedding 的完整写入和检索链路：

```bash
BEGINNER_AGENT_MEMORY_BACKEND=postgres \
DATABASE_URL=postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent \
BEGINNER_AGENT_EMBEDDING_PROVIDER=omlx \
BEGINNER_AGENT_EMBEDDING_DIM=1024 \
OMLX_BASE_URL=http://127.0.0.1:8000/v1 \
OMLX_API_KEY=local-omlx-key \
OMLX_EMBEDDING_MODEL=Qwen3-Embedding-8B-4bit-DWQ \
uv run python scripts/check_vector_memory.py
```

## 关键 State 字段

```text
task_type            Router 输出的任务类型
risk_level           Router 输出的风险等级
needs_tool           是否需要工具
route_reason         Router 判断原因
next_action          当前复杂 agent 下一步动作

root_task_id         根任务 ID
task_tree            任务树
agenda               待处理 task_id 队列
current_task_id      当前任务 ID
completed_tasks      已完成任务记录
memory_notes         可观察的轻量记忆
memory_context       Memory Retriever 读取出的相关记忆上下文
pending_memory       等待 Memory Writer 保存的本轮任务经验
patch_history        apply_patch 成功后的修改历史
human_approvals      人工审批结果，控制写工具能否执行
pending_approval     当前等待人工确认的工具调用

planner_reason       Planner 判断原因
plan_validation_status  valid / invalid / none
plan_validation_reason  Plan Validator 判断原因
policy_decision      allow / ask / deny
policy_reason        工具权限判断原因
evaluation_decision  complete / retry / expand / fail
evaluation_reason    Evaluator 判断原因
parent_evaluation    父任务状态评估
goal_progress        目标进度评估

tool_name            当前工具名
tool_args            当前工具参数
tool_result          最近一次工具结果
tool_result_data     Pydantic ToolResult 的 dict，保存结构化工具结果
tool_result_status   success / failed / blocked / empty / partial / none
execution_status     Executor 视角的执行状态，例如 completed / completed_over_budget
active_execution     当前或最近一次执行尝试摘要
execution_attempts   所有工具执行尝试记录，会自动追加
max_tool_duration_ms 单次工具执行预算，当前用于标记超预算
execution_monitor_status  Watchdog 对执行过程的观察状态
execution_monitor_reason  Watchdog 的观察原因
recovery_action      Recovery Planner 给出的恢复动作
recovery_reason      Recovery Planner 的恢复原因
partial_result       已经拿到的部分结果，供总结和下次继续
resume_hint          下次继续任务时的建议入口

max_steps            最大循环轮数
max_depth            最大任务树深度
max_total_tasks      最大任务总数
max_task_retries     单个任务最大重试次数
allowed_tools        工具白名单
permission_policy    工具权限策略
```

## 模块文件拆分

```text
main.py            程序入口，创建初始 State 并调用 graph.invoke(...)
graph.py           组装 LangGraph 节点、边、条件边和 MemorySaver
state.py           定义复杂 agent 共享的 State
nodes.py           节点导出入口，graph.py 从这里统一导入节点

router.py          Router / Classifier
scheduler.py       Scheduler / Agenda Manager
planner.py         Planner / Decomposer
plan_validator.py  Plan Validator
policy.py          Tool Policy / Permission Layer
executor.py        Executor
execution_monitor.py Execution Monitor / Watchdog
recovery.py        Recovery Planner
evaluator.py       Evaluator / Verifier
simple_nodes.py    search / write / chat / summarize 简单分支
node_utils.py      多个节点共享的常量和工具函数

tools.py           工具统一入口，继续向外暴露 run_tool / validate_tool_request
tooling/           具体工具实现目录，按读取、搜索、诊断、git、patch 等分类
llm_client.py      本地 OMLX 模型调用封装
```

## 运行

在仓库根目录运行：

```bash
uv run --no-dev --project libs/langgraph python beginner_agent/main.py
```

## 安全边界

当前项目只提供只读工具：

```text
1. 只能访问 beginner_agent 目录内部
2. 拒绝绝对路径，例如 /Users/...
3. 拒绝路径穿越，例如 ../../secret.txt
4. 只允许读取 .py / .md / .txt
5. 不提供任意写入、删除、移动能力
6. 不提供开放式 shell command runner
7. git_diff 只是固定执行 git diff -- beginner_agent，用来观察当前项目差异
8. apply_patch 只能做精确 old_text -> new_text 替换
9. rollback 只能恢复 patch_history 中保存过的内容
10. 写工具默认 ask，不会在没有 human_approvals 的情况下自动执行
```

这意味着它可以作为学习复杂 agent 的基础，但不会修改你的 Mac 文件。
