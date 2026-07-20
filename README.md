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
```

ToolResult 的作用：

```text
tool_result       给人看的字符串输出
tool_result_data  给机器看的结构化结果

Evaluator / Memory / Audit 后续应该优先读取 tool_result_data。
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

项目里有两种记忆：

```text
1. LangGraph checkpoint
   MemorySaver 保存图执行过程中的状态。

2. State 里的 memory_notes
   方便你直接在输出 JSON 里观察 agent 记住了哪些任务结果。

3. Memory Retriever / Writer 节点
   memory_retriever_node 负责把短期记忆和持久记忆读成 memory_context。
   memory_writer_node 负责把 pending_memory 标准化成结构化 MemoryRecord。

4. 本地持久化 memory.jsonl
   位置：beginner_agent/.agent_state/memory/memory.jsonl
   格式：一行一个 JSON。
   内容包含 kind、task_id、tool_name、status、paths、tags、confidence、metadata。
   当前使用规则检索；后续可以升级成 embedding 向量检索。

5. Pydantic MemoryRecord
   memory.py 使用 Pydantic model 校验记忆结构。
   可以导出 JSON Schema，方便前端、数据库、审计系统对齐字段。

6. 可选 Postgres 后端
   默认 backend=jsonl。
   如果要启用 Postgres：
   BEGINNER_AGENT_MEMORY_BACKEND=postgres
   DATABASE_URL=postgresql://user:password@host:port/dbname
   如果 Postgres 连接失败，会安全回退到 JSONL，并在 memory_context.backend_error 里记录原因。
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
