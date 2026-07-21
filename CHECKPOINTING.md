# Checkpointing

中文注释：
这个文件解释 beginner_agent 如何使用 Postgres 保存 LangGraph checkpoint。

## 1. Memory 和 Checkpoint 的区别

`memory.py` 保存的是 agent 的长期经验：

```text
这次任务做了什么
哪个工具失败了
哪些文件相关
哪些经验下次可以复用
embedding / pgvector 检索
```

`checkpointing.py` 保存的是 LangGraph 运行状态：

```text
State 当前是什么
运行到了哪个节点
thread_id 对应哪次会话
中断后能不能恢复
```

所以它们不是同一个东西：

```text
Memory     = 经验库
Checkpoint = 运行快照
```

## 2. 当前代码位置

```text
checkpointing.py
graph.py
scripts/check_postgres_checkpoint.py
```

`graph.py` 里不再直接写：

```python
MemorySaver()
```

而是：

```python
build_checkpointer()
```

这样 `graph.py` 只负责图编排，不关心 checkpoint 后端细节。

## 3. 环境变量

生产级原则：

```text
数据库连接串必须来自环境变量，不能写死在源码里。
```

原因是：

```text
本地、测试、生产环境的数据库地址通常不同
数据库账号和密码不应该提交进 Git
缺配置时应该明确报错，而不是静默连接某个默认数据库
```

本地 Docker Postgres 可以这样配置：

```text
DATABASE_URL=postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent
BEGINNER_AGENT_CHECKPOINT_BACKEND=postgres
```

如果你想临时退回内存 checkpoint：

```text
BEGINNER_AGENT_CHECKPOINT_BACKEND=memory
```

如果 checkpoint 想用独立数据库：

```text
BEGINNER_AGENT_CHECKPOINT_DATABASE_URL=postgresql://...
```

如果没有设置 `BEGINNER_AGENT_CHECKPOINT_DATABASE_URL`，
系统会复用 `DATABASE_URL`。

如果两个都没有设置，`checkpointing.py` 会直接报错。
这比源码里内置默认数据库地址更符合生产级配置管理方式。

## 4. .env 自动加载

项目现在有一个统一配置入口：

```text
config.py
```

它负责把本地 `.env` 读入 `os.environ`：

```text
.env
  -> config.py
  -> os.environ
  -> checkpointing.py / memory.py / llm_client.py
```

关系是：

```text
.env.example  配置模板，可以提交 GitHub
.env          本地真实配置，被 .gitignore 忽略，不提交 GitHub
config.py     本地开发时自动加载 .env
```

默认规则：

```text
shell / Docker / CI 已经注入的环境变量优先
.env 只补充缺失的环境变量
```

这样本地开发方便，生产部署也不会被 `.env` 覆盖。

## 5. 启动本地 Postgres

```bash
docker compose up -d postgres
```

当前 docker-compose 使用：

```text
pgvector/pgvector:pg16
数据库：beginner_agent
用户：beginner_agent
端口：55432
```

这同一个 Postgres 可以同时保存：

```text
memory records
memory embeddings
LangGraph checkpoints
```

## 6. 验证 checkpoint

```bash
BEGINNER_AGENT_CHECKPOINT_BACKEND=postgres \
DATABASE_URL=postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent \
uv run python scripts/check_postgres_checkpoint.py
```

成功时会看到：

```text
Postgres checkpoint check passed.
backend=postgres
checkpointer=PostgresSaver
```

第一次运行会自动创建 LangGraph checkpoint 表。

## 7. 为什么生产环境不用 MemorySaver

`MemorySaver` 只存在于当前 Python 进程内。

如果进程退出：

```text
checkpoint 消失
thread_id 无法恢复
长任务不能从中间状态继续
```

Postgres checkpoint 可以解决：

```text
进程重启后仍能找回 checkpoint
多个服务实例可以共享持久化状态
可以审计和备份运行快照
更适合长任务 agent
```

## 7. 大厂通常怎么做

大厂通常会把 checkpoint 当作运行时基础设施：

```text
Graph runtime
  -> Checkpoint backend
  -> Durable database
  -> thread_id / run_id / checkpoint_id
  -> resume / replay / audit
```

常见后端：

```text
Postgres
Redis
SQLite
云数据库
专门的 workflow state store
```

当前 beginner_agent 使用 Postgres，已经比本地 `MemorySaver` 更接近长任务工程形态。
