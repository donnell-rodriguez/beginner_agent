# Remote Worker TODO

中文注释：
当前 `async_job_store.py` 提供的是本地 SQLite job contract。
`async_jobs.py` 已经能根据 job_id 轮询，但 Executor 还没有真正把任务提交到远程 worker。

后续生产级升级路线：

1. Job submission
   - Executor 遇到长任务工具时，不直接同步运行。
   - 创建 job payload：
     - run_id
     - task_id
     - tool_name
     - tool_args
     - sandbox_mode
     - timeout_seconds
   - 提交给 worker queue。

2. Queue backend
   - 本地开发：SQLite queue。
   - 中小规模：Redis / Postgres queue。
   - 大规模：Kafka / SQS / PubSub / 内部任务系统。

3. Worker result contract
   - status：queued / running / success / failed / cancelled / timeout。
   - output：给人看的结果。
   - result_data：机器可读结构化结果。
   - logs：stdout/stderr。
   - artifacts：产物路径。
   - changed_files：修改文件。

4. Cancellation and timeout
   - Async Job Waiter 超时后写 timeout。
   - Worker 收到 cancellation 后停止执行。
   - Sandbox 清理临时目录/容器。

5. Retry strategy
   - retryable=true 的失败可以进入 Recovery Planner。
   - 非 retryable 的失败进入 Evaluator，总结 blocked 原因。

6. Observability
   - 每个 job 写 metrics：
     - queued_at
     - started_at
     - finished_at
     - duration_ms
     - worker_id
     - sandbox_id
