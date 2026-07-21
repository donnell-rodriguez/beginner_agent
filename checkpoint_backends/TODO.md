# Checkpoint Backends TODO

中文注释：
当前 `checkpointing.py` 已经支持 memory / postgres。
`checkpoint_node.py` 会把 backend、run_id、恢复合同写进 State。

后续生产级升级路线：

1. End-to-end recovery test
   - 用固定 thread_id 启动 graph。
   - 在 Approval Interrupt 或 Async Job Waiter 暂停。
   - 进程退出。
   - 重新 build_graph。
   - 用同一个 thread_id 恢复。
   - 确认 task_tree / agenda / messages 没丢。

2. Checkpoint backend matrix
   - memory：本地教学。
   - sqlite：轻量单机持久化。
   - postgres：本地/团队生产。
   - redis：短期高速状态。

3. Migration strategy
   - checkpoint 表结构版本记录。
   - 升级 LangGraph checkpoint 包前先跑恢复测试。
   - 保留旧 checkpoint 到 TTL 后再清理。

4. Thread and run mapping
   - thread_id 表示一次可恢复会话。
   - run_id 表示一次 agent 运行。
   - 一个 thread 可以有多次 resume。
   - Artifact / Observability 应同时记录 thread_id 和 run_id。

5. Failure handling
   - Postgres 连接失败时不要静默 fallback 到 memory。
   - 明确报错，让用户知道长任务无法恢复。
   - 后续可以增加 read-only degraded mode。
