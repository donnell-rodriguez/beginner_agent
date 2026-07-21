# Observability Backends TODO

中文注释：
当前 `observability_store.py` 把每轮 loop 的 report 写入本地 SQLite。
这已经是真实持久化，但不是完整 metrics/logs/traces 平台。

后续生产级升级路线：

1. Structured logs
   - 每个节点进入/退出都写 JSON log。
   - 包含 run_id、thread_id、node_name、duration_ms、status。

2. Metrics
   - loop step count。
   - tool success/failure count。
   - approval wait time。
   - sandbox duration。
   - async job queue latency。
   - evaluator retry rate。

3. Tracing
   - 每次 run 是一个 trace。
   - 每个 node 是一个 span。
   - 每次 tool call 是子 span。
   - 可接 OpenTelemetry / LangSmith / 自建 tracing。

4. Alerting
   - max_steps hit rate 过高报警。
   - rollback rate 过高报警。
   - approval timeout 过高报警。
   - sandbox failure 过高报警。

5. Query interface
   - 按 run_id 查看完整运行轨迹。
   - 按 tool_name 查看失败趋势。
   - 按 risk_level 查看审批情况。
   - 按 project root 查看长期质量。
