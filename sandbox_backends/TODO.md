# Sandbox Backends TODO

中文注释：
当前 `sandbox.py` 的可运行 backend 是 `local_controlled`。
它依赖 ToolSpec、Pydantic schema、安全路径、白名单命令和 patch plan 治理。
这已经比“任意 shell”安全，但还不是真正容器/虚拟机隔离。

后续生产级升级路线：

1. Docker sandbox
   - 为每次 run_id 创建临时容器。
   - 只挂载 active project 的临时副本。
   - 禁止挂载用户 HOME。
   - 默认禁网，只有需要联网的工具才开白名单网络。
   - 执行完成后导出 diff、logs、artifacts。

2. Firecracker / microVM sandbox
   - 用更强隔离运行高风险代码。
   - 每个任务使用干净 rootfs。
   - 限制 CPU、内存、磁盘、网络。
   - 支持超时强杀和资源审计。

3. Remote sandbox service
   - 将任务提交到远程 runner。
   - 返回 job_id 给 `Async Job Waiter`。
   - 远程 worker 写入 async job store 或消息队列。
   - 本地图只轮询状态，不直接执行高风险命令。

4. Sandbox policy
   - low risk：local_controlled。
   - medium risk：Docker sandbox。
   - high risk：microVM + human approval。
   - write tools：必须先审批，再进入 sandbox。

5. Required reports
   - command stdout/stderr。
   - exit_code。
   - duration_ms。
   - resource usage。
   - changed_files。
   - generated_artifacts。
   - sandbox image / runner version。
