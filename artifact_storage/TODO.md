# Artifact Storage TODO

中文注释：
当前 `artifact_store.py` 已经把 manifest 写到本地文件系统。
这比只存在 State 里更进一步，因为进程退出后 artifact 索引仍然存在。

后续生产级升级路线：

1. Object storage
   - 把 manifest、diff、logs、test reports 写到 S3 / MinIO / 内部对象存储。
   - artifact id 使用 run_id + task_id + content hash。
   - 保存 content_type、size、sha256、created_at。

2. Artifact types
   - patch diff。
   - test report。
   - lint/typecheck report。
   - sandbox logs。
   - generated files。
   - final summary。

3. Retention policy
   - 本地开发保留最近 N 天。
   - 生产环境按项目/风险等级配置 TTL。
   - 高风险代码修改 artifact 保留更久，便于审计。

4. Integrity
   - 每个 artifact 记录 sha256。
   - Summary 中引用 artifact id，而不是只写文件路径。
   - 防止文件被修改后无法追溯。

5. API
   - GET /artifacts
   - GET /artifacts/{artifact_id}
   - GET /runs/{run_id}/artifacts
