# Memory Query API / Dashboard TODO

当前已经实现 FastAPI 只读查询接口：

- `GET /health`
- `GET /memories`
- `GET /memories/{memory_id}`
- `GET /memories/{memory_id}/why`
- `GET /memories/{memory_id}/usage`
- `GET /audit`
- `GET /contradictions/{contradiction_key}`
- `GET /pinned`
- `GET /failures/patterns`
- `GET /files/{file_path:path}/memories`
- `GET /runs/{run_id}/lineage`

当前已经补上的生产治理能力：

- auth：支持 Bearer token / `X-API-Key`，可通过 env 开关强制开启。
- RBAC：区分 `memory_reader`、`audit_reader`、`sensitive_reader`、`admin`。
- request_id：每个请求返回 `request_id`，并写入 response header。
- rate limit：进程内基础限流，可通过 env 配置。
- tenant isolation：按 tenant/workspace/project/user 过滤 memory 可见性。
- cursor pagination：列表接口返回 `page.next_cursor`。
- 审计查询权限：audit / why / usage / lineage / telemetry 需要 `audit_reader`。
- 敏感内容审批：`include_sensitive=true` 需要 `sensitive_reader/admin` 或审批 token。
- 敏感访问审计：访问 confidential / secret memory 会写入 audit event。

后续 Dashboard / Admin UI 可以继续补：

- OAuth/OIDC：替换本地 token，接入真实用户登录和 token introspection。
- 分布式限流：用 Redis / API Gateway 替代当前进程内限流。
- 更严格的 keyset pagination：用 `created_at + id` 替代当前 offset cursor。
- 审计页面：展示某条 memory 为什么保存、何时被检索、影响了哪些任务。
- contradiction 时间线：展示同一个 contradiction_key 如何从旧记忆演化到新记忆。
- failure pattern 页面：聚合同类失败、不可重试失败、环境问题和成功修复路径。
- file memory 页面：按文件查看相关任务、补丁、测试失败和项目理解记忆。
- pinned preference 页面：查看长期用户偏好和项目偏好，支持人工禁用或覆盖。
- 敏感信息审批 UI：把审批 token 升级成审批工单、审批人身份、过期时间和撤销能力。
- 审计日志不可篡改：把 audit event 写入 append-only store 或外部审计系统。
- 指标看板：memory 写入量、命中率、过期率、rejected 率、失败复用率。
- run lineage 页面：按 run_id 查看 checkpoint、memory、tool result、audit event 和最终状态。
- OpenAPI 导出：用于给前端或其他 agent 自动发现查询能力。
