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

后续 Dashboard / Admin UI 可以继续补：

- 登录鉴权：本地 token、管理员账号、只读权限。
- RBAC：区分普通用户、项目管理员、平台管理员。
- 分页游标：替代当前简单 limit。
- 审计页面：展示某条 memory 为什么保存、何时被检索、影响了哪些任务。
- contradiction 时间线：展示同一个 contradiction_key 如何从旧记忆演化到新记忆。
- failure pattern 页面：聚合同类失败、不可重试失败、环境问题和成功修复路径。
- file memory 页面：按文件查看相关任务、补丁、测试失败和项目理解记忆。
- pinned preference 页面：查看长期用户偏好和项目偏好，支持人工禁用或覆盖。
- 敏感信息审批：`include_sensitive=true` 需要管理员确认和额外审计。
- 指标看板：memory 写入量、命中率、过期率、rejected 率、失败复用率。
- OpenAPI 导出：用于给前端或其他 agent 自动发现查询能力。
