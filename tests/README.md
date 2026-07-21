# Beginner Agent Test Suite

这个目录把原来的 smoke test 升级成系统化测试套件。

当前覆盖：

- `test_graph_smoke.py`：验证 LangGraph 主图能构建。
- `test_memory_api_integration.py`：验证 Memory API auth、RBAC、request_id、分页、敏感审批。
- `test_memory_compaction.py`：验证 memory compaction 会生成 summary 并 supersede 源记忆。
- `test_memory_lifecycle_scheduler.py`：验证 lifecycle scheduler 的幂等、运行历史和重试。
- `test_memory_migrations.py`：验证 migration 结构和可选 Postgres migration 幂等。
- `test_memory_privacy.py`：验证 secret/PII 脱敏、prompt 禁入。
- `test_memory_retrieval_quality.py`：验证 reranker 的相关性、可靠性和 pinned 规则。

普通测试：

```bash
PYTHONPATH=.. uv run --with pytest pytest -q
```

Postgres 集成测试：

```bash
BEGINNER_AGENT_RUN_POSTGRES_TESTS=true \
DATABASE_URL=postgresql://beginner_agent:beginner_agent@127.0.0.1:55432/beginner_agent \
PYTHONPATH=.. uv run --with pytest pytest -q -m postgres
```

中文注释：
默认不跑真实 Postgres，是为了让本地和 CI 的基础测试稳定。
需要验证数据库 migration / fixture 时，再显式打开 `BEGINNER_AGENT_RUN_POSTGRES_TESTS=true`。
