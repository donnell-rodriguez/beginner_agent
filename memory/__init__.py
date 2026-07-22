from __future__ import annotations

# 中文注释：
# memory/__init__.py 现在是兼容导出层。
# 真正实现已经按生产级职责拆到多个小模块里：
# - models.py：MemoryRecord / MemoryAuditEvent / 类型定义。
# - store.py：MemoryStore 协议、后端选择、写入治理。
# - postgres_store.py：Postgres + pgvector 实现。
# - jsonl_store.py：JSONL fallback 实现。
# - retrieval.py：召回、ACL 过滤、rerank。
# - policy.py：写入策略、TTL、ACL、隐私字段治理。
# - audit.py：审计事件构造。
# - nodes.py：LangGraph 节点。
# - feedback.py：人工/系统反馈闭环。
# - eval_cases.py：离线 retrieval eval case。
# - rerank_observability.py：rerank telemetry / A/B bucket。
#
# 这样外部旧代码仍然可以 from beginner_agent.memory import MemoryRecord，
# 但单个大文件不再承担所有实现细节。

from .audit import _build_audit_event
from .eval_cases import MemoryEvalCase, append_memory_eval_case, read_memory_eval_cases
from .effectiveness import (
    MemoryUsageEvent,
    append_memory_usage,
    close_memory_usage_loop,
    read_memory_usage,
    record_retrieved_memory_usage,
    summarize_memory_usage,
)
from .feedback import (
    MemoryFeedbackEvent,
    append_memory_feedback,
    feedback_summary_for_memory,
    read_memory_feedback,
)
from .jsonl_store import (
    JsonlMemoryStore,
    _read_jsonl_audit_events,
    _read_jsonl_memory_records,
)
from .models import (
    MemoryAuditAction,
    MemoryAuditEvent,
    MemoryKind,
    MemoryPolicyAction,
    MemoryPolicyDecision,
    MemoryRecord,
    MemoryScope,
    MemoryVisibility,
    MemoryWriterRoute,
    RetentionPolicy,
    SensitivityLevel,
    ValidityStatus,
    memory_record_json_schema,
)
from .nodes import memory_retriever_node, memory_writer_node, route_after_memory_writer
from .observability_sinks import (
    append_memory_observability_event,
    read_memory_observability_events,
)
from .online_eval import (
    MemoryOnlineEvalEvent,
    append_online_eval_event,
    read_online_eval_events,
    record_retrieval_online_eval,
    summarize_online_eval,
)
from .policy import (
    _memory_access_context,
    _safe_memory_value,
    _stable_memory_id,
)
from .postgres_performance import memory_postgres_governance_report
from .postgres_store import PostgresMemoryStore
from .settings import (
    DEFAULT_MEMORY_BACKEND,
    DEFAULT_MEMORY_TTL_DAYS,
    DEFAULT_PROJECT_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_USER_ID,
    DEFAULT_WORKSPACE_ID,
    MAX_INDEXED_VECTOR_DIMENSION,
    MAX_MEMORY_AUDIT_EVENTS,
    MAX_MEMORY_EVAL_CASES,
    MAX_MEMORY_FEEDBACK_EVENTS,
    MAX_MEMORY_RECORDS,
    MAX_MEMORY_RERANK_TELEMETRY_EVENTS,
    MAX_MEMORY_TEXT_CHARS,
    MAX_RERANK_CANDIDATES,
    MAX_RETRIEVED_RECORDS,
    MEMORY_AUDIT_FILE,
    MEMORY_DIR,
    MEMORY_EVAL_CASES_FILE,
    MEMORY_FEEDBACK_FILE,
    MEMORY_FILE,
    MEMORY_RERANK_TELEMETRY_FILE,
    MEMORY_PROMOTION_SUCCESS_THRESHOLD,
    MIN_RERANK_SCORE,
)
from .rerank_observability import (
    read_rerank_telemetry,
    rerank_ab_bucket,
    summarize_rerank_telemetry,
)
from .store import (
    MemoryStore,
    _configured_store,
    _upsert_memory_audit_event,
    _upsert_memory_record,
)

__all__ = [
    "DEFAULT_MEMORY_BACKEND",
    "DEFAULT_MEMORY_TTL_DAYS",
    "DEFAULT_PROJECT_ID",
    "DEFAULT_TENANT_ID",
    "DEFAULT_USER_ID",
    "DEFAULT_WORKSPACE_ID",
    "JsonlMemoryStore",
    "MAX_INDEXED_VECTOR_DIMENSION",
    "MAX_MEMORY_AUDIT_EVENTS",
    "MAX_MEMORY_EVAL_CASES",
    "MAX_MEMORY_FEEDBACK_EVENTS",
    "MAX_MEMORY_RECORDS",
    "MAX_MEMORY_RERANK_TELEMETRY_EVENTS",
    "MAX_MEMORY_TEXT_CHARS",
    "MAX_RERANK_CANDIDATES",
    "MAX_RETRIEVED_RECORDS",
    "MEMORY_AUDIT_FILE",
    "MEMORY_DIR",
    "MEMORY_EVAL_CASES_FILE",
    "MEMORY_FEEDBACK_FILE",
    "MEMORY_FILE",
    "MEMORY_RERANK_TELEMETRY_FILE",
    "MEMORY_PROMOTION_SUCCESS_THRESHOLD",
    "MIN_RERANK_SCORE",
    "MemoryAuditAction",
    "MemoryAuditEvent",
    "MemoryEvalCase",
    "MemoryFeedbackEvent",
    "MemoryKind",
    "MemoryPolicyAction",
    "MemoryPolicyDecision",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStore",
    "MemoryOnlineEvalEvent",
    "MemoryUsageEvent",
    "MemoryVisibility",
    "MemoryWriterRoute",
    "PostgresMemoryStore",
    "RetentionPolicy",
    "SensitivityLevel",
    "ValidityStatus",
    "_build_audit_event",
    "_configured_store",
    "_memory_access_context",
    "_read_jsonl_audit_events",
    "_read_jsonl_memory_records",
    "_safe_memory_value",
    "_stable_memory_id",
    "_upsert_memory_audit_event",
    "_upsert_memory_record",
    "append_memory_eval_case",
    "append_memory_feedback",
    "append_memory_observability_event",
    "append_memory_usage",
    "append_online_eval_event",
    "close_memory_usage_loop",
    "feedback_summary_for_memory",
    "memory_postgres_governance_report",
    "memory_record_json_schema",
    "memory_retriever_node",
    "memory_writer_node",
    "read_memory_eval_cases",
    "read_memory_feedback",
    "read_memory_observability_events",
    "read_memory_usage",
    "read_online_eval_events",
    "read_rerank_telemetry",
    "record_retrieved_memory_usage",
    "record_retrieval_online_eval",
    "rerank_ab_bucket",
    "route_after_memory_writer",
    "summarize_memory_usage",
    "summarize_online_eval",
    "summarize_rerank_telemetry",
]
