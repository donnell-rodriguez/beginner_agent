from __future__ import annotations

import os
from typing import Any

from .memory_audit import _build_audit_event
from .memory_models import MemoryWriterRoute, memory_record_json_schema
from .memory_policy import _build_memory_record, _memory_access_context, _memory_policy_for_pending, _safe_memory_value
from .memory_retrieval import _preference_records_for_state, _retrieve_relevant_records, _seed_default_preference_memories
from .memory_store import _upsert_memory_audit_event, _upsert_memory_record
from .node_utils import goal_progress_snapshot
from .preference_memory import default_preference_payloads, merged_preference_context
from .state import State

def memory_retriever_node(state: State) -> dict[str, object]:
    """Memory Retriever：在复杂 agent loop 开始前读取相关记忆。

    中文注释：
    升级后这里不只是读 State.memory_notes。
    它会同时读取并治理：
    - 当前 State 里的短期记忆。
    - JSONL / Postgres 里的持久化记忆。
    - 默认过滤 superseded / rejected / expired 记忆。
    - 先做 pgvector / rule 召回，再用 MemoryReranker 做任务感知重排。

    这样 agent 下一次运行时，也能看到之前任务沉淀下来的经验。
    """

    access_context = _memory_access_context(state)
    preference_seed_report = _seed_default_preference_memories(state)
    state_notes = list(state["memory_notes"])[-5:]
    persisted_records, backend, backend_error = _retrieve_relevant_records(state)
    preference_records, preference_backend, preference_error = _preference_records_for_state(
        state
    )
    preference_context = merged_preference_context(
        default_preference_payloads(access_context),
        preference_records,
    )
    retrieved_ids = [str(record.get("id", "")) for record in persisted_records]
    audit_backend, audit_error = _upsert_memory_audit_event(
        _build_audit_event(
            action="retrieve",
            memory_id=state.get("current_task_id", "") or "memory_retriever",
            reason="Memory Retriever 将相关记忆写入 memory_context。",
            backend=backend,
            metadata={
                "run_id": state["run_id"],
                "retrieved_memory_ids": retrieved_ids,
                "retrieved_count": len(retrieved_ids),
                "backend_error": backend_error,
                "access_context": access_context,
                "preference_seed_report": preference_seed_report,
                "preference_count": preference_context["count"],
            },
        )
    )
    memory_context = {
        "source": "state.memory_notes + postgres/pgvector memory with jsonl fallback",
        "backend": backend,
        "backend_error": backend_error,
        "preference_backend": preference_backend,
        "preference_error": preference_error,
        "preference_seed_report": preference_seed_report,
        "audit_backend": audit_backend,
        "audit_error": audit_error,
        "record_schema": memory_record_json_schema(),
        "governance": {
            "filters": [
                "active",
                "not_expired",
                "scope_matched",
                "acl_visible",
                "prompt_allowed",
            ],
            "access_context": access_context,
            "access_control": [
                "tenant_id",
                "workspace_id",
                "project_id",
                "user_id",
                "visibility",
                "sensitivity_level",
            ],
            "ranking": [
                "vector_distance",
                "rule_score",
                "importance",
                "confidence",
                "quality_score",
                "trust_score",
                "decay_score",
                "pinned",
                "rerank_score",
            ],
            "reranker": [
                "task_aware",
                "semantic_similarity",
                "rule_score",
                "token_overlap",
                "reliability",
                "recency",
                "path_overlap",
                "misleading_risk",
                "failure_memory_signal",
            ],
            "quality_evaluation": [
                "MemoryEvaluator",
                "MemoryQualityScore",
                "MemoryDecay",
                "MemoryTrustScore",
            ],
            "failure_memory_library": [
                "failure_category",
                "stack_signature",
                "retry_class",
                "failure_owner",
                "similar_failure_ids",
                "successful_repair_paths",
                "non_retryable_or_ask_human",
            ],
            "long_term_user_preferences": [
                "default_preference_seed",
                "persisted_preference_memory",
                "user_scope_private",
                "project_scope_project_visible",
                "priority_sorted",
                "override_by_key",
            ],
        },
        "user_preferences": preference_context,
        "state_note_count": len(state_notes),
        "persisted_match_count": len(persisted_records),
        "recent_notes": state_notes,
        "relevant_records": persisted_records,
    }
    return {
        "memory_context": memory_context,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Memory Retriever：读取到 "
                    f"{len(state_notes)} 条短期记忆，"
                    f"{len(persisted_records)} 条相关持久记忆，"
                    f"{preference_context['count']} 条用户/项目偏好，"
                    f"backend={backend}。"
                ),
            }
        ],
    }


def memory_writer_node(state: State) -> dict[str, object]:
    """Memory Writer：把本轮任务经验写入轻量记忆和持久化记忆库。

    中文注释：
    当前是本地生产化的记忆写入链路：
    - State.memory_notes：方便你在运行结果里直接看到。
    - Postgres + pgvector：主记忆库，支持结构化查询和向量检索。
    - memory.jsonl：fallback，只有主记忆库不可用时兜底。
    - MemoryRecord：让记忆有类型、状态、来源、路径和可信度。

    当前已经引入记忆治理：
    - MemoryPolicy 先判断 store / discard。
    - MemoryRecord 带 scope、retention_policy、validity_status、importance。
    - 敏感字段和长文本会被裁剪。
    - Postgres 后端会写 pgvector embedding。
    """

    pending_memory = dict(state["pending_memory"])
    task_tree = dict(state["task_tree"])
    goal_progress = goal_progress_snapshot(state, task_tree)
    update: dict[str, object] = {
        "pending_memory": {},
        "goal_progress": goal_progress,
        "next_action": "finish" if state["done"] else "schedule",
        "messages": [
            {
                "role": "assistant",
                "content": "Memory Writer：没有新的 pending_memory，回到 Scheduler。",
            }
        ],
    }
    if not pending_memory:
        return update

    task_id = str(pending_memory.get("task_id", state["current_task_id"]))
    task = dict(task_tree.get(task_id, {}))
    tool_name = str(task.get("tool") or state["tool_name"] or "none")
    tool_result_status = str(
        pending_memory.get("tool_result_status")
        or task.get("tool_result_status")
        or state["tool_result_status"]
        or "none"
    )
    policy = _memory_policy_for_pending(
        state,
        pending_memory,
        tool_name=tool_name,
        tool_result_status=tool_result_status,
    )
    if policy.action == "discard":
        _upsert_memory_audit_event(
            _build_audit_event(
                action="discard",
                memory_id=task_id,
                reason=policy.reason,
                backend="memory_policy",
                metadata={
                    "tool_name": tool_name,
                    "tool_result_status": tool_result_status,
                    "pending_memory": _safe_memory_value(pending_memory),
                },
            )
        )
        update["messages"] = [
            {
                "role": "assistant",
                "content": f"Memory Writer：跳过写入记忆，原因：{policy.reason}",
            }
        ]
        return update

    record = _build_memory_record(state, pending_memory)
    backend, backend_error, governed_record = _upsert_memory_record(record)
    record_dict = governed_record.model_dump(mode="json")
    if backend_error:
        record_dict["backend_warning"] = backend_error
    record_dict["backend"] = backend
    update["memory_notes"] = [record_dict]
    update["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Memory Writer：写入结构化记忆 "
                f"{governed_record.id}，类型={governed_record.kind}，"
                f"工具={governed_record.tool_name}，"
                f"quality={governed_record.quality_score}，backend={backend}。"
            ),
        }
    ]
    return update


def route_after_memory_writer(state: State) -> MemoryWriterRoute:
    """Memory Writer 后的路由。"""

    if os.getenv("BEGINNER_AGENT_MEMORY_COMPACTION_ENABLED", "true").lower() == "true":
        return "compact"
    if state["done"] or state["next_action"] == "finish":
        return "finish"
    return "schedule"
