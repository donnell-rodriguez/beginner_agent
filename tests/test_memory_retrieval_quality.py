from __future__ import annotations

from beginner_agent.memory.retrieval import _rerank_memory_candidates
from beginner_agent.state_factory import create_initial_state


def test_reranker_prefers_relevant_reliable_memory() -> None:
    state = create_initial_state("请帮我修复 memory.py 里面的 Postgres 迁移问题")
    state["current_task_id"] = "task-1"
    state["tool_name"] = "read_file"
    state["task_tree"] = {
        "task-1": {
            "args": {"path": "memory.py"},
        }
    }

    relevant = {
        "id": "relevant",
        "title": "Postgres memory migration repair",
        "summary": "修复 memory.py 的 migration、schema version 和 rollback。",
        "tool_name": "read_file",
        "paths": ["memory.py"],
        "quality_score": 0.95,
        "trust_score": 0.95,
        "importance": 0.9,
        "decay_score": 0.0,
        "validity_status": "active",
        "created_at": "2026-07-21T00:00:00+00:00",
        "rule_score": 10,
    }
    irrelevant = {
        "id": "irrelevant",
        "title": "Frontend color palette note",
        "summary": "按钮颜色和 landing page 排版。",
        "tool_name": "none",
        "paths": ["ui.css"],
        "quality_score": 0.4,
        "trust_score": 0.4,
        "importance": 0.2,
        "decay_score": 0.8,
        "validity_status": "active",
        "created_at": "2020-01-01T00:00:00+00:00",
        "rule_score": 0,
    }

    ranked = _rerank_memory_candidates([irrelevant, relevant], state)

    assert ranked
    assert ranked[0]["id"] == "relevant"
    assert ranked[0]["rerank_decision"] == "include"
    if len(ranked) > 1:
        assert ranked[0]["rerank_score"] > ranked[-1]["rerank_score"]
    assert all(record["id"] != "irrelevant" for record in ranked[:1])


def test_pinned_memory_survives_low_score_filter() -> None:
    state = create_initial_state("普通查询")
    pinned = {
        "id": "pinned",
        "title": "用户偏好",
        "summary": "用户喜欢中文注释。",
        "tool_name": "none",
        "paths": [],
        "quality_score": 0.1,
        "trust_score": 0.1,
        "importance": 0.1,
        "decay_score": 1.0,
        "validity_status": "active",
        "created_at": "2020-01-01T00:00:00+00:00",
        "rule_score": 0,
        "pinned": True,
    }

    ranked = _rerank_memory_candidates([pinned], state)

    assert ranked[0]["id"] == "pinned"
    assert ranked[0]["rerank_decision"] == "include"
