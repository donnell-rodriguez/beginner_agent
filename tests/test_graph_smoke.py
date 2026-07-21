from __future__ import annotations

from beginner_agent.graph import build_graph


def test_build_graph_smoke() -> None:
    graph = build_graph()

    assert type(graph).__name__ == "CompiledStateGraph"
