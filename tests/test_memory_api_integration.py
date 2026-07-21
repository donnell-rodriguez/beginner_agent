from __future__ import annotations

from fastapi.testclient import TestClient

from beginner_agent.memory_api.app import create_app


def test_memory_api_auth_rbac_and_request_id(monkeypatch) -> None:
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_READER_TOKEN", "reader-token")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_AUDITOR_TOKEN", "auditor-token")

    client = TestClient(create_app())

    health = client.get("/health")
    denied = client.get("/memories")
    reader = client.get("/memories", headers={"Authorization": "Bearer reader-token"})
    audit_denied = client.get("/audit", headers={"Authorization": "Bearer reader-token"})
    audit_ok = client.get("/audit", headers={"Authorization": "Bearer auditor-token"})

    assert health.status_code == 200
    assert health.json()["request_id"]
    assert denied.status_code == 401
    assert denied.json()["request_id"]
    assert reader.status_code == 200
    assert "page" in reader.json()
    assert audit_denied.status_code == 403
    assert audit_ok.status_code == 200


def test_sensitive_api_access_requires_role_or_approval(monkeypatch) -> None:
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_REQUIRE_AUTH", "true")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_READER_TOKEN", "reader-token")
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_SENSITIVE_APPROVAL_TOKEN", "approval-token")

    client = TestClient(create_app())

    denied = client.get(
        "/memories?include_sensitive=true",
        headers={"Authorization": "Bearer reader-token"},
    )
    approved = client.get(
        "/memories?include_sensitive=true",
        headers={
            "Authorization": "Bearer reader-token",
            "X-Memory-Sensitive-Approval": "approval-token",
        },
    )

    assert denied.status_code == 403
    assert approved.status_code == 200


def test_memory_api_cursor_pagination_shape(monkeypatch) -> None:
    monkeypatch.setenv("BEGINNER_AGENT_MEMORY_API_REQUIRE_AUTH", "false")

    client = TestClient(create_app())
    response = client.get("/memories?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert body["page"]["limit"] == 1
    assert "next_cursor" in body["page"]
