from __future__ import annotations

from beginner_agent.privacy_governance import (
    memory_prompt_allowed_by_privacy,
    redact_value_for_memory,
    scan_text_for_privacy,
)


def test_secret_text_is_redacted_and_blocked_from_prompt() -> None:
    report = scan_text_for_privacy("api_key=sk-test-secret-value-12345")

    assert report.sensitivity_level == "secret"
    assert report.prompt_allowed is False
    assert "sk-test-secret-value-12345" not in report.redacted_text
    assert report.findings[0].category == "secret"


def test_pii_text_is_confidential_and_redacted() -> None:
    report = scan_text_for_privacy("contact me at user@example.com")

    assert report.sensitivity_level == "confidential"
    assert report.prompt_allowed is False
    assert "user@example.com" not in report.redacted_text


def test_sensitive_field_value_is_replaced_with_fingerprint() -> None:
    redacted = redact_value_for_memory({"password": "super-secret-password"})

    assert redacted["password"]["redacted"] is True
    assert "super-secret-password" not in str(redacted)


def test_retrieval_only_memory_cannot_enter_prompt() -> None:
    assert (
        memory_prompt_allowed_by_privacy(
            {
                "visibility": "retrieval_only",
                "sensitivity_level": "internal",
                "metadata": {},
            }
        )
        is False
    )
