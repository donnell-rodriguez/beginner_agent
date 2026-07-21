from __future__ import annotations

import hashlib
import json
from typing import Any

from .memory_models import MemoryAuditAction, MemoryAuditEvent
from .memory_policy import _safe_memory_value

def _stable_audit_id(event: dict[str, Any]) -> str:
    """为 audit event 生成稳定 ID。"""

    raw = json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _build_audit_event(
    *,
    action: MemoryAuditAction,
    memory_id: str,
    reason: str,
    backend: str,
    metadata: dict[str, Any] | None = None,
) -> MemoryAuditEvent:
    """构造标准化 memory audit event。"""

    safe_metadata = _safe_memory_value(metadata or {})
    raw_event = {
        "action": action,
        "memory_id": memory_id,
        "reason": reason,
        "backend": backend,
        "metadata": safe_metadata,
    }
    return MemoryAuditEvent(
        id=_stable_audit_id(raw_event),
        action=action,
        memory_id=memory_id,
        reason=reason,
        backend=backend,
        metadata=safe_metadata,
    )
