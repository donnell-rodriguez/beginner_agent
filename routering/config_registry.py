from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..config import load_project_env


# 中文注释：
# config_registry.py 是 Router 配置中心的本地实现。
#
# 现在它读取一个本地 JSON manifest；
# 以后可以把 RouterConfigRegistry 换成远程服务客户端。
# 其它模块只关心“给我当前应该使用的 prompt/rules/security/model 配置”，
# 不需要知道配置来自本地文件还是配置中心服务。

RouterConfigArtifactType = Literal["prompt", "rules", "security_policy", "model_strategy"]
RouterConfigArtifactStatus = Literal["draft", "candidate", "active", "rollback", "disabled"]


@dataclass(frozen=True)
class RouterConfigArtifact:
    """一个可发布/回滚/灰度的 Router 配置 artifact。"""

    artifact_id: str
    artifact_type: RouterConfigArtifactType
    version: str
    status: RouterConfigArtifactStatus
    path: str = ""
    rollback_from: str = ""
    rollout_percent: int = 100
    experiment_group: str = "control"
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def enabled_for(self, text: str) -> bool:
        if self.status in {"draft", "disabled"}:
            return False
        if self.status in {"active", "rollback"}:
            return True
        return _rollout_enabled(text, self.artifact_id, self.rollout_percent)

    def resolved_path(self) -> Path | None:
        if not self.path:
            return None
        resolved = Path(self.path).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "version": self.version,
            "status": self.status,
            "path": self.path,
            "rollback_from": self.rollback_from,
            "rollout_percent": self.rollout_percent,
            "experiment_group": self.experiment_group,
            "env": self.env,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RouterConfigRegistry:
    """Router 配置中心快照。"""

    version: str
    source: str
    artifacts: tuple[RouterConfigArtifact, ...]

    def select(
        self,
        artifact_type: RouterConfigArtifactType,
        *,
        text: str,
    ) -> RouterConfigArtifact | None:
        """选择当前输入应使用的 artifact。"""

        candidates = [
            artifact
            for artifact in self.artifacts
            if artifact.artifact_type == artifact_type and artifact.enabled_for(text)
        ]
        for status in ("rollback", "candidate", "active"):
            for artifact in candidates:
                if artifact.status == status:
                    return artifact
        return None

    def selected_artifacts(self, *, text: str) -> tuple[RouterConfigArtifact, ...]:
        """返回本次 Router 可能使用到的 registry artifacts。"""

        result: list[RouterConfigArtifact] = []
        for artifact_type in ("prompt", "rules", "security_policy", "model_strategy"):
            artifact = self.select(artifact_type, text=text)
            if artifact is not None:
                result.append(artifact)
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


def load_router_config_registry() -> RouterConfigRegistry | None:
    """加载 Router 配置中心 manifest。

    中文注释：
    通过 .env 指定：

        BEGINNER_AGENT_ROUTER_CONFIG_REGISTRY_PATH=.agent_state/router/config_registry.json

    如果没有设置，就返回 None，保持现有 .env + 本地 JSON 配置方式。
    """

    load_project_env()
    path = os.getenv("BEGINNER_AGENT_ROUTER_CONFIG_REGISTRY_PATH", "").strip()
    if not path:
        return None
    resolved = _resolve_path(path)
    if not resolved.exists():
        return None
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    artifacts = tuple(
        artifact
        for item in data.get("artifacts", [])
        if isinstance(item, dict)
        for artifact in [_artifact_from_dict(item)]
        if artifact is not None
    )
    return RouterConfigRegistry(
        version=str(data.get("version", "router-config-registry-local-v1")).strip()
        or "router-config-registry-local-v1",
        source=str(resolved),
        artifacts=artifacts,
    )


def resolve_router_config_artifact(
    artifact_type: RouterConfigArtifactType,
    *,
    text: str = "",
) -> RouterConfigArtifact | None:
    """从配置中心选择某类 artifact。"""

    registry = load_router_config_registry()
    if registry is None:
        return None
    return registry.select(artifact_type, text=text)


def router_config_registry_snapshot(*, text: str) -> dict[str, Any]:
    """生成本次 Router 使用的配置中心快照摘要。"""

    registry = load_router_config_registry()
    if registry is None:
        return {
            "enabled": False,
            "version": "",
            "source": "",
            "selected_artifacts": [],
        }
    return {
        "enabled": True,
        "version": registry.version,
        "source": registry.source,
        "selected_artifacts": [
            artifact.as_dict() for artifact in registry.selected_artifacts(text=text)
        ],
    }


def registry_env_value(
    artifact_type: RouterConfigArtifactType,
    key: str,
    *,
    text: str = "",
) -> str:
    """读取某类 artifact 携带的 env 值。"""

    artifact = resolve_router_config_artifact(artifact_type, text=text)
    if artifact is None:
        return ""
    return artifact.env.get(key, "").strip()


def _artifact_from_dict(data: dict[str, Any]) -> RouterConfigArtifact | None:
    artifact_type = data.get("type", data.get("artifact_type"))
    status = data.get("status", "active")
    if artifact_type not in {"prompt", "rules", "security_policy", "model_strategy"}:
        return None
    if status not in {"draft", "candidate", "active", "rollback", "disabled"}:
        return None
    artifact_id = str(data.get("id", data.get("artifact_id", ""))).strip()
    if not artifact_id:
        stable = hashlib.sha256(
            json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        artifact_id = f"{artifact_type}.{stable}"
    return RouterConfigArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        version=str(data.get("version", "local-v1")).strip() or "local-v1",
        status=status,
        path=str(data.get("path", "")).strip(),
        rollback_from=str(data.get("rollback_from", "")).strip(),
        rollout_percent=_bounded_percent(data.get("rollout_percent", 100)),
        experiment_group=str(data.get("experiment_group", "control")).strip() or "control",
        env={str(key): str(value) for key, value in dict(data.get("env", {})).items()},
        metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
    )


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _bounded_percent(value: Any) -> int:
    try:
        return max(0, min(int(value), 100))
    except (TypeError, ValueError):
        return 100


def _rollout_enabled(text: str, artifact_id: str, rollout_percent: int) -> bool:
    if rollout_percent >= 100:
        return True
    if rollout_percent <= 0:
        return False
    raw = f"{artifact_id}:{text}".encode("utf-8")
    bucket = int(hashlib.sha256(raw).hexdigest()[:8], 16) % 100
    return bucket < rollout_percent
