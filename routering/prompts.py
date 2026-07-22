from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_project_env


# 中文注释：
# prompts.py 是 Router prompt registry。
#
# 生产级系统通常不会把 prompt 直接写死在 router.py 里，
# 而是把 prompt 当成“可治理资产”：
# - version：知道当前用的是哪一版。
# - experiment_group：支持 A/B 或灰度实验。
# - rollout_percent：控制某个实验组覆盖多少请求。
# - rollback path：新 prompt 出问题时可以快速回滚。
# - source：知道 prompt 来自内置默认值还是外部配置文件。


BUILTIN_ROUTER_PROMPT_VERSION = "router-prompt-builtin-v1"
BUILTIN_ROUTER_PROMPT_EXPERIMENT = "control"
BUILTIN_ROUTER_SYSTEM_PROMPT = (
    "你是 agent 的 Router / Classifier。"
    "请判断任务类型、风险等级、是否需要工具。"
    "task_type 只能是 search、write、chat、agent。"
    "risk_level 只能是 low、medium、high。"
    "如果用户需要读取文件、理解项目、查看源码，task_type=agent，needs_tool=true。"
    "如果用户要求修复代码、修改代码、运行测试、查看 diff，也应该 task_type=agent。"
    "如果用户要求修改、删除、执行命令，risk_level=high。"
    "只返回严格 JSON，不要解释。"
    '格式：{"task_type":"agent","risk_level":"low",'
    '"needs_tool":true,"reason":"一句话原因","confidence":0.8}'
)


@dataclass(frozen=True)
class RouterPromptSpec:
    """Router prompt 的可治理数据结构。"""

    version: str
    template: str
    experiment_group: str = BUILTIN_ROUTER_PROMPT_EXPERIMENT
    source: str = "builtin"
    rollout_percent: int = 100
    temperature: float = 0.0
    max_tokens: int = 240
    rollback_from: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "experiment_group": self.experiment_group,
            "source": self.source,
            "rollout_percent": self.rollout_percent,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "rollback_from": self.rollback_from,
        }


def select_router_prompt(text: str) -> RouterPromptSpec:
    """选择本次 Router 调用使用的 prompt。

    中文注释：
    选择顺序：

    1. BEGINNER_AGENT_ROUTER_PROMPT_ROLLBACK_PATH
       如果设置，优先加载回滚 prompt。

    2. BEGINNER_AGENT_ROUTER_PROMPT_PATH
       正常加载 prompt registry 文件。

    3. builtin prompt
       配置不存在或格式错误时回退到内置 prompt。

    如果配置文件里有 variants，会按 user_input 做稳定 hash，
    在 rollout_percent 范围内选择对应实验组。
    """

    load_project_env()
    rollback_path = os.getenv("BEGINNER_AGENT_ROUTER_PROMPT_ROLLBACK_PATH", "").strip()
    if rollback_path:
        prompt = _load_prompt_file(
            _resolve_path(rollback_path),
            text=text,
            source_prefix="rollback:",
        )
        if prompt is not None:
            return RouterPromptSpec(
                version=prompt.version,
                template=prompt.template,
                experiment_group=prompt.experiment_group,
                source=prompt.source,
                rollout_percent=prompt.rollout_percent,
                temperature=prompt.temperature,
                max_tokens=prompt.max_tokens,
                rollback_from=os.getenv("BEGINNER_AGENT_ROUTER_PROMPT_PATH", "").strip(),
            )

    path = os.getenv("BEGINNER_AGENT_ROUTER_PROMPT_PATH", "").strip()
    if path:
        prompt = _load_prompt_file(_resolve_path(path), text=text)
        if prompt is not None:
            return prompt

    return RouterPromptSpec(
        version=os.getenv("BEGINNER_AGENT_ROUTER_PROMPT_VERSION", BUILTIN_ROUTER_PROMPT_VERSION),
        template=BUILTIN_ROUTER_SYSTEM_PROMPT,
        experiment_group=os.getenv(
            "BEGINNER_AGENT_ROUTER_PROMPT_EXPERIMENT_GROUP",
            BUILTIN_ROUTER_PROMPT_EXPERIMENT,
        ),
        source="builtin",
        temperature=_float_env("BEGINNER_AGENT_ROUTER_PROMPT_TEMPERATURE", 0.0),
        max_tokens=_int_env("BEGINNER_AGENT_ROUTER_PROMPT_MAX_TOKENS", 240),
    )


def _load_prompt_file(
    path: Path,
    *,
    text: str,
    source_prefix: str = "",
) -> RouterPromptSpec | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    selected = _select_prompt_config(data, text)
    if selected is None:
        return None
    template = str(selected.get("template", "")).strip()
    if not template:
        return None

    version = str(selected.get("version", data.get("version", "router-prompt-custom-v1"))).strip()
    group = str(selected.get("experiment_group", data.get("experiment_group", "control"))).strip()
    return RouterPromptSpec(
        version=version or "router-prompt-custom-v1",
        template=template,
        experiment_group=group or "control",
        source=f"{source_prefix}{path}",
        rollout_percent=_bounded_percent(selected.get("rollout_percent", 100)),
        temperature=_float_from_dict(
            selected,
            "temperature",
            _float_from_dict(data, "temperature", 0.0),
        ),
        max_tokens=_int_from_dict(selected, "max_tokens", _int_from_dict(data, "max_tokens", 240)),
        rollback_from=str(data.get("rollback_from", "")).strip(),
    )


def _select_prompt_config(data: dict[str, Any], text: str) -> dict[str, Any] | None:
    variants = data.get("variants")
    if isinstance(variants, list):
        for item in variants:
            if not isinstance(item, dict):
                continue
            rollout = _bounded_percent(item.get("rollout_percent", 0))
            if _rollout_enabled(text, str(item.get("experiment_group", "")), rollout):
                return item
    return data


def _rollout_enabled(text: str, experiment_group: str, rollout_percent: int) -> bool:
    if rollout_percent >= 100:
        return True
    if rollout_percent <= 0:
        return False
    raw = f"{experiment_group}:{text}".encode("utf-8")
    bucket = int(hashlib.sha256(raw).hexdigest()[:8], 16) % 100
    return bucket < rollout_percent


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


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_from_dict(data: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _float_from_dict(data: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default
