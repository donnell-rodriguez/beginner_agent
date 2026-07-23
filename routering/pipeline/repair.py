from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import ValidationError

from ...node_utils import json_loads_from_model
from ..failure_policy import RouterFailurePolicy
from ..governance import router_stage_model
from ..prompts import RouterPromptSpec
from .models import ROUTER_DECISION_FIELDS, RepairInfo, StageModelT
from .runtime import stage_max_tokens, stage_timeout_seconds


# 中文注释：
# repair.py 只负责“模型输出不符合 schema 时如何修复”。
# 它不判断任务类型，也不做风险策略，只处理 JSON/schema 可靠性问题。


def parse_stage_model_with_repair(
    response: str,
    *,
    model_cls: type[StageModelT],
    required_fields: set[str],
    stage_title: str,
    schema_hint: str,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    failure_policy: RouterFailurePolicy,
    max_tokens_env: str,
) -> tuple[StageModelT, RepairInfo]:
    """解析子阶段输出，失败时做有限 JSON repair。"""

    try:
        return model_cls.model_validate(_stage_payload(response, required_fields)), RepairInfo(
            final_response=response
        )
    except (ValueError, json.JSONDecodeError, AttributeError, ValidationError) as exc:
        if not failure_policy.repair_retry_enabled or failure_policy.max_repair_attempts <= 0:
            raise
        raw_invalid_response = response
        validation_error_type = type(exc).__name__
        last_error: Exception = exc
        for attempt in range(1, failure_policy.max_repair_attempts + 1):
            repaired = _call_repair_router(
                stage_title=stage_title,
                schema_hint=schema_hint,
                original_response=raw_invalid_response,
                validation_error=f"{type(last_error).__name__}: {last_error}",
                prompt=prompt,
                chat_completion=chat_completion,
                max_tokens_env=max_tokens_env,
            )
            try:
                parsed = model_cls.model_validate(_stage_payload(repaired, required_fields))
            except (ValueError, json.JSONDecodeError, AttributeError, ValidationError) as repair_exc:
                last_error = repair_exc
                continue
            return parsed, RepairInfo(
                attempt_count=attempt,
                success=True,
                raw_invalid_response=raw_invalid_response,
                validation_error_type=validation_error_type,
                final_response=repaired,
            )
        raise last_error


def _stage_payload(response: str, required_fields: set[str]) -> dict[str, object]:
    data = json_loads_from_model(response)
    if not isinstance(data, dict):
        raise ValueError("Router 子阶段输出不是 JSON object。")
    extra = set(data) - ROUTER_DECISION_FIELDS
    if extra:
        raise ValueError(f"Router 子阶段输出包含未治理字段：{sorted(extra)}")
    missing = required_fields - set(data)
    if missing:
        raise ValueError(f"Router 子阶段缺少字段：{sorted(missing)}")
    return {
        key: value
        for key, value in data.items()
        if key in required_fields or key in {"reason", "confidence"}
    }


def _call_repair_router(
    *,
    stage_title: str,
    schema_hint: str,
    original_response: str,
    validation_error: str,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    max_tokens_env: str,
) -> str:
    """请求模型修复 Router 子阶段 JSON。"""

    model = router_stage_model(f"{stage_title} JSON Repair")
    kwargs = {
        "temperature": 0,
        "max_tokens": stage_max_tokens(max_tokens_env, prompt.max_tokens),
        # 中文注释：
        # repair 也是 Router 主路径的一部分，所以也必须有硬超时。
        # 否则“修复坏 JSON”这个补救动作反而可能把入口节点卡住。
        "timeout_seconds": stage_timeout_seconds("BEGINNER_AGENT_ROUTER_REPAIR_TIMEOUT_MS"),
    }
    if model:
        kwargs["model"] = model
    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    f"{prompt.template}\n\n"
                    f"{stage_title} JSON Repair：你只负责把模型输出修成合法 JSON。\n"
                    "不要解释，不要改变原始语义，不要添加未要求字段。\n"
                    f"目标 schema 示例：{schema_hint}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始输出：\n{original_response}\n\n"
                    f"校验错误：\n{validation_error}\n\n"
                    "请只返回修复后的 JSON。"
                ),
            },
        ],
        **kwargs,
    )
