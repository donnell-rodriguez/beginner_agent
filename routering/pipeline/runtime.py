from __future__ import annotations

import os
from collections.abc import Callable

from ...config import load_project_env
from ..model_strategy import RouterModelTier, select_router_stage_model
from ..prompts import RouterPromptSpec


# 中文注释：
# 这里放子 Router 阶段共用的 LLM 调用逻辑。
# 后续如果每个阶段要配置不同模型、超时、预算，可以优先扩展这里。


def call_stage_router(
    text: str,
    *,
    prompt: RouterPromptSpec,
    chat_completion: Callable[..., str],
    stage_title: str,
    instruction: str,
    max_tokens_env: str,
    timeout_ms_env: str,
    model_tier: RouterModelTier = "primary",
) -> str:
    """调用某一个 Router 子阶段。"""

    model_selection = select_router_stage_model(stage_title, tier=model_tier)
    kwargs = {
        "temperature": prompt.temperature,
        "max_tokens": stage_max_tokens(max_tokens_env, prompt.max_tokens),
        # 中文注释：
        # Router 是入口节点，不能因为某个 LLM 阶段慢而拖住整个图。
        # 这里把 .env 中的毫秒预算转成 urllib 需要的秒数，
        # 然后传给 llm_client.chat_completion 做真正的硬超时。
        "timeout_seconds": stage_timeout_seconds(timeout_ms_env),
    }
    if model_selection.model:
        kwargs["model"] = model_selection.model
    return chat_completion(
        [
            {
                "role": "system",
                "content": f"{prompt.template}\n\n{stage_title}：{instruction}",
            },
            {"role": "user", "content": text},
        ],
        **kwargs,
    )


def stage_max_tokens(name: str, default: int) -> int:
    """读取某个 Router 子阶段的 max_tokens。"""

    load_project_env()
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def stage_timeout_seconds(name: str, default_ms: int = 1000) -> float:
    """读取某个 Router 子阶段的硬超时，并转换成秒。

    中文注释：
    .env 里使用毫秒是因为入口层预算通常按 ms 管理；
    urllib.request.urlopen(...) 使用秒，所以这里统一做换算。

    例子：

        BEGINNER_AGENT_ROUTER_INTENT_TIMEOUT_MS=1200

    会变成：

        timeout_seconds = 1.2
    """

    load_project_env()
    try:
        value = int(os.getenv(name, str(default_ms)))
    except ValueError:
        value = default_ms
    if value <= 0:
        value = default_ms
    return value / 1000
