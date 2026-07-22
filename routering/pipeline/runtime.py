from __future__ import annotations

import os
from collections.abc import Callable

from ...config import load_project_env
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
) -> str:
    """调用某一个 Router 子阶段。"""

    return chat_completion(
        [
            {
                "role": "system",
                "content": f"{prompt.template}\n\n{stage_title}：{instruction}",
            },
            {"role": "user", "content": text},
        ],
        temperature=prompt.temperature,
        max_tokens=stage_max_tokens(max_tokens_env, prompt.max_tokens),
    )


def stage_max_tokens(name: str, default: int) -> int:
    """读取某个 Router 子阶段的 max_tokens。"""

    load_project_env()
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
