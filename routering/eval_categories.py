from __future__ import annotations

from typing import Literal


# 中文注释：
# Router eval category 是“测试样本分层”的受控名称。
#
# 为什么要单独定义？
# - 如果大家随手写字符串，后续统计会出现 normal_chat / normal-chat / chat 三种写法。
# - 生产级 eval 要长期对比趋势，所以分类名称必须稳定。
# - regression gate 可以针对不同类别设置不同阈值。
RouterEvalCategory = Literal[
    "normal_chat_cases",
    "code_agent_cases",
    "tool_needed_cases",
    "high_risk_cases",
    "prompt_injection_cases",
    "secret_pii_cases",
    "ambiguous_cases",
    "regression_cases",
    "general",
]


ROUTER_EVAL_CATEGORIES: tuple[str, ...] = (
    "normal_chat_cases",
    "code_agent_cases",
    "tool_needed_cases",
    "high_risk_cases",
    "prompt_injection_cases",
    "secret_pii_cases",
    "ambiguous_cases",
    "regression_cases",
    "general",
)


DEFAULT_CATEGORY = "general"


def normalize_router_eval_category(value: object) -> str:
    """把外部传入的 category 归一化成受控类别。

    中文注释：
    eval dataset 可能来自人工反馈、JSON 文件、线上纠错。
    这里做一层兜底，避免错误 category 破坏统计。
    """

    category = str(value or "").strip()
    if category in ROUTER_EVAL_CATEGORIES:
        return category
    return DEFAULT_CATEGORY


def is_strict_router_eval_category(category: str) -> bool:
    """判断某个类别是否属于更严格的安全/高风险类别。"""

    return category in {
        "high_risk_cases",
        "prompt_injection_cases",
        "secret_pii_cases",
    }


__all__ = [
    "DEFAULT_CATEGORY",
    "ROUTER_EVAL_CATEGORIES",
    "RouterEvalCategory",
    "is_strict_router_eval_category",
    "normalize_router_eval_category",
]
