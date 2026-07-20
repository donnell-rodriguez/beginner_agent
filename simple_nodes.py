from __future__ import annotations

import json
from typing import Any

from .llm_client import chat_completion
from .state import State


def search_node(state: State) -> dict[str, Any]:
    """简单 search 分支。"""

    try:
        draft = chat_completion(
            [
                {
                    "role": "system",
                    "content": "你是资料整理节点。当前没有联网工具，请基于已有知识做结构化整理。",
                },
                {"role": "user", "content": state["user_input"]},
            ],
            temperature=0.2,
            max_tokens=700,
        )
    except RuntimeError as exc:
        draft = f"搜索节点调用本地模型失败：{exc}"
    return {"draft": draft, "messages": [{"role": "assistant", "content": f"Search：{draft}"}]}


def write_node(state: State) -> dict[str, Any]:
    """简单 write 分支。"""

    try:
        draft = chat_completion(
            [
                {"role": "system", "content": "你是中文写作助手。不要输出思考过程，只输出正文。"},
                {"role": "user", "content": state["user_input"]},
            ],
            temperature=0.5,
            max_tokens=900,
        )
    except RuntimeError as exc:
        draft = f"写作节点调用本地模型失败：{exc}"
    return {"draft": draft, "messages": [{"role": "assistant", "content": f"Write：{draft}"}]}


def chat_node(state: State) -> dict[str, Any]:
    """简单 chat 分支。"""

    try:
        draft = chat_completion(
            [
                {"role": "system", "content": "你是耐心的中文技术学习助手。"},
                {"role": "user", "content": state["user_input"]},
            ],
            temperature=0.3,
            max_tokens=900,
        )
    except RuntimeError as exc:
        draft = f"问答节点调用本地模型失败：{exc}"
    return {"draft": draft, "messages": [{"role": "assistant", "content": f"Chat：{draft}"}]}


def summarize_node(state: State) -> dict[str, Any]:
    """最终汇总节点。"""

    draft = state["draft"]
    if state["task_type"] == "agent":
        draft = (
            "这是一个分层复杂 agent 执行结果。\n\n"
            f"用户目标：{state['user_input']}\n\n"
            f"Router：task_type={state['task_type']}, risk_level={state['risk_level']}, "
            f"needs_tool={state['needs_tool']}, reason={state['route_reason']}\n\n"
            f"任务树：\n{json.dumps(state['task_tree'], ensure_ascii=False, indent=2)}\n\n"
            f"记忆：\n{json.dumps(state['memory_notes'], ensure_ascii=False, indent=2)}\n\n"
            f"已完成任务：\n{json.dumps(state['completed_tasks'], ensure_ascii=False, indent=2)}"
        )

    try:
        final_answer = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你是最终答案整理节点。"
                        "请把复杂 agent 的执行结果整理成清楚、适合直接给用户看的中文回答。"
                    ),
                },
                {"role": "user", "content": f"用户原始输入：{state['user_input']}\n\n草稿：{draft}"},
            ],
            temperature=0.2,
            max_tokens=900,
        )
    except RuntimeError:
        final_answer = draft

    return {
        "final_answer": final_answer,
        "messages": [{"role": "assistant", "content": f"Summarize：{final_answer}"}],
    }
