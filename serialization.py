from __future__ import annotations

from typing import Any


def serialize_for_json(value: Any) -> Any:
    """把 LangGraph State 转成 json.dumps 可以处理的普通对象。

    中文注释：
    messages: Annotated[list, add_messages] 可能把普通 dict 消息
    转成 LangChain 的 HumanMessage / AIMessage 对象。

    这些对象不能直接 json.dumps(...)。
    所以这里做一层转换：

        HumanMessage(content="你好")
        -> {"type": "human", "content": "你好"}
    """

    if isinstance(value, list):
        return [serialize_for_json(item) for item in value]

    if isinstance(value, dict):
        return {key: serialize_for_json(item) for key, item in value.items()}

    if hasattr(value, "type") and hasattr(value, "content"):
        return {
            "type": value.type,
            "content": value.content,
        }

    return value
