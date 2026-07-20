# 中文注释：
# __future__.annotations 表示延迟解析类型注解。
# 这里主要是让类型注解在运行时更轻量，也方便后续写更复杂的类型。
from __future__ import annotations

# 中文注释：
# json 是 Python 标准库，用来处理 JSON。
# 调用 OpenAI-compatible 接口时，请求体和响应体都是 JSON。
import json

# 中文注释：
# os 是 Python 标准库，用来读取环境变量。
# 这样你以后可以通过环境变量切换模型、接口地址、API key。
import os

# 中文注释：
# re 是 Python 标准库，用来做正则匹配。
# 这里用于清理模型可能输出的 <think>...</think> 思考内容。
import re

# 中文注释：
# urllib.error 提供 HTTP 请求错误类型。
# 例如接口返回 401、500，或者连接失败时，我们可以给出更清楚的报错。
import urllib.error

# 中文注释：
# urllib.request 是 Python 标准库里的 HTTP 客户端。
# 这里不用额外安装 openai 包，方便你先理解最底层请求是怎么发出去的。
import urllib.request

# 中文注释：
# Any 表示“任意类型”。
# messages 是一个 list[dict[str, Any]]，因为 message 里面的字段可能是字符串、列表等。
from typing import Any


# 中文注释：
# OMLX_BASE_URL 是本地 OMLX 的 OpenAI-compatible API 地址。
#
# 我已经在你机器上探测到：
#   http://127.0.0.1:8000/v1/models
#
# 可以正常返回模型列表。
#
# 如果你以后端口变了，可以在运行前设置环境变量：
#   export OMLX_BASE_URL="http://127.0.0.1:8000/v1"
OMLX_BASE_URL = os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")

# 中文注释：
# OMLX_API_KEY 是本地 OMLX 的访问 key。
#
# 这里按你的本地配置给默认值。
# 真实生产项目里不建议把 key 写进源码，通常只从环境变量读取。
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "local-omlx-key")

# 中文注释：
# OMLX_MODEL 是要调用的本地模型名。
#
# 你本地模型列表里可用的是：
#   Qwen3.6-27B-bf16
OMLX_MODEL = os.getenv("OMLX_MODEL", "Qwen3.6-27B-bf16")


# 中文注释：
# _clean_visible_answer 负责清理模型输出。
#
# 为什么需要它？
#   有些推理模型会把“思考过程”也输出出来。
#   真实 agent 给用户看时，通常只应该展示最终答案。
#
# 这里做两类基础清理：
#   1. 删除 <think>...</think> 这样的显式思考块。
#   2. 如果模型用英文输出 "Here's a thinking process:"，给出提醒文本。
def _clean_visible_answer(content: str) -> str:
    # 中文注释：
    # 删除 <think>...</think> 块。
    # flags=re.DOTALL 表示 . 可以匹配换行。
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # 中文注释：
    # 有些模型不会用 <think> 标签，而是直接输出：
    #   Here's a thinking process:
    #
    # 这种情况下很难可靠地从长文本中自动截出最终答案。
    # 所以这里先加一个明显提示，提醒我们应该继续优化 prompt 或模型配置。
    if cleaned.lower().startswith("here's a thinking process"):
        return (
            "模型返回了思考过程，不适合直接作为最终答案展示。\n"
            "请检查 prompt 或关闭模型的 thinking 输出。\n\n"
            f"原始输出片段：\n{cleaned[:800]}"
        )

    # 中文注释：
    # 返回清理后的可见内容。
    return cleaned


# 中文注释：
# chat_completion 是一个最小 LLM 调用函数。
#
# 它做的事情是：
#   1. 组装 OpenAI-compatible /chat/completions 请求。
#   2. 发送给本地 OMLX。
#   3. 解析 choices[0].message.content。
#   4. 返回模型生成的文本。
def chat_completion(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    # 中文注释：
    # base_url.rstrip("/") 去掉末尾多余的斜杠。
    #
    # 例如：
    #   http://127.0.0.1:8000/v1/
    #
    # 会变成：
    #   http://127.0.0.1:8000/v1
    base_url = OMLX_BASE_URL.rstrip("/")

    # 中文注释：
    # chat_url 是真正请求的接口地址。
    #
    # OpenAI-compatible 的聊天接口通常是：
    #   /v1/chat/completions
    chat_url = f"{base_url}/chat/completions"

    # 中文注释：
    # payload 是 HTTP 请求体。
    #
    # model：指定使用哪个模型。
    # messages：对话消息列表。
    # temperature：生成随机性，越低越稳定。
    # max_tokens：最多生成多少 token。
    # stream=False：这次先用非流式，方便小白理解。
    # chat_template_kwargs：
    #   这是 OMLX 支持的额外参数。
    #   enable_thinking=False 表示尽量关闭模型显式思考输出。
    #   preserve_thinking=False 表示不要把历史 thinking 内容保留下来。
    payload = {
        "model": OMLX_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }

    # 中文注释：
    # json.dumps(payload) 把 Python dict 转成 JSON 字符串。
    # encode("utf-8") 再把字符串转成 HTTP 可以发送的 bytes。
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # 中文注释：
    # Request 表示一个 HTTP 请求对象。
    #
    # Authorization:
    #   用 Bearer token 形式携带 API key。
    #
    # Content-Type:
    #   告诉服务端，请求体是 JSON。
    request = urllib.request.Request(
        chat_url,
        data=body,
        headers={
            "Authorization": f"Bearer {OMLX_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        # 中文注释：
        # urlopen(...) 真正发起 HTTP 请求。
        #
        # timeout=120 表示最多等待 120 秒。
        # 本地大模型首次加载可能比较慢，所以这里给长一点。
        with urllib.request.urlopen(request, timeout=120) as response:
            # 中文注释：
            # response.read() 读取响应 bytes。
            # decode("utf-8") 转回字符串。
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # 中文注释：
        # HTTPError 表示服务端返回了错误状态码。
        # 例如 401 key 不对、404 路径不对、500 模型服务内部错误。
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OMLX HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        # 中文注释：
        # URLError 通常表示连接不上服务。
        # 例如 OMLX 没启动、端口不对、网络被阻止。
        raise RuntimeError(f"无法连接 OMLX 服务：{exc}") from exc

    # 中文注释：
    # json.loads(...) 把响应 JSON 字符串转回 Python dict。
    data = json.loads(response_text)

    try:
        # 中文注释：
        # OpenAI-compatible 响应通常长这样：
        #
        #   {
        #     "choices": [
        #       {
        #         "message": {
        #           "content": "模型回答..."
        #         }
        #       }
        #     ]
        #   }
        #
        # 所以这里取 choices[0].message.content。
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        # 中文注释：
        # 如果服务端返回格式不是预期结构，就抛出清楚的错误。
        raise RuntimeError(f"无法解析 OMLX 响应：{data}") from exc

    # 中文注释：
    # str(content) 确保 content 是字符串。
    # _clean_visible_answer(...) 会去掉常见思考块。
    return _clean_visible_answer(str(content))
