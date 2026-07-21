from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


DEFAULT_QWEN3_EMBEDDING_DIM = 1024


class EmbeddingProvider(Protocol):
    """Embedding provider 协议。

    中文注释：
    向量数据库只负责“存向量、查向量”。
    真正把文本变成向量的是 embedding provider。

    这样拆开后：
    - 当前使用 OmlxEmbeddingProvider 调用真正的 embedding 模型。
    - 以后也可以换 OpenAI、bge、Qwen embedding、sentence-transformers。
    """

    # 中文注释：
    # Protocol 可以理解成“接口”或“约定”，它不是用来直接干活的类。
    # 这里的意思是：任何 embedding provider，只要有下面这些属性和方法，
    # 就可以被当作 EmbeddingProvider 使用。
    #
    # Python 的 Protocol 是“看结构”的：
    # 具体 provider 不需要显式继承 EmbeddingProvider。
    # 只要它也有 provider_name、model_name、dimension 和 embed_text(...)，
    # 类型检查器就认为它符合这个协议。
    provider_name: str
    model_name: str
    dimension: int

    def embed_text(self, text: str) -> list[float]:
        """把文本转成固定维度向量。

        中文注释：
        这个函数里只有 docstring，没有 pass，也没有真正的代码，
        是可以通过的。
        因为 docstring 本身就是一个合法的函数体。

        但这里并不是实际生成向量的地方。
        Protocol 里的这个方法只是声明“实现类必须提供 embed_text 方法”。
        真正的实现请看下面的 OmlxEmbeddingProvider.embed_text。
        """


class OmlxEmbeddingProvider:
    """OpenAI-compatible OMLX embedding provider。

    中文注释：
    只有当你的本地 OMLX 服务支持：

        POST /v1/embeddings

    并且所选模型是 embedding 模型时，这个 provider 才能工作。

    截图里的 Qwen3-ASR 是语音识别模型，不适合作为 embedding 模型。
    如果 OMLX 里没有 embedding 模型，
    系统会直接报错，提醒你先配置真实 embedding 模型。
    """

    provider_name = "omlx"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        dimension: int,
        send_dimensions: bool = True,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimension
        self.send_dimensions = send_dimensions
        self.timeout = timeout

    def embed_text(self, text: str) -> list[float]:
        payload: dict[str, object] = {
            "model": self.model_name,
            "input": text,
        }
        if self.send_dimensions:
            # 中文注释：
            # Qwen3 Embedding 支持 MRL，也就是可以指定较低输出维度。
            # 对本项目来说，1024 维已经适合 memory/code 检索，
            # 同时比 4096 维更容易被 pgvector 索引和管理。
            payload["dimensions"] = self.dimension
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(f"OMLX embedding 请求失败：{exc}") from exc

        try:
            embedding = payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"无法解析 OMLX embedding 响应：{payload}") from exc
        if not isinstance(embedding, list):
            raise RuntimeError(f"OMLX embedding 格式不正确：{payload}")
        vector = [float(value) for value in embedding]
        if len(vector) != self.dimension:
            raise RuntimeError(
                "OMLX embedding 维度不匹配："
                f"期望 {self.dimension}，实际 {len(vector)}。"
            )
        return vector


def configured_embedding_provider() -> EmbeddingProvider:
    """根据环境变量选择 embedding provider。"""

    provider = os.getenv("BEGINNER_AGENT_EMBEDDING_PROVIDER", "omlx").strip().lower()
    if provider == "omlx":
        dimension = int(
            os.getenv("BEGINNER_AGENT_EMBEDDING_DIM", str(DEFAULT_QWEN3_EMBEDDING_DIM))
        )
        return OmlxEmbeddingProvider(
            base_url=os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("OMLX_API_KEY", "local-omlx-key"),
            model_name=os.getenv(
                "OMLX_EMBEDDING_MODEL", "Qwen3-Embedding-8B-4bit-DWQ"
            ),
            dimension=dimension,
            send_dimensions=os.getenv(
                "OMLX_EMBEDDING_SEND_DIMENSIONS", "true"
            ).strip().lower()
            not in {"0", "false", "no"},
        )
    raise ValueError(f"不支持的 embedding provider：{provider}。当前只支持 omlx。")


def safe_embedding(text: str) -> tuple[list[float], str, str, int]:
    """生成真实 embedding。

    返回：
        vector, provider_name, model_name, dimension
    """

    provider = configured_embedding_provider()
    return (
        provider.embed_text(text),
        provider.provider_name,
        provider.model_name,
        provider.dimension,
    )


def vector_to_sql(vector: list[float]) -> str:
    """把 Python list 转成 pgvector 接受的字符串格式。"""

    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"
