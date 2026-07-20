from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Protocol


DEFAULT_EMBEDDING_DIM = 384
DEFAULT_QWEN3_EMBEDDING_DIM = 1024


class EmbeddingProvider(Protocol):
    """Embedding provider 协议。

    中文注释：
    向量数据库只负责“存向量、查向量”。
    真正把文本变成向量的是 embedding provider。

    这样拆开后：
    - 本地测试可以用 HashEmbeddingProvider。
    - 如果 OMLX 支持 /embeddings，可以切到 OmlxEmbeddingProvider。
    - 以后也可以换 OpenAI、bge、Qwen embedding、sentence-transformers。
    """

    provider_name: str
    model_name: str
    dimension: int

    def embed_text(self, text: str) -> list[float]:
        """把文本转成固定维度向量。"""


class HashEmbeddingProvider:
    """确定性本地 embedding provider。

    中文注释：
    这不是语义 embedding，不能理解文本意思。
    它的价值是：
    - 不依赖外部模型。
    - 维度固定。
    - 每次同一文本生成同一向量。
    - 适合测试 pgvector 写入、查询、排序链路。

    真正生产效果要换成 OMLX/OpenAI/本地 embedding 模型。
    """

    provider_name = "hash"
    model_name = "hash-embedding-v1"

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        values: list[float] = []
        seed = text.encode("utf-8")
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(seed + str(counter).encode("ascii")).digest()
            for byte in digest:
                # 中文注释：
                # 把 0..255 压到 -1..1，形成稳定数值。
                values.append((byte / 127.5) - 1.0)
                if len(values) >= self.dimension:
                    break
            counter += 1
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [round(value / norm, 8) for value in values]


class OmlxEmbeddingProvider:
    """OpenAI-compatible OMLX embedding provider。

    中文注释：
    只有当你的本地 OMLX 服务支持：

        POST /v1/embeddings

    并且所选模型是 embedding 模型时，这个 provider 才能工作。

    截图里的 Qwen3-ASR 是语音识别模型，不适合作为 embedding 模型。
    如果 OMLX 里没有 embedding 模型，系统会回退到 HashEmbeddingProvider。
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
                f"OMLX embedding 维度不匹配：期望 {self.dimension}，实际 {len(vector)}。"
            )
        return vector


def configured_embedding_provider() -> EmbeddingProvider:
    """根据环境变量选择 embedding provider。"""

    provider = os.getenv("BEGINNER_AGENT_EMBEDDING_PROVIDER", "hash").strip().lower()
    if provider == "omlx":
        dimension = int(
            os.getenv("BEGINNER_AGENT_EMBEDDING_DIM", str(DEFAULT_QWEN3_EMBEDDING_DIM))
        )
        return OmlxEmbeddingProvider(
            base_url=os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("OMLX_API_KEY", "local-omlx-key"),
            model_name=os.getenv("OMLX_EMBEDDING_MODEL", "Qwen3-Embedding-8B"),
            dimension=dimension,
            send_dimensions=os.getenv(
                "OMLX_EMBEDDING_SEND_DIMENSIONS", "true"
            ).strip().lower()
            not in {"0", "false", "no"},
        )
    dimension = int(os.getenv("BEGINNER_AGENT_EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))
    return HashEmbeddingProvider(dimension=dimension)


def safe_embedding(text: str) -> tuple[list[float], str, str, int]:
    """生成 embedding，失败时回退 hash provider。

    返回：
        vector, provider_name, model_name, dimension
    """

    provider = configured_embedding_provider()
    try:
        return provider.embed_text(text), provider.provider_name, provider.model_name, provider.dimension
    except Exception:
        # 中文注释：
        # 如果 OMLX 模型还没下载完或服务暂时不可用，
        # 这里回退到同维度 hash embedding。
        # 这样数据库表维度仍然一致，memory 链路不会被模型下载状态卡死。
        fallback = HashEmbeddingProvider(dimension=provider.dimension)
        return fallback.embed_text(text), fallback.provider_name, fallback.model_name, fallback.dimension


def vector_to_sql(vector: list[float]) -> str:
    """把 Python list 转成 pgvector 接受的字符串格式。"""

    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"
