from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from beginner_agent.embeddings import configured_embedding_provider  # noqa: E402


def main() -> None:
    """验证 OMLX 是否已经能提供真正的 embedding。

    中文注释：
    这个脚本故意不使用 safe_embedding(...)。

    原因：
    - safe_embedding 失败时会回退到 hash embedding。
    - 这里我们要确认 Qwen3-Embedding-8B 本身是否真的可用。

    等你的模型下载完成后，可以运行：

        BEGINNER_AGENT_EMBEDDING_PROVIDER=omlx \
        BEGINNER_AGENT_EMBEDDING_DIM=1024 \
        OMLX_BASE_URL=http://127.0.0.1:8000/v1 \
        OMLX_API_KEY=local-omlx-key \
        OMLX_EMBEDDING_MODEL=Qwen3-Embedding-8B \
        uv run python scripts/check_omlx_embedding.py
    """

    os.environ.setdefault("BEGINNER_AGENT_EMBEDDING_PROVIDER", "omlx")
    os.environ.setdefault("BEGINNER_AGENT_EMBEDDING_DIM", "1024")
    os.environ.setdefault("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
    os.environ.setdefault("OMLX_API_KEY", "local-omlx-key")
    os.environ.setdefault("OMLX_EMBEDDING_MODEL", "Qwen3-Embedding-8B")
    os.environ.setdefault("OMLX_EMBEDDING_SEND_DIMENSIONS", "true")

    provider = configured_embedding_provider()
    vector = provider.embed_text("用 Qwen3-Embedding-8B 验证 beginner_agent 的 memory 向量检索。")
    print("OMLX embedding check passed.")
    print(f"provider={provider.provider_name}")
    print(f"model={provider.model_name}")
    print(f"dimension={len(vector)}")
    print(f"first_values={vector[:5]}")


if __name__ == "__main__":
    main()
