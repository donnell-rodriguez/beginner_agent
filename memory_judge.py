from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from .llm_client import OMLX_API_KEY, OMLX_BASE_URL, OMLX_MODEL


JudgeDecision = Literal["use", "deprioritize", "reject"]


@dataclass(frozen=True)
class MemoryJudgeResult:
    """LLM / cross-encoder judge 的统一输出。"""

    enabled: bool
    provider: str
    score: float
    decision: JudgeDecision
    reason: str
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "score": self.score,
            "decision": self.decision,
            "reason": self.reason,
            "error": self.error,
        }


def _enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("judge 返回的不是 JSON object。")
    return parsed


def llm_judge_memory_quality(record: dict[str, Any]) -> MemoryJudgeResult:
    """可选 LLM judge：判断 memory 是否值得长期信任。

    中文注释：
    默认关闭，因为本地开发不能假设 LLM 服务一直可用。
    打开方式：

        BEGINNER_AGENT_MEMORY_LLM_JUDGE_ENABLED=true

    这层不是替代本地规则，而是在本地规则后做第二意见。
    """

    if not _enabled("BEGINNER_AGENT_MEMORY_LLM_JUDGE_ENABLED"):
        return MemoryJudgeResult(False, "llm", 0.0, "use", "LLM judge 未启用。")
    base_url = os.getenv("BEGINNER_AGENT_MEMORY_LLM_JUDGE_BASE_URL", OMLX_BASE_URL).rstrip("/")
    model = os.getenv("BEGINNER_AGENT_MEMORY_LLM_JUDGE_MODEL", OMLX_MODEL)
    api_key = os.getenv("BEGINNER_AGENT_MEMORY_LLM_JUDGE_API_KEY", OMLX_API_KEY)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 Memory Quality Judge。"
                    "只返回 JSON："
                    '{"score":0到1,"decision":"use|deprioritize|reject","reason":"一句话"}。'
                    "判断这条记忆是否准确、具体、可复用、有证据。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(record, ensure_ascii=False)[:6000],
            },
        ],
        "temperature": 0,
        "max_tokens": 160,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        content = response_data["choices"][0]["message"]["content"]
        data = _json_object(content)
        decision = str(data.get("decision", "use"))
        if decision not in {"use", "deprioritize", "reject"}:
            decision = "use"
        return MemoryJudgeResult(
            True,
            "llm",
            _clamp(float(data.get("score", 0.5))),
            decision,  # type: ignore[arg-type]
            str(data.get("reason", "LLM judge 未提供原因。"))[:500],
        )
    except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return MemoryJudgeResult(True, "llm", 0.0, "use", "LLM judge 调用失败。", str(exc))


def cross_encoder_rerank_score(query: str, record: dict[str, Any]) -> MemoryJudgeResult:
    """可选 cross-encoder reranker adapter。

    中文注释：
    真实大厂常用 cross-encoder / reranker 模型判断 query 与 memory 是否匹配。
    当前项目不强绑具体模型，只约定一个 HTTP JSON 接口：

        POST BEGINNER_AGENT_MEMORY_CROSS_ENCODER_URL
        {"query":"...","record":{...}}

    返回：
        {"score":0.0到1.0,"reason":"..."}
    """

    url = os.getenv("BEGINNER_AGENT_MEMORY_CROSS_ENCODER_URL", "").strip()
    if not url:
        return MemoryJudgeResult(False, "cross_encoder", 0.0, "use", "cross-encoder 未配置。")
    payload = {"query": query, "record": record}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        score = _clamp(float(data.get("score", 0.5)))
        decision: JudgeDecision = "use" if score >= 0.5 else "deprioritize"
        return MemoryJudgeResult(
            True,
            "cross_encoder",
            score,
            decision,
            str(data.get("reason", "cross-encoder score。"))[:500],
        )
    except (ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return MemoryJudgeResult(
            True,
            "cross_encoder",
            0.0,
            "use",
            "cross-encoder 调用失败，回退本地 reranker。",
            str(exc),
        )

