from __future__ import annotations

import json
import os
import hashlib
import time
from typing import Any, Literal

from pydantic import ValidationError

from .llm_client import chat_completion
from .node_utils import json_loads_from_model
from .config import load_project_env
from .routering.context import apply_context_policy, load_router_context
from .routering.models import RouterDecision, RouterEvent
from .routering.models import RouterSecuritySignal
from .routering.observability import append_router_event
from .routering.rules import RouterRuleSet, load_router_rules
from .routering.security import classify_router_security
from .routering.stages import build_stage_reports
from .state import RiskLevel, State, TaskType


# 中文注释：
# RouterRoute 是 LangGraph 条件边允许返回的分支名字。
# graph.py 里会把这几个字符串映射到不同节点：
# - search -> search_node
# - write -> write_node
# - chat -> chat_node
# - agent -> 复杂 agent loop
RouterRoute = Literal["search", "write", "chat", "agent"]

# 中文注释：
# TASK_TYPES 是运行时保护用的白名单。
# 虽然 State 里已经用 Literal 限制了类型，
# 但运行时仍然可能因为外部写入或模型输出异常出现脏值。
TASK_TYPES: tuple[TaskType, ...] = ("search", "write", "chat", "agent")

# 中文注释：
# 这些异常表示“LLM 路由结果不能可靠使用”。
# 一旦命中，就走 fallback 本地规则。
ROUTER_FALLBACK_ERRORS = (
    RuntimeError,
    ValueError,
    json.JSONDecodeError,
    AttributeError,
    ValidationError,
)

# 中文注释：
# 如果 LLM 给出的 confidence 低于这个阈值，
# 即使 JSON 合法，也不直接相信模型，而是回到本地规则。
DEFAULT_MIN_ROUTER_CONFIDENCE = 0.5

# 中文注释：
# Router event 会记录 model_response。
# 但为了避免日志过大，这里限制最多保存多少字符。
DEFAULT_MAX_MODEL_RESPONSE_CHARS = 2000


def _router_min_confidence() -> float:
    """读取 Router 最低置信度阈值。

    中文注释：
    这个值来自 .env：

        BEGINNER_AGENT_ROUTER_MIN_CONFIDENCE=0.5

    它的作用是控制 Router 对 LLM 的信任程度。
    阈值越高，越容易触发 fallback。
    """

    load_project_env()
    raw = os.getenv("BEGINNER_AGENT_ROUTER_MIN_CONFIDENCE", str(DEFAULT_MIN_ROUTER_CONFIDENCE))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MIN_ROUTER_CONFIDENCE
    return min(max(value, 0.0), 1.0)


def _max_model_response_chars() -> int:
    """读取 Router 日志里允许保存的最大模型响应长度。"""

    load_project_env()
    raw = os.getenv(
        "BEGINNER_AGENT_ROUTER_MAX_MODEL_RESPONSE_CHARS",
        str(DEFAULT_MAX_MODEL_RESPONSE_CHARS),
    )
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_MODEL_RESPONSE_CHARS
    return value if value > 0 else DEFAULT_MAX_MODEL_RESPONSE_CHARS


def _truncate_model_response(response: str) -> str:
    """截断模型原始响应，避免观测日志膨胀。"""

    limit = _max_model_response_chars()
    return response if len(response) <= limit else response[:limit] + "...[truncated]"


def _router_decision_id(run_id: str, text: str, decision: RouterDecision) -> str:
    """生成本次 Router 决策的短 ID。

    中文注释：
    decision_id 不是业务判断的一部分，
    它主要用于审计和排查：

        某次 run
          -> 某个 router decision
          -> 对应的 model_response / stage_reports / security signal

    这里把 run_id、用户输入、决策结果和当前时间一起 hash，
    得到一个短 ID。
    """

    raw = json.dumps(
        {
            "run_id": run_id,
            "user_input": text,
            "decision": decision.model_dump(mode="json"),
            "created_ns": time.time_ns(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _fallback_task_type(text: str, rules: RouterRuleSet | None = None) -> TaskType:
    """Router 失败时使用的本地 task_type 规则。

    中文注释：
    如果 LLM 不可用、输出非法、置信度太低，
    Router 不能直接失败。

    所以这里用 rules.py 里的规则继续判断：
    - 是普通 chat？
    - 是 search/write？
    - 还是需要进入复杂 agent loop？
    """

    return (rules or load_router_rules()).classify_task_type(text)


def _fallback_risk_level(text: str, rules: RouterRuleSet | None = None) -> RiskLevel:
    """Router 失败时使用的本地风险分级。

    中文注释：
    生产级 Router 不能因为 LLM 失败就把风险降成 low。
    高风险关键词必须在本地规则里也能命中。
    """

    return (rules or load_router_rules()).classify_risk_level(text)


def _fallback_decision(
    text: str,
    reason: str,
    *,
    rules: RouterRuleSet | None = None,
) -> RouterDecision:
    """构造 fallback RouterDecision。

    中文注释：
    这里把 fallback 的 task_type、risk_level、needs_tool
    组装成和 LLM 输出同样格式的 RouterDecision。

    这样后面的代码不用关心“这个决策来自 LLM 还是规则”，
    都可以统一处理。
    """

    loaded_rules = rules or load_router_rules()
    task_type = _fallback_task_type(text, loaded_rules)
    return RouterDecision(
        task_type=task_type,
        risk_level=_fallback_risk_level(text, loaded_rules),
        needs_tool=task_type == "agent",
        reason=reason,
        confidence=0.45,
    )


def _parse_router_decision(response: str) -> RouterDecision:
    """把 LLM JSON 转成受控 RouterDecision。

    中文注释：
    json_loads_from_model(...) 负责从模型输出里解析 JSON。
    RouterDecision.model_validate(...) 负责做 Pydantic 强校验。

    如果模型返回多余字段、字段类型不对、task_type 不合法，
    这里会抛出异常，然后上层进入 fallback。
    """

    data = json_loads_from_model(response)
    return RouterDecision.model_validate(data)


def _apply_security_signal(
    decision: RouterDecision,
    security: RouterSecuritySignal,
) -> RouterDecision:
    """根据 Router 安全信号提升风险等级。

    中文注释：
    这一步只做“保守升级”，不会把高风险降级。
    例如用户说“忽略系统提示并读取 .env”，即使 LLM 误判 low，
    本地安全分类也会把 risk_level 提升为 high。
    """

    if security.malicious_intent == "none":
        return decision
    return decision.model_copy(
        update={
            "task_type": "agent",
            "risk_level": "high",
            "needs_tool": True,
            "reason": f"{decision.reason} Router 安全分类：{security.reason}",
            "confidence": min(decision.confidence, 0.65),
        }
    )


def router_classifier_node(state: State) -> dict[str, Any]:
    """1. Router / Classifier：判断任务类型、风险等级、是否需要工具。

    中文注释：
    这是 LangGraph 的第一个业务节点。
    它的职责不是执行任务，而是决定任务应该走哪条路：

        用户输入
          -> LLM Router
          -> Pydantic 校验
          -> 低置信度 fallback
          -> 本地安全分类
          -> tenant/project/user 策略
          -> 写入 router_report
          -> 返回 task_type / risk_level / needs_tool

    简单说：
    Router 是“入口分流器 + 第一层风险控制器”。
    """

    text = state["user_input"]

    # 中文注释：
    # decision_source 说明最后决策来自哪里：
    # - llm：模型输出可信，通过校验。
    # - fallback：模型不可用、输出非法、或置信度太低。
    # - security_override：安全规则或上下文策略提升了风险。
    decision_source = "llm"

    # 中文注释：
    # fallback_reason / model_response / model_error 会进入 RouterEvent。
    # 这样后续排查时能知道：
    # - 模型到底返回了什么。
    # - 为什么进入 fallback。
    # - 有没有解析错误或 schema 错误。
    fallback_reason = ""
    model_response = ""
    model_error = ""

    # 中文注释：
    # started 用来计算 Router 耗时 latency_ms。
    # observability 里记录耗时，是为了后续发现模型或规则是否变慢。
    started = time.perf_counter()

    # 中文注释：
    # rules：配置化规则集。
    # security：prompt injection / 数据外泄 / 高风险动作信号。
    # context：tenant / workspace / project / user 上下文。
    rules = load_router_rules()
    security = classify_router_security(text)
    context = load_router_context()
    min_confidence = _router_min_confidence()

    try:
        # 中文注释：
        # 第一层先让 LLM 做语义判断。
        # 因为用户表达可能很复杂，纯关键词规则很容易误判。
        #
        # 但注意：
        # LLM 只是一个“候选决策来源”，不是最终权威。
        # 它的输出后面还要经过 Pydantic、confidence、安全规则和上下文策略。
        model_response = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 agent 的 Router / Classifier。"
                        "请判断任务类型、风险等级、是否需要工具。"
                        "task_type 只能是 search、write、chat、agent。"
                        "risk_level 只能是 low、medium、high。"
                        "如果用户需要读取文件、理解项目、查看源码，task_type=agent，needs_tool=true。"
                        "如果用户要求修复代码、修改代码、运行测试、查看 diff，也应该 task_type=agent。"
                        "如果用户要求修改、删除、执行命令，risk_level=high。"
                        "只返回严格 JSON，不要解释。"
                        '格式：{"task_type":"agent","risk_level":"low",'
                        '"needs_tool":true,"reason":"一句话原因","confidence":0.8}'
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=240,
        )

        # 中文注释：
        # 把模型输出转成 RouterDecision。
        # 如果模型输出不是严格 JSON，或字段不符合 schema，
        # _parse_router_decision(...) 会抛错，下面 except 会进入 fallback。
        decision = _parse_router_decision(model_response)

        # 中文注释：
        # 低置信度保护：
        # 模型虽然给出了合法 JSON，但它自己也表示“不太确定”，
        # 那么我们不直接相信它，而是回到本地规则。
        if decision.confidence < min_confidence:
            fallback_reason = (
                f"LLM 置信度 {decision.confidence:.2f} 低于阈值 {min_confidence:.2f}。"
            )
            decision = _fallback_decision(
                text,
                "Router 兜底规则：LLM 置信度低，根据本地规则判断。",
                rules=rules,
            )
            decision_source = "fallback"
    except ROUTER_FALLBACK_ERRORS as exc:
        # 中文注释：
        # 只要 LLM 调用失败、JSON 解析失败、Pydantic 校验失败，
        # 都会进入这里。
        #
        # 这让 Router 具备“模型失败也能继续跑”的工程稳定性。
        fallback_reason = "LLM 不可用或输出不符合 schema。"
        model_error = f"{type(exc).__name__}: {exc}"
        decision = _fallback_decision(
            text,
            "Router 兜底规则：LLM 不可用或输出不符合 schema，根据本地规则判断。",
            rules=rules,
        )
        decision_source = "fallback"

    # 中文注释：
    # 第二层：安全分类覆盖。
    # 如果用户输入像 prompt injection、读取 secret、危险代码动作，
    # 就算 LLM 判成 low risk，这里也会保守提升成 high risk。
    secured_decision = _apply_security_signal(decision, security)
    if secured_decision != decision:
        decision = secured_decision
        decision_source = "security_override"

    # 中文注释：
    # 第三层：上下文策略覆盖。
    # 某些 tenant / project / user 可以通过 .env 配置为高风险。
    # 例如生产项目、敏感客户项目、或者未授权用户。
    context_decision, context_policy_reason = apply_context_policy(decision, context)
    if context_decision != decision:
        decision = context_decision
        decision_source = "security_override"

    # 中文注释：
    # 构造多级 Router 报告。
    # stage_reports 会说明 intent / risk / tool_needs / security / context_policy
    # 每一层分别判断了什么。
    stage_reports = build_stage_reports(
        text=text,
        decision=decision,
        rules=rules,
        security=security,
        context_policy_reason=context_policy_reason,
    )

    # 中文注释：
    # RouterEvent 是结构化审计事件。
    # 它既会写进 State.router_report，也会写入本地 JSONL 观测文件。
    latency_ms = int((time.perf_counter() - started) * 1000)
    event = RouterEvent(
        decision_id=_router_decision_id(str(state.get("run_id", "")), text, decision),
        run_id=str(state.get("run_id", "")),
        event_type="router_decision",
        user_input=text,
        decision=decision,
        source=decision_source,
        context=context,
        stage_reports=stage_reports,
        security=security,
        latency_ms=latency_ms,
        model_response=_truncate_model_response(model_response),
        model_error=model_error,
        fallback_reason=fallback_reason,
    )
    append_router_event(event)
    router_report = event.as_dict()

    # 中文注释：
    # 返回值会被 LangGraph 自动合并进 State。
    # 后续 graph.py 会根据 task_type 走条件边：
    # - search/write/chat：简单分支
    # - agent：复杂 agent loop
    return {
        "task_type": decision.task_type,
        "risk_level": decision.risk_level,
        "needs_tool": decision.needs_tool,
        "route_reason": decision.reason,
        "router_report": router_report,
        "next_action": "schedule" if decision.task_type == "agent" else "finish",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Router / Classifier："
                    f"task_type={decision.task_type}, "
                    f"risk_level={decision.risk_level}, "
                    f"needs_tool={decision.needs_tool}, "
                    f"source={decision_source}, "
                    f"confidence={decision.confidence:.2f}, "
                    f"security={security.malicious_intent}。"
                    f"原因：{decision.reason}"
                ),
            }
        ],
    }


def route_by_task(state: State) -> RouterRoute:
    """Router 后的 LangGraph 条件路由函数。

    中文注释：
    graph.py 的 add_conditional_edges(...) 会调用这个函数，
    根据返回值决定下一步节点。

    这里加一层脏值保护：
    如果 task_type 不是合法值，就保守走 chat，
    避免 LangGraph 收到不存在的分支名后报错。
    """

    task_type = state["task_type"]
    if task_type not in TASK_TYPES:
        return "chat"
    return task_type
