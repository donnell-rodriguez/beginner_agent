from __future__ import annotations

from .models import RouterDecision, RouterSecuritySignal, RouterStageReport
from .rules import RouterRuleSet


# 中文注释：
# stages.py 负责把 Router 的“最终决策”拆成多个可观察的阶段报告。
#
# 为什么需要它？
#
# 如果 Router 只返回：
#
#     task_type="agent", risk_level="high", needs_tool=True
#
# 人只能看到结果，却不知道中间怎么判断的。
#
# 生产级 agent 更需要知道：
# - intent 这一层怎么判断？
# - risk 这一层怎么判断？
# - 是否需要工具是怎么判断的？
# - security 有没有命中风险？
# - tenant/project/user 策略有没有介入？
#
# 所以这里会生成一组 RouterStageReport。
# 它们不是用来控制 graph 路由的核心数据，
# 而是给 observability / audit / eval / debug 使用的解释性数据。


def build_stage_reports(
    *,
    text: str,
    decision: RouterDecision,
    rules: RouterRuleSet,
    security: RouterSecuritySignal,
    context_policy_reason: str,
) -> list[RouterStageReport]:
    """构造多级 Router 子决策报告。

    中文注释：
    这让 Router 不只是给一个最终结果，
    还明确暴露每一层判断：
    intent -> risk -> tool_needs -> security -> context_policy。

    参数说明：
    - text：用户原始输入。
    - decision：Router 经过 LLM / fallback / security / context policy 后的最终决策。
    - rules：本地配置化规则，主要用于对照“如果只靠规则会怎么判”。
    - security：安全分类结果，例如 prompt injection / data exfiltration。
    - context_policy_reason：tenant/project/user 策略命中的原因；没有命中则为空字符串。

    返回值：
    - list[RouterStageReport]
    - 每个 RouterStageReport 表示一个子阶段的判断。
    """

    # 中文注释：
    # 这里先计算“纯规则兜底会怎么判断”。
    #
    # 注意：
    # rule_intent / rule_risk 不一定等于最终 decision。
    #
    # 因为最终 decision 可能来自：
    # - LLM 判断。
    # - 低置信度 fallback。
    # - security override。
    # - context policy override。
    #
    # 把规则判断也记录下来，是为了后续排查：
    #   “最终判断和规则判断为什么不一样？”
    rule_intent = rules.explain_task_type(text)
    rule_risk = rules.explain_risk_level(text)

    # 中文注释：
    # 下面返回 5 个阶段报告。
    #
    # 你可以把它理解成 Router 的“分层解释链”：
    #
    #   intent
    #     -> risk
    #     -> tool_needs
    #     -> security
    #     -> context_policy
    #
    # 这些报告最终会进入 RouterEvent.stage_reports，
    # 也会出现在 State.router_report 里。
    return [
        RouterStageReport(
            # 中文注释：
            # intent 阶段回答：
            # “用户这个请求整体属于什么类型？”
            #
            # 例如：
            # - chat：普通问答。
            # - search：搜索/查找。
            # - write：写作/生成。
            # - agent：需要进入复杂 agent loop。
            stage="intent",
            decision=decision.task_type,
            reason=(
                f"最终 intent={decision.task_type}；"
                f"规则兜底会判为 {rule_intent.outcome}；"
                f"ruleset={rule_intent.ruleset_version}；"
                f"source={rule_intent.ruleset_source}；"
                f"selected_rule={rule_intent.selected_rule_id or 'none'}；"
                f"原因：{rule_intent.selected_rule_reason}"
            ),
            confidence=decision.confidence,
        ),
        RouterStageReport(
            # 中文注释：
            # risk 阶段回答：
            # “这个请求风险等级是多少？”
            #
            # low / medium / high 会影响后面的 Tool Policy。
            # high 通常意味着更可能需要人工审批。
            stage="risk",
            decision=decision.risk_level,
            reason=(
                f"最终 risk={decision.risk_level}；"
                f"规则兜底会判为 {rule_risk.outcome}；"
                f"ruleset={rule_risk.ruleset_version}；"
                f"source={rule_risk.ruleset_source}；"
                f"selected_rule={rule_risk.selected_rule_id or 'none'}；"
                f"原因：{rule_risk.selected_rule_reason}"
            ),
            confidence=decision.confidence,
        ),
        RouterStageReport(
            # 中文注释：
            # tool_needs 阶段回答：
            # “这个请求是否需要工具？”
            #
            # 例如读取文件、运行测试、分析项目结构，都需要工具。
            # 普通聊天通常不需要进入复杂工具 loop。
            stage="tool_needs",
            decision=str(decision.needs_tool).lower(),
            reason="agent 分支通常需要工具；search/write/chat 通常不进入复杂工具 loop。",
            confidence=decision.confidence,
        ),
        RouterStageReport(
            # 中文注释：
            # security 阶段回答：
            # “有没有命中安全风险？”
            #
            # 例如：
            # - prompt_injection：试图让系统忽略规则。
            # - data_exfiltration：试图读取/泄露敏感信息。
            # - unsafe_code_action：要求修改/删除/执行高风险动作。
            stage="security",
            decision=security.malicious_intent,
            reason=security.reason,
            confidence=0.9 if security.malicious_intent != "none" else 0.7,
        ),
        RouterStageReport(
            # 中文注释：
            # context_policy 阶段回答：
            # “有没有因为 tenant/project/user 策略而提升风险？”
            #
            # 例如某个项目被配置为高风险项目，
            # 即使用户只是普通请求，也可以被提升到 high risk，
            # 让后续 Policy / Approval 更谨慎。
            stage="context_policy",
            decision="high_risk_override" if context_policy_reason else "none",
            reason=context_policy_reason or "未命中 tenant/project/user 路由策略。",
            confidence=0.9,
        ),
    ]
