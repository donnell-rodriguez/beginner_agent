from __future__ import annotations

import json

from ..models import RouterStageReport
from .models import MultiStageRouterResult, RouterStageDecision


# 中文注释：
# reporting.py 负责把内部阶段决策转换成可观测报告。
# 内部执行和外部展示分开，后续接 API / Dashboard 会更清晰。


def build_multistage_reports(result: MultiStageRouterResult) -> list[RouterStageReport]:
    """把独立子阶段结果转成 RouterEvent 使用的 stage_reports。"""

    reports: list[RouterStageReport] = []
    for stage in result.stage_decisions:
        reason = stage.reason
        if stage.source == "fallback":
            reason = f"{reason}；fallback_reason={stage.fallback_reason}"
        if stage.repair_attempt_count:
            reason = (
                f"{reason}；repair_attempts={stage.repair_attempt_count}；"
                f"repair_success={stage.repair_success}"
            )
        if stage.failure_policy_applied:
            reason = f"{reason}；failure_policy={stage.failure_policy_applied}"
        if stage.model_error:
            reason = f"{reason}；model_error={stage.model_error}"
        reports.append(
            RouterStageReport(
                stage=stage.stage,
                decision=stage.decision,
                reason=f"source={stage.source}；{reason}",
                confidence=stage.confidence,
            )
        )
    return reports


def combine_model_responses(stages: tuple[RouterStageDecision, ...]) -> str:
    payload = {
        stage.stage: stage.model_response
        for stage in stages
        if stage.model_response
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def combine_stage_field(stages: tuple[RouterStageDecision, ...], field_name: str) -> str:
    values = [
        f"{stage.stage}: {getattr(stage, field_name)}"
        for stage in stages
        if getattr(stage, field_name)
    ]
    return "；".join(values)
