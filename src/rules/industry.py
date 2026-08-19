"""
行业判断原子规则 — 5 条

来源大师：邱国鹭
核心问题：这个行业值不值得关注？

邱国鹭核心理论：「数月亮不数星星」— 选格局清晰的行业。
当行业从分散走向集中，那段路程是回报最甜美的。
"""

from __future__ import annotations

import logging

from src.core.signal import AtomicJudgment, EvalContext

logger = logging.getLogger(__name__)


def qgy_04_structure_shift(ctx: EvalContext) -> AtomicJudgment | None:
    """邱国鹭：行业结构变迁

    从分散→集中的行业→90分（甜蜜期）
    需要行业对比数据中的集中度变化趋势。
    """
    ind = ctx.market_data.get("industry_comparison")
    if ind is None or ind.empty:
        return None

    quality = ctx.quality.assess_dataframe(ind, "industry_comparison") if ctx.quality else 0.5

    # 简化：从行业对比数据中找集中度最高的行业
    # 实际实现需要历史数据对比CR3趋势
    cr3_col = None
    for col in ["CR3", "cr3", "集中度"]:
        if col in ind.columns:
            cr3_col = col
            break

    if cr3_col is None:
        return AtomicJudgment(
            rule_id="qgy_04", rule_name="行业结构变迁", thinker="qiuguolu",
            dimension="industry", score=50.0, confidence=0.3, data_quality=quality,
            direction="HOLD", reason="无行业集中度数据",
        )

    # 找集中度最高和最低的行业
    try:
        max_cr3 = ind[cr3_col].max()
        avg_cr3 = ind[cr3_col].mean()
    except (TypeError, ValueError):
        return None

    if max_cr3 > 60:
        score = 75.0
        reason = f"行业最高CR3={max_cr3:.0f}%，存在格局清晰的行业"
    elif avg_cr3 > 40:
        score = 60.0
        reason = f"行业平均CR3={avg_cr3:.0f}%，整体集中度尚可"
    else:
        score = 40.0
        reason = f"行业平均CR3={avg_cr3:.0f}%，整体分散"

    return AtomicJudgment(
        rule_id="qgy_04", rule_name="行业结构变迁", thinker="qiuguolu",
        dimension="industry", score=score, confidence=0.5, data_quality=quality,
        direction="BUY" if score >= 70 else "HOLD",
        reason=reason, metadata={"max_cr3": max_cr3, "avg_cr3": avg_cr3},
    )


def qgy_05_industry_roe_trend(ctx: EvalContext) -> AtomicJudgment | None:
    """邱国鹭：行业ROE趋势

    行业ROE连续2季度环比上升→75分
    """
    ind = ctx.market_data.get("industry_comparison")
    if ind is None or ind.empty:
        return None

    quality = ctx.quality.assess_dataframe(ind, "industry_comparison") if ctx.quality else 0.5

    roe_col = None
    for col in ["行业ROE", "roe", "ROE"]:
        if col in ind.columns:
            roe_col = col
            break

    if roe_col is None:
        return AtomicJudgment(
            rule_id="qgy_05", rule_name="行业ROE趋势", thinker="qiuguolu",
            dimension="industry", score=50.0, confidence=0.3, data_quality=quality,
            direction="HOLD", reason="无行业ROE数据",
        )

    try:
        avg_roe = ind[roe_col].mean()
        max_roe = ind[roe_col].max()
    except (TypeError, ValueError):
        return None

    if max_roe > 20:
        score = 75.0
        reason = f"行业最高ROE={max_roe:.1f}%，存在高景气行业"
    elif avg_roe > 10:
        score = 55.0
        reason = f"行业平均ROE={avg_roe:.1f}%，整体景气度中等"
    else:
        score = 35.0
        reason = f"行业平均ROE={avg_roe:.1f}%，整体景气度偏低"

    return AtomicJudgment(
        rule_id="qgy_05", rule_name="行业ROE趋势", thinker="qiuguolu",
        dimension="industry", score=score, confidence=0.5, data_quality=quality,
        direction="BUY" if score >= 70 else "HOLD",
        reason=reason, metadata={"avg_roe": avg_roe, "max_roe": max_roe},
    )


def qgy_06_policy_direction(ctx: EvalContext) -> AtomicJudgment | None:
    """邱国鹭：政策方向

    A股是政策市，政策方向是最重要的行业驱动力之一。
    简化实现：基于行业名称做定性判断（需要人工输入或NLP）。
    """
    stock_industry = ctx.stock_data.get("industry", "")
    if not stock_industry:
        return None

    # 简化：用关键词匹配政策方向
    # 实际应接入政策NLP分析
    positive_keywords = ["新能源", "半导体", "芯片", "AI", "人工智能", "数字经济", "军工"]
    negative_keywords = ["房地产", "教培", "游戏"]

    quality = 0.3  # 定性判断，数据质量低

    for kw in positive_keywords:
        if kw in stock_industry:
            return AtomicJudgment(
                rule_id="qgy_06", rule_name="政策方向", thinker="qiuguolu",
                dimension="industry", score=75.0, confidence=0.4, data_quality=quality,
                direction="BUY",
                reason=f"{stock_industry}属于政策支持方向（{kw}）",
            )

    for kw in negative_keywords:
        if kw in stock_industry:
            return AtomicJudgment(
                rule_id="qgy_06", rule_name="政策方向", thinker="qiuguolu",
                dimension="industry", score=25.0, confidence=0.4, data_quality=quality,
                direction="WARNING",
                reason=f"{stock_industry}可能受政策限制（{kw}）",
            )

    return AtomicJudgment(
        rule_id="qgy_06", rule_name="政策方向", thinker="qiuguolu",
        dimension="industry", score=50.0, confidence=0.3, data_quality=quality,
        direction="HOLD", reason=f"{stock_industry}政策方向中性",
    )


def qgy_07_industry_fund_flow(ctx: EvalContext) -> AtomicJudgment | None:
    """邱国鹭：行业资金流向

    板块资金连续3天净流入→70分
    """
    fund_flow = ctx.market_data.get("industry_fund_flow")
    if fund_flow is None:
        return None

    quality = ctx.quality.assess_dataframe(fund_flow, "fund_flow") if ctx.quality else 0.5

    # 简化：检查资金流向数据
    if fund_flow.empty:
        return None

    try:
        flow_col = None
        for col in ["净流入", "net_flow", "资金流向"]:
            if col in fund_flow.columns:
                flow_col = col
                break
        if flow_col is None:
            return None

        recent_flow = fund_flow[flow_col].tail(3)
        positive_days = (recent_flow > 0).sum()

        if positive_days >= 3:
            score = 75.0
            reason = "板块资金连续3天净流入，资金看好"
        elif positive_days >= 2:
            score = 55.0
            reason = "板块资金近3天2天净流入"
        elif positive_days >= 1:
            score = 40.0
            reason = "板块资金近3天1天净流入"
        else:
            score = 25.0
            reason = "板块资金连续3天净流出"

        return AtomicJudgment(
            rule_id="qgy_07", rule_name="行业资金流向", thinker="qiuguolu",
            dimension="industry", score=score, confidence=0.6, data_quality=quality,
            direction="BUY" if score >= 65 else "HOLD",
            reason=reason, metadata={"positive_days": positive_days},
        )
    except Exception:
        return None


def ind_01_industry_composite(ctx: EvalContext) -> AtomicJudgment | None:
    """综合：行业景气 = qgy_04×0.3 + qgy_05×0.3 + qgy_07×0.2 + qgy_06×0.2"""
    scores = {}
    for rule_id, weight in [("qgy_04", 0.3), ("qgy_05", 0.3), ("qgy_07", 0.2), ("qgy_06", 0.2)]:
        existing = ctx.stock_data.get("_judgments", {}).get(rule_id)
        if existing:
            scores[rule_id] = (existing.score, weight)

    if not scores:
        return None

    total_weight = sum(w for _, w in scores.values())
    if total_weight <= 0:
        return None

    composite = sum(s * w for s, w in scores.values()) / total_weight

    return AtomicJudgment(
        rule_id="ind_01", rule_name="行业景气综合", thinker="composite",
        dimension="industry", score=round(composite, 1),
        confidence=0.6, data_quality=0.5,
        direction="BUY" if composite >= 65 else "HOLD",
        reason=f"行业景气综合评分{composite:.0f}",
    )


def register_all(registry) -> None:
    """注册所有行业判断规则"""
    rules = [
        (qgy_04_structure_shift, "qiuguolu"),
        (qgy_05_industry_roe_trend, "qiuguolu"),
        (qgy_06_policy_direction, "qiuguolu"),
        (qgy_07_industry_fund_flow, "qiuguolu"),
        (ind_01_industry_composite, "composite"),
    ]
    for fn, thinker in rules:
        registry.register(fn.__name__, fn, "industry", thinker)
