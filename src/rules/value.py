"""
价值判断原子规则 — 12 条

来源大师：段永平、邱国鹭、Klarman
核心问题：这家公司值不值得关注？

每条规则独立打分（0-100），附带置信度和数据质量。
"""

from __future__ import annotations

import logging

from src.core.signal import AtomicJudgment, EvalContext

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 段永平规则
# ──────────────────────────────────────────────────────────────

def dyp_01_business_clarity(ctx: EvalContext) -> AtomicJudgment | None:
    """段永平：商业模式清晰度（3句话测试）

    定性规则，无法完全自动化。基于净利率和毛利率推断。
    净利率>15%且毛利率>40% → 商业模式可能清晰。
    """
    q = ctx.stock_data.get("quarterly", {})
    if not q:
        return None

    net_margin = q.get("net_margin", 0)
    gross_margin = q.get("gross_margin", 0)
    quality = ctx.quality.assess_dict(q, "quarterly") if ctx.quality else 0.5

    if net_margin > 15 and gross_margin > 40:
        score, direction = 80.0, "BUY"
        reason = f"净利率{net_margin:.1f}%+毛利率{gross_margin:.1f}%，商业模式可能清晰"
    elif net_margin > 10 and gross_margin > 30:
        score, direction = 60.0, "HOLD"
        reason = f"净利率{net_margin:.1f}%+毛利率{gross_margin:.1f}%，商业模式一般"
    elif net_margin > 0:
        score, direction = 40.0, "HOLD"
        reason = f"净利率{net_margin:.1f}%偏低，商业模式待验证"
    else:
        score, direction = 20.0, "WARNING"
        reason = f"净利率{net_margin:.1f}%，商业模式可能有问题"

    return AtomicJudgment(
        rule_id="dyp_01", rule_name="商业模式清晰度", thinker="duanyongping",
        dimension="value", score=score, confidence=0.6, data_quality=quality,
        direction=direction, reason=reason,
    )


def dyp_02_roe_persistence(ctx: EvalContext) -> AtomicJudgment | None:
    """段永平：ROE持续性

    ROE连续3年>20%→90分，>15%→70分，>10%→50分
    """
    q = ctx.stock_data.get("quarterly", {})
    if not q:
        return None

    roe_list = q.get("roe_list", [])
    quality = ctx.quality.assess_dict(q, "quarterly") if ctx.quality else 0.5

    if len(roe_list) < 3:
        return AtomicJudgment(
            rule_id="dyp_02", rule_name="ROE持续性", thinker="duanyongping",
            dimension="value", score=30.0, confidence=0.3, data_quality=quality,
            direction="HOLD", reason=f"ROE数据不足({len(roe_list)}年)，无法判断",
        )

    recent_3 = roe_list[-3:]
    avg_roe = sum(recent_3) / 3
    all_above_15 = all(r > 15 for r in recent_3)
    all_above_20 = all(r > 20 for r in recent_3)

    if all_above_20:
        score = 90.0
        direction = "BUY"
        reason = f"ROE连续3年>20%（均值{avg_roe:.1f}%），段永平：好生意"
    elif all_above_15:
        score = 70.0
        direction = "BUY"
        reason = f"ROE连续3年>15%（均值{avg_roe:.1f}%），赚钱能力强"
    elif avg_roe > 10:
        score = 50.0
        direction = "HOLD"
        reason = f"ROE均值{avg_roe:.1f}%，中等水平"
    else:
        score = 25.0
        direction = "WARNING"
        reason = f"ROE均值{avg_roe:.1f}%偏低，资本回报率不足"

    return AtomicJudgment(
        rule_id="dyp_02", rule_name="ROE持续性", thinker="duanyongping",
        dimension="value", score=score, confidence=0.85, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"roe_list": roe_list, "avg_roe": avg_roe},
    )


def dyp_03_free_cash_flow(ctx: EvalContext) -> AtomicJudgment | None:
    """段永平：自由现金流

    FCF>0且趋势上升→80分，FCF>0→60分，FCF<0→20分
    """
    q = ctx.stock_data.get("quarterly", {})
    if not q:
        return None

    fcf_list = q.get("fcf_list", [])
    quality = ctx.quality.assess_dict(q, "quarterly") if ctx.quality else 0.5

    if not fcf_list:
        return AtomicJudgment(
            rule_id="dyp_03", rule_name="自由现金流", thinker="duanyongping",
            dimension="value", score=40.0, confidence=0.3, data_quality=quality,
            direction="HOLD", reason="无自由现金流数据",
        )

    latest_fcf = fcf_list[-1]
    if len(fcf_list) >= 2:
        trend_up = fcf_list[-1] > fcf_list[-2]
    else:
        trend_up = False

    if latest_fcf > 0 and trend_up:
        score = 80.0
        direction = "BUY"
        reason = f"FCF={latest_fcf:.1f}亿且趋势上升，段永平：真金白银"
    elif latest_fcf > 0:
        score = 60.0
        direction = "HOLD"
        reason = f"FCF={latest_fcf:.1f}亿>0，现金流为正"
    else:
        score = 20.0
        direction = "WARNING"
        reason = f"FCF={latest_fcf:.1f}亿<0，纸面利润风险"

    return AtomicJudgment(
        rule_id="dyp_03", rule_name="自由现金流", thinker="duanyongping",
        dimension="value", score=score, confidence=0.8, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"fcf_list": fcf_list},
    )


def dyp_04_not_to_do_list(ctx: EvalContext) -> AtomicJudgment | None:
    """段永平：不为清单

    净利率<0/ROE全<5%/FCF<0 → 直接排除（硬性规则）
    """
    q = ctx.stock_data.get("quarterly", {})
    if not q:
        return None

    net_margin = q.get("net_margin", 0)
    roe_list = q.get("roe_list", [])
    fcf_list = q.get("fcf_list", [])
    quality = ctx.quality.assess_dict(q, "quarterly") if ctx.quality else 0.5

    violations = []
    if net_margin < 0:
        violations.append(f"净利率{net_margin:.1f}%<0")
    if roe_list and all(r < 5 for r in roe_list[-3:]):
        violations.append(f"ROE连续<5%")
    if fcf_list and fcf_list[-1] < 0:
        violations.append(f"FCF<0")

    if violations:
        return AtomicJudgment(
            rule_id="dyp_04", rule_name="不为清单", thinker="duanyongping",
            dimension="value", score=0.0, confidence=0.95, data_quality=quality,
            direction="SELL", reason=f"不为清单排除：{'、'.join(violations)}",
            metadata={"violations": violations, "excluded": True},
        )

    return AtomicJudgment(
        rule_id="dyp_04", rule_name="不为清单", thinker="duanyongping",
        dimension="value", score=80.0, confidence=0.9, data_quality=quality,
        direction="BUY", reason="通过不为清单检查",
        metadata={"violations": [], "excluded": False},
    )


# ──────────────────────────────────────────────────────────────
# 邱国鹭规则
# ──────────────────────────────────────────────────────────────

def qgy_01_industry_concentration(ctx: EvalContext) -> AtomicJudgment | None:
    """邱国鹭：行业集中度

    CR3>60%→80分（格局清晰），CR3趋势上升→额外+15分
    """
    ind = ctx.market_data.get("industry_comparison")
    stock_industry = ctx.stock_data.get("industry", "")

    if ind is None or ind.empty or not stock_industry:
        return None

    quality = ctx.quality.assess_dataframe(ind, "industry_comparison") if ctx.quality else 0.5

    # 从行业对比数据中查找该行业的CR3
    cr3 = 0.0
    for col in ["CR3", "cr3", "集中度"]:
        if col in ind.columns:
            row = ind[ind.get("行业", ind.columns[0]) == stock_industry]
            if not row.empty:
                cr3 = float(row[col].iloc[0])
                break

    if cr3 <= 0:
        return AtomicJudgment(
            rule_id="qgy_01", rule_name="行业集中度", thinker="qiuguolu",
            dimension="value", score=45.0, confidence=0.3, data_quality=quality,
            direction="HOLD", reason=f"无{stock_industry}行业CR3数据",
        )

    if cr3 >= 60:
        score = 80.0
        reason = f"CR3={cr3:.0f}%>60%，邱国鹭：格局清晰，龙头受益"
    elif cr3 >= 40:
        score = 60.0
        reason = f"CR3={cr3:.0f}%，行业集中度中等"
    elif cr3 >= 20:
        score = 40.0
        reason = f"CR3={cr3:.0f}%，行业分散，竞争激烈"
    else:
        score = 25.0
        reason = f"CR3={cr3:.0f}%极低，邱国鹭：避免混战期"

    return AtomicJudgment(
        rule_id="qgy_01", rule_name="行业集中度", thinker="qiuguolu",
        dimension="value", score=score, confidence=0.7, data_quality=quality,
        direction="BUY" if score >= 60 else "HOLD",
        reason=reason, metadata={"cr3": cr3, "industry": stock_industry},
    )


def qgy_02_pricing_power(ctx: EvalContext) -> AtomicJudgment | None:
    """邱国鹭：定价权（毛利率代理）

    毛利率>40%且稳定→85分（护城河代理指标）
    """
    q = ctx.stock_data.get("quarterly", {})
    if not q:
        return None

    gross_margin = q.get("gross_margin", 0)
    quality = ctx.quality.assess_dict(q, "quarterly") if ctx.quality else 0.5

    if gross_margin >= 60:
        score = 90.0
        reason = f"毛利率{gross_margin:.1f}%极高，强定价权（白酒/医药型）"
    elif gross_margin >= 40:
        score = 75.0
        reason = f"毛利率{gross_margin:.1f}%较高，有定价权"
    elif gross_margin >= 25:
        score = 50.0
        reason = f"毛利率{gross_margin:.1f}%中等，定价权一般"
    else:
        score = 25.0
        reason = f"毛利率{gross_margin:.1f}%低，缺乏定价权"

    return AtomicJudgment(
        rule_id="qgy_02", rule_name="定价权", thinker="qiuguolu",
        dimension="value", score=score, confidence=0.75, data_quality=quality,
        direction="BUY" if score >= 70 else "HOLD",
        reason=reason, metadata={"gross_margin": gross_margin},
    )


def qgy_03_valuation_percentile(ctx: EvalContext) -> AtomicJudgment | None:
    """邱国鹭：估值分位

    PE<历史中位数→80分，PB<历史中位数→额外+10分
    """
    v = ctx.stock_data.get("valuation", {})
    if not v:
        return None

    pe_ttm = v.get("pe_ttm", 0)
    pb = v.get("pb", 0)
    quality = ctx.quality.assess_dict(v, "valuation") if ctx.quality else 0.5

    score = 50.0

    # PE 评估（逆向：越低越好）
    if 0 < pe_ttm <= 15:
        score += 25.0
    elif 0 < pe_ttm <= 25:
        score += 15.0
    elif 0 < pe_ttm <= 40:
        score += 5.0
    elif pe_ttm > 40:
        score -= 10.0

    # PB 评估（逆向：越低越好）
    if 0 < pb <= 1.5:
        score += 15.0
    elif 0 < pb <= 3.0:
        score += 8.0
    elif pb > 5.0:
        score -= 5.0

    score = max(0.0, min(100.0, score))

    return AtomicJudgment(
        rule_id="qgy_03", rule_name="估值分位", thinker="qiuguolu",
        dimension="value", score=score, confidence=0.7, data_quality=quality,
        direction="BUY" if score >= 65 else "HOLD",
        reason=f"PE={pe_ttm:.1f} PB={pb:.2f}，估值评分{score:.0f}",
        metadata={"pe_ttm": pe_ttm, "pb": pb},
    )


# ──────────────────────────────────────────────────────────────
# Klarman 规则
# ──────────────────────────────────────────────────────────────

def klm_01_safety_margin(ctx: EvalContext) -> AtomicJudgment | None:
    """Klarman：安全边际

    用 PB 近似：PB<1 → 以低于净资产的价格买入，安全边际最大
    (内在价值-市价)/内在价值>30%→90分
    """
    v = ctx.stock_data.get("valuation", {})
    if not v:
        return None

    pb = v.get("pb", 0)
    quality = ctx.quality.assess_dict(v, "valuation") if ctx.quality else 0.5

    if pb <= 0:
        return None

    # 用 PB 近似安全边际：PB越低，安全边际越大
    # 假设内在价值 ≈ 1.5×净资产（简化DCF）
    implied_margin = (1.5 - pb) / 1.5

    if implied_margin >= 0.3:
        score = 90.0
        reason = f"PB={pb:.2f}，安全边际{implied_margin:.0%}，Klarman：深度价值"
    elif implied_margin >= 0.15:
        score = 70.0
        reason = f"PB={pb:.2f}，安全边际{implied_margin:.0%}，有一定折扣"
    elif implied_margin >= 0:
        score = 50.0
        reason = f"PB={pb:.2f}，安全边际{implied_margin:.0%}，估值合理"
    else:
        score = 25.0
        reason = f"PB={pb:.2f}，安全边际为负，估值偏高"

    return AtomicJudgment(
        rule_id="klm_01", rule_name="安全边际", thinker="seth_klarman",
        dimension="value", score=max(0, score), confidence=0.65, data_quality=quality,
        direction="BUY" if score >= 70 else "HOLD",
        reason=reason, metadata={"pb": pb, "implied_margin": implied_margin},
    )


def klm_02_peg_valuation(ctx: EvalContext) -> AtomicJudgment | None:
    """Klarman + a-stock-data：PEG估值

    PEG<1→85分，PEG 1-1.5→60分，PEG>1.5→30分
    """
    v = ctx.stock_data.get("valuation", {})
    q = ctx.stock_data.get("quarterly", {})
    if not v or not q:
        return None

    pe_ttm = v.get("pe_ttm", 0)
    # 用近似增长率（ROE作为增长代理）
    roe_list = q.get("roe_list", [])
    quality = ctx.quality.assess_dict(v, "valuation") if ctx.quality else 0.5

    if pe_ttm <= 0 or len(roe_list) < 2:
        return None

    # 近似增长率 = ROE × (1 - 分红率)，简化用 ROE 本身
    growth_rate = roe_list[-1]  # 简化
    if growth_rate <= 0:
        return None

    peg = pe_ttm / growth_rate

    if peg <= 0.5:
        score = 90.0
        reason = f"PEG={peg:.2f}<0.5，严重低估"
    elif peg <= 1.0:
        score = 75.0
        reason = f"PEG={peg:.2f}<1，估值有吸引力"
    elif peg <= 1.5:
        score = 55.0
        reason = f"PEG={peg:.2f}，估值合理"
    elif peg <= 2.0:
        score = 35.0
        reason = f"PEG={peg:.2f}>1.5，偏贵"
    else:
        score = 15.0
        reason = f"PEG={peg:.2f}>2，明显高估"

    return AtomicJudgment(
        rule_id="klm_02", rule_name="PEG估值", thinker="seth_klarman",
        dimension="value", score=score, confidence=0.6, data_quality=quality,
        direction="BUY" if score >= 65 else "HOLD",
        reason=reason, metadata={"peg": peg, "pe_ttm": pe_ttm, "growth": growth_rate},
    )


def klm_03_catalyst_exists(ctx: EvalContext) -> AtomicJudgment | None:
    """Klarman：催化剂存在性

    有催化剂→+20分加成，无催化剂→不扣分但标注
    催化剂类型：财报超预期、纳入指数、大股东增持、并购重组
    """
    f10 = ctx.stock_data.get("f10", {})
    quality = ctx.quality.assess_dict(f10, "f10") if ctx.quality else 0.3

    catalysts = []

    # 检查近期公告中的催化剂关键词
    announcements = f10.get("announcements", [])
    catalyst_keywords = ["增持", "回购", "纳入", "超预期", "业绩预增", "重大合同"]
    for ann in announcements:
        title = str(ann.get("title", ""))
        for kw in catalyst_keywords:
            if kw in title:
                catalysts.append(f"{kw}: {title[:30]}")
                break

    # 检查解禁日历（解禁后企稳 = 催化剂）
    # 这个需要额外数据，暂用默认

    if catalysts:
        score = 70.0 + min(20.0, len(catalysts) * 10.0)
        direction = "BUY"
        reason = f"发现{len(catalysts)}个催化剂：{catalysts[0]}"
    else:
        score = 50.0
        direction = "HOLD"
        reason = "未发现明确催化剂，Klarman：便宜但无催化剂可能是价值陷阱"

    return AtomicJudgment(
        rule_id="klm_03", rule_name="催化剂存在", thinker="seth_klarman",
        dimension="value", score=min(100, score), confidence=0.5, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"catalysts": catalysts},
    )


# ──────────────────────────────────────────────────────────────
# 综合规则
# ──────────────────────────────────────────────────────────────

def cross_01_business_quality(ctx: EvalContext) -> AtomicJudgment | None:
    """综合：商业质量 = dyp_02×0.4 + dyp_03×0.3 + qgy_02×0.3"""
    scores = {}
    for rule_id, weight in [("dyp_02", 0.4), ("dyp_03", 0.3), ("qgy_02", 0.3)]:
        # 从 context 中已有的判断结果中获取
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
        rule_id="cross_01", rule_name="商业质量综合", thinker="composite",
        dimension="value", score=round(composite, 1),
        confidence=0.75, data_quality=0.6,
        direction="BUY" if composite >= 65 else "HOLD",
        reason=f"商业质量综合评分{composite:.0f}",
    )


def cross_02_valuation_safety(ctx: EvalContext) -> AtomicJudgment | None:
    """综合：估值安全 = qgy_03×0.5 + klm_01×0.3 + klm_02×0.2"""
    scores = {}
    for rule_id, weight in [("qgy_03", 0.5), ("klm_01", 0.3), ("klm_02", 0.2)]:
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
        rule_id="cross_02", rule_name="估值安全综合", thinker="composite",
        dimension="value", score=round(composite, 1),
        confidence=0.7, data_quality=0.55,
        direction="BUY" if composite >= 65 else "HOLD",
        reason=f"估值安全综合评分{composite:.0f}",
    )


# ──────────────────────────────────────────────────────────────
# 注册所有规则
# ──────────────────────────────────────────────────────────────

def register_all(registry) -> None:
    """注册所有价值判断规则"""
    rules = [
        (dyp_01_business_clarity, "duanyongping"),
        (dyp_02_roe_persistence, "duanyongping"),
        (dyp_03_free_cash_flow, "duanyongping"),
        (dyp_04_not_to_do_list, "duanyongping"),
        (qgy_01_industry_concentration, "qiuguolu"),
        (qgy_02_pricing_power, "qiuguolu"),
        (qgy_03_valuation_percentile, "qiuguolu"),
        (klm_01_safety_margin, "seth_klarman"),
        (klm_02_peg_valuation, "seth_klarman"),
        (klm_03_catalyst_exists, "seth_klarman"),
        (cross_01_business_quality, "composite"),
        (cross_02_valuation_safety, "composite"),
    ]
    for fn, thinker in rules:
        registry.register(fn.__name__, fn, "value", thinker)
