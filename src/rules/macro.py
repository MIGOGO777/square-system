"""
宏观判断原子规则 — 8 条

来源大师：Howard Marks（霍华德·马克斯）、Ray Dalio（达利欧）
核心问题：宏观环境是否支持这个操作？

Marks核心理论：
- 「钟摆理论」：市场在极度乐观和极度悲观之间摆动
- 「第二层思维」：不是"会怎样"，而是"共识是什么，共识错在哪"
- 「周期定位」：知道自己在周期的哪个位置

Dalio核心理论：
- 「大周期」：经济在长期债务周期中运行
- 「全天候」：不同宏观环境配置不同资产
"""

from __future__ import annotations

import logging

from src.core.signal import AtomicJudgment, EvalContext

logger = logging.getLogger(__name__)


def marks_01_pendulum(ctx: EvalContext) -> AtomicJudgment | None:
    """Marks：钟摆位置

    PE分位+北向方向+涨停扩散→钟摆偏左/中/右
    钟摆偏左（悲观）= 机会，钟摆偏右（乐观）= 风险
    """
    pe_percentile = ctx.market_data.get("pe_percentile", 50.0)
    north_flow = ctx.market_data.get("north_flow")
    limit_up = ctx.market_data.get("limit_up_count", 0)
    quality = 0.7

    # 钟摆评分：综合多个维度
    pendulum_score = 50.0

    # PE分位（逆向：越低越左，越高越右）
    if pe_percentile < 20:
        pendulum_score -= 25.0  # 极度偏左（悲观）
    elif pe_percentile < 40:
        pendulum_score -= 10.0
    elif pe_percentile > 80:
        pendulum_score += 25.0  # 极度偏右（乐观）
    elif pe_percentile > 60:
        pendulum_score += 10.0

    # 北向资金方向
    if north_flow is not None and not north_flow.empty:
        try:
            flow_col = None
            for col in ("net_flow", "净流入", "north_net"):
                if col in north_flow.columns:
                    flow_col = col
                    break
            if flow_col:
                flow_val = float(north_flow[flow_col].iloc[-1])
                if flow_val > 100:
                    pendulum_score += 10.0  # 外资看多
                elif flow_val < -100:
                    pendulum_score -= 10.0  # 外资看空
        except Exception:
            pass

    # 涨停数（市场热度代理）
    if limit_up >= 80:
        pendulum_score += 15.0  # 市场极热
    elif limit_up >= 50:
        pendulum_score += 5.0
    elif limit_up <= 20:
        pendulum_score -= 10.0  # 市场极冷

    pendulum_score = max(0.0, min(100.0, pendulum_score))

    # 逆向评分：钟摆越偏左（悲观），分数越高（机会越大）
    if pendulum_score <= 25:
        score = 90.0
        position = "极度偏左（极度悲观）"
        reason = f"钟摆{position}，Marks：逆向布局黄金期"
    elif pendulum_score <= 40:
        score = 75.0
        position = "偏左（悲观）"
        reason = f"钟摆{position}，Marks：机会大于风险"
    elif pendulum_score <= 60:
        score = 55.0
        position = "中性"
        reason = f"钟摆{position}，正常操作"
    elif pendulum_score <= 75:
        score = 35.0
        position = "偏右（乐观）"
        reason = f"钟摆{position}，Marks：谨慎行事"
    else:
        score = 15.0
        position = "极度偏右（极度乐观）"
        reason = f"钟摆{position}，Marks：准备离场"

    return AtomicJudgment(
        rule_id="marks_01", rule_name="钟摆位置", thinker="howard_marks",
        dimension="macro", score=score, confidence=0.75, data_quality=quality,
        direction="BUY" if score >= 65 else ("SELL" if score < 30 else "HOLD"),
        reason=reason,
        metadata={"pendulum_score": pendulum_score, "pe_percentile": pe_percentile, "position": position},
    )


def marks_02_consensus_deviation(ctx: EvalContext) -> AtomicJudgment | None:
    """Marks：共识偏差

    北向资金连续流入天数→共识乐观程度
    共识越一致→反转概率越大（Marks第二层思维）
    """
    north_flow = ctx.market_data.get("north_flow")
    quality = 0.7

    if north_flow is None or north_flow.empty:
        return None

    try:
        flow_col = None
        for col in ("net_flow", "净流入", "north_net"):
            if col in north_flow.columns:
                flow_col = col
                break
        if flow_col is None:
            return None

        recent = north_flow[flow_col].dropna().tail(10)
        if len(recent) < 5:
            return None

        consecutive_in = 0
        consecutive_out = 0
        for val in reversed(recent.tolist()):
            if val > 0:
                consecutive_in += 1
                consecutive_out = 0
            elif val < 0:
                consecutive_out += 1
                consecutive_in = 0
            else:
                break

        # 逆向逻辑：连续流入越久→共识越强→风险越大
        if consecutive_in >= 8:
            score = 25.0
            direction = "WARNING"
            reason = f"北向连续{consecutive_in}天流入，Marks：共识过强，警惕反转"
        elif consecutive_in >= 5:
            score = 45.0
            direction = "HOLD"
            reason = f"北向连续{consecutive_in}天流入，共识偏强"
        elif consecutive_out >= 8:
            score = 75.0
            direction = "BUY"
            reason = f"北向连续{consecutive_out}天流出，Marks：极度悲观=机会"
        elif consecutive_out >= 5:
            score = 65.0
            direction = "BUY"
            reason = f"北向连续{consecutive_out}天流出，共识偏空，逆向机会"
        else:
            score = 50.0
            direction = "HOLD"
            reason = f"北向方向不明确（近10日净流入/流出交替）"

        return AtomicJudgment(
            rule_id="marks_02", rule_name="共识偏差", thinker="howard_marks",
            dimension="macro", score=score, confidence=0.65, data_quality=quality,
            direction=direction, reason=reason,
            metadata={"consecutive_in": consecutive_in, "consecutive_out": consecutive_out},
        )
    except Exception:
        return None


def marks_03_cycle_position(ctx: EvalContext) -> AtomicJudgment | None:
    """Marks：周期定位

    综合M2/PMI/CPI→扩张/峰值/收缩/谷底
    扩张期→70分，谷底→90分（逆向），峰值→20分，收缩→30分
    """
    m2_growth = ctx.market_data.get("m2_growth", 0)
    pmi = ctx.market_data.get("pmi", 50)
    cpi = ctx.market_data.get("cpi", 2.0)
    quality = 0.5  # 宏观数据更新慢

    # 简化周期判断
    # M2高+PMI高+CPI高 = 峰值
    # M2高+PMI低+CPI低 = 谷底（政策已发力但未见效）
    # M2低+PMI高+CPI低 = 扩张
    # M2低+PMI低+CPI高 = 滞胀/收缩

    cycle_score = 50.0

    # M2增速评分
    if m2_growth > 12:
        cycle_score += 15.0  # 流动性充裕
    elif m2_growth > 8:
        cycle_score += 5.0
    elif m2_growth < 6:
        cycle_score -= 10.0  # 流动性收紧

    # PMI评分
    if pmi > 52:
        cycle_score += 10.0  # 扩张
    elif pmi > 50:
        cycle_score += 5.0
    elif pmi < 48:
        cycle_score -= 15.0  # 收缩
    else:
        cycle_score -= 5.0

    # CPI评分（温和通胀好，过高或通缩差）
    if 1.5 <= cpi <= 3.0:
        cycle_score += 5.0  # 温和通胀
    elif cpi > 4.0:
        cycle_score -= 10.0  # 高通胀风险
    elif cpi < 0:
        cycle_score -= 15.0  # 通缩风险

    cycle_score = max(0.0, min(100.0, cycle_score))

    # 周期判断
    if cycle_score >= 70:
        phase = "扩张"
        score = 70.0
        reason = f"M2={m2_growth:.1f}% PMI={pmi:.1f} CPI={cpi:.1f}%，周期扩张期"
    elif cycle_score >= 50:
        phase = "温和"
        score = 55.0
        reason = f"M2={m2_growth:.1f}% PMI={pmi:.1f} CPI={cpi:.1f}%，周期温和期"
    elif cycle_score >= 30:
        phase = "收缩"
        score = 35.0
        reason = f"M2={m2_growth:.1f}% PMI={pmi:.1f} CPI={cpi:.1f}%，周期收缩期"
    else:
        phase = "谷底"
        score = 80.0
        reason = f"M2={m2_growth:.1f}% PMI={pmi:.1f} CPI={cpi:.1f}%，Marks：谷底=逆向机会"

    return AtomicJudgment(
        rule_id="marks_03", rule_name="周期定位", thinker="howard_marks",
        dimension="macro", score=score, confidence=0.6, data_quality=quality,
        direction="BUY" if score >= 65 else ("SELL" if score < 30 else "HOLD"),
        reason=reason,
        metadata={"phase": phase, "m2_growth": m2_growth, "pmi": pmi, "cpi": cpi, "cycle_score": cycle_score},
    )


def marks_04_second_level(ctx: EvalContext) -> AtomicJudgment | None:
    """Marks：第二层思维（简化版）

    不是"会怎样"，而是"共识是什么，共识错在哪"
    用情绪极端值+估值极端值近似判断
    """
    emotion_phase = ctx.market_data.get("emotion_phase", "")
    pe_percentile = ctx.market_data.get("pe_percentile", 50.0)
    quality = 0.5  # 第二层思维需要深度判断，自动化只能近似

    # 极端共识判断
    extreme_score = 50.0

    # 情绪极端
    if emotion_phase == "高潮":
        extreme_score += 25.0  # 极度乐观共识
    elif emotion_phase == "冰点":
        extreme_score -= 25.0  # 极度悲观共识
    elif emotion_phase == "退潮":
        extreme_score -= 15.0

    # 估值极端
    if pe_percentile > 80:
        extreme_score += 15.0  # 估值偏高
    elif pe_percentile < 20:
        extreme_score -= 15.0  # 估值偏低

    extreme_score = max(0.0, min(100.0, extreme_score))

    # 第二层思维：极端共识的反面
    if extreme_score >= 75:
        score = 30.0
        direction = "WARNING"
        reason = f"共识极度乐观（情绪{emotion_phase}+PE{pe_percentile:.0f}%分位），Marks：共识的反面=风险"
    elif extreme_score >= 60:
        score = 45.0
        direction = "HOLD"
        reason = "共识偏乐观，第二层思维：注意潜在失望"
    elif extreme_score <= 25:
        score = 80.0
        direction = "BUY"
        reason = f"共识极度悲观（情绪{emotion_phase}+PE{pe_percentile:.0f}%分位），Marks：共识的反面=机会"
    elif extreme_score <= 40:
        score = 65.0
        direction = "BUY"
        reason = "共识偏悲观，第二层思维：可能被低估"
    else:
        score = 50.0
        direction = "HOLD"
        reason = "共识中性，第二层思维无明显信号"

    return AtomicJudgment(
        rule_id="marks_04", rule_name="第二层思维", thinker="howard_marks",
        dimension="macro", score=score, confidence=0.5, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"extreme_score": extreme_score, "emotion_phase": emotion_phase, "pe_percentile": pe_percentile},
    )


def dalio_01_macro_cycle(ctx: EvalContext) -> AtomicJudgment | None:
    """Dalio：大周期阶段

    简化4阶段模型：
    早期扩张→75分（加杠杆）
    晚期扩张→40分（减杠杆）
    收缩→25分（防守）
    去杠杆尾声→85分（逆向布局）
    """
    m2_growth = ctx.market_data.get("m2_growth", 0)
    pmi = ctx.market_data.get("pmi", 50)
    leverage = ctx.market_data.get("leverage_ratio", 0)
    quality = 0.5

    # 简化阶段判断
    if m2_growth > 10 and pmi > 51:
        stage = "早期扩张"
        score = 75.0
        reason = f"M2高增{m2_growth:.1f}%+PMI扩张{pmi:.1f}，Dalio：早期扩张，加杠杆"
    elif m2_growth > 8 and pmi > 50:
        stage = "中期扩张"
        score = 60.0
        reason = f"M2={m2_growth:.1f}%+PMI={pmi:.1f}，扩张中期"
    elif pmi < 49:
        stage = "收缩"
        score = 25.0
        reason = f"PMI收缩{pmi:.1f}，Dalio：收缩期，防守为主"
    elif m2_growth < 7 and pmi < 50:
        stage = "去杠杆尾声"
        score = 85.0
        reason = f"M2={m2_growth:.1f}%+PMI={pmi:.1f}，Dalio：去杠杆尾声，逆向机会"
    else:
        stage = "过渡期"
        score = 50.0
        reason = f"M2={m2_growth:.1f}%+PMI={pmi:.1f}，周期过渡中"

    return AtomicJudgment(
        rule_id="dalio_01", rule_name="大周期阶段", thinker="ray_dalio",
        dimension="macro", score=score, confidence=0.55, data_quality=quality,
        direction="BUY" if score >= 65 else ("SELL" if score < 30 else "HOLD"),
        reason=reason,
        metadata={"stage": stage, "m2_growth": m2_growth, "pmi": pmi},
    )


def dalio_02_risk_parity(ctx: EvalContext) -> AtomicJudgment | None:
    """Dalio：风险平价信号

    股债性价比：PE倒数(盈利收益率) vs 10年国债收益率
    盈利收益率 > 国债收益率×1.5 → 股票有吸引力
    """
    pe_ttm = ctx.market_data.get("pe_ttm", 0)
    bond_yield = ctx.market_data.get("bond_yield_10y", 2.5)
    quality = 0.6

    if pe_ttm <= 0:
        return None

    earnings_yield = 1.0 / pe_ttm * 100  # 盈利收益率 (%)
    spread = earnings_yield - bond_yield

    if spread > 3.0:
        score = 85.0
        direction = "BUY"
        reason = f"盈利收益率{earnings_yield:.1f}%>>国债{bond_yield:.1f}%，Dalio：股票极具吸引力"
    elif spread > 1.5:
        score = 70.0
        direction = "BUY"
        reason = f"盈利收益率{earnings_yield:.1f}%>国债{bond_yield:.1f}%，股票有吸引力"
    elif spread > 0:
        score = 55.0
        direction = "HOLD"
        reason = f"盈利收益率{earnings_yield:.1f}%略>国债{bond_yield:.1f}%，性价比一般"
    else:
        score = 25.0
        direction = "WARNING"
        reason = f"盈利收益率{earnings_yield:.1f}%<国债{bond_yield:.1f}%，Dalio：债券更优"

    return AtomicJudgment(
        rule_id="dalio_02", rule_name="风险平价", thinker="ray_dalio",
        dimension="macro", score=score, confidence=0.65, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"earnings_yield": earnings_yield, "bond_yield": bond_yield, "spread": spread},
    )


def macro_01_liquidity(ctx: EvalContext) -> AtomicJudgment | None:
    """Marks+邱国鹭：流动性评分

    M2增速-GDP增速→超额流动性
    超额流动性>5%→80分（水涨船高）
    超额流动性<0%→20分（流动性收紧）
    """
    m2_growth = ctx.market_data.get("m2_growth", 0)
    gdp_growth = ctx.market_data.get("gdp_growth", 5.0)
    quality = 0.5

    excess_liquidity = m2_growth - gdp_growth

    if excess_liquidity > 5:
        score = 80.0
        direction = "BUY"
        reason = f"超额流动性{excess_liquidity:.1f}%充裕，水涨船高"
    elif excess_liquidity > 2:
        score = 65.0
        direction = "BUY"
        reason = f"超额流动性{excess_liquidity:.1f}%，流动性偏松"
    elif excess_liquidity > 0:
        score = 50.0
        direction = "HOLD"
        reason = f"超额流动性{excess_liquidity:.1f}%，流动性中性"
    else:
        score = 20.0
        direction = "SELL"
        reason = f"超额流动性{excess_liquidity:.1f}%为负，流动性收紧"

    return AtomicJudgment(
        rule_id="macro_01", rule_name="流动性评分", thinker="composite",
        dimension="macro", score=score, confidence=0.6, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"excess_liquidity": excess_liquidity, "m2_growth": m2_growth, "gdp_growth": gdp_growth},
    )


def macro_02_macro_composite(ctx: EvalContext) -> AtomicJudgment | None:
    """综合：宏观 = marks_01×0.25 + marks_03×0.25 + macro_01×0.25 + dalio_01×0.25"""
    scores = {}
    for rule_id, weight in [("marks_01", 0.25), ("marks_03", 0.25), ("macro_01", 0.25), ("dalio_01", 0.25)]:
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
        rule_id="macro_02", rule_name="宏观综合", thinker="composite",
        dimension="macro", score=round(composite, 1),
        confidence=0.6, data_quality=0.6,
        direction="BUY" if composite >= 65 else ("SELL" if composite < 30 else "HOLD"),
        reason=f"宏观综合评分{composite:.0f}",
    )


def register_all(registry) -> None:
    """注册所有宏观判断规则"""
    rules = [
        (marks_01_pendulum, "howard_marks"),
        (marks_02_consensus_deviation, "howard_marks"),
        (marks_03_cycle_position, "howard_marks"),
        (marks_04_second_level, "howard_marks"),
        (dalio_01_macro_cycle, "ray_dalio"),
        (dalio_02_risk_parity, "ray_dalio"),
        (macro_01_liquidity, "composite"),
        (macro_02_macro_composite, "composite"),
    ]
    for fn, thinker in rules:
        registry.register(fn.__name__, fn, "macro", thinker)
