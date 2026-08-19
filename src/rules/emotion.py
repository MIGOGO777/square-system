"""
情绪判断原子规则 — 8 条

来源大师：炒股养家、乌合之众（勒庞）
核心问题：市场情绪是否支持这个操作？

炒股养家：「买在分歧，卖在共识」
勒庞：「群体不善于推理，但急于行动」
"""

from __future__ import annotations

import logging

from src.core.signal import AtomicJudgment, EvalContext

logger = logging.getLogger(__name__)

# 情绪阶段 → 评分映射
PHASE_SCORES = {
    "冰点": (90.0, "BUY", "😱冰点：群体恐慌，逆向布局窗口"),
    "试探": (75.0, "BUY", "🌱试探：情绪回暖，逐步建仓"),
    "发酵": (80.0, "BUY", "🔥发酵：主升段，积极持仓"),
    "高潮": (30.0, "WARNING", "🚀高潮：市场最热，准备减仓"),
    "分歧": (50.0, "HOLD", "⚡分歧：方向不明，观望为主"),
    "退潮": (10.0, "SELL", "📉退潮：清仓离场"),
}


def cgyj_01_emotion_phase(ctx: EvalContext) -> AtomicJudgment | None:
    """炒股养家：情绪阶段

    冰点→90分(逆向) / 试探→75 / 发酵→80 / 高潮→30 / 分歧→50 / 退潮→10
    """
    phase = ctx.market_data.get("emotion_phase", "")
    temp = ctx.market_data.get("emotion_temp", 50.0)
    quality = 0.8  # 情绪数据基于实时涨停池，质量较高

    if not phase:
        return None

    score, direction, reason = PHASE_SCORES.get(phase, (50.0, "HOLD", f"未知阶段: {phase}"))

    return AtomicJudgment(
        rule_id="cgyj_01", rule_name="情绪阶段", thinker="chaoguyangjia",
        dimension="emotion", score=score, confidence=0.8, data_quality=quality,
        direction=direction, reason=f"{reason}（温度{temp:.0f}）",
        metadata={"phase": phase, "temperature": temp},
    )


def cgyj_02_divergence(ctx: EvalContext) -> AtomicJudgment | None:
    """炒股养家：分歧度

    涨停数vs炸板率背离→分歧信号
    涨停多+炸板高 = 分歧（市场意见不统一）
    涨停多+炸板低 = 共识（一致看好，危险）
    涨停少+炸板低 = 冰点（一致看空，机会）
    """
    limit_up = ctx.market_data.get("limit_up_count", 0)
    break_rate = ctx.market_data.get("break_rate", 0.0)
    quality = 0.8

    if limit_up <= 0:
        return None

    # 分歧度 = 涨停数与炸板率的背离程度
    # 正常关系：涨停多→炸板低，涨停少→炸板高
    # 背离 = 实际炸板率 - 预期炸板率
    if limit_up >= 80:
        expected_break = 10.0
    elif limit_up >= 50:
        expected_break = 18.0
    elif limit_up >= 30:
        expected_break = 25.0
    else:
        expected_break = 35.0

    divergence = break_rate - expected_break

    if divergence > 15:
        score = 35.0
        direction = "WARNING"
        reason = f"高度分歧：涨停{limit_up}家但炸板{break_rate:.0f}%（预期{expected_break:.0f}%）"
    elif divergence > 5:
        score = 50.0
        direction = "HOLD"
        reason = f"轻度分歧：涨停{limit_up}家，炸板{break_rate:.0f}%"
    elif divergence > -10:
        score = 70.0
        direction = "BUY"
        reason = f"共识形成：涨停{limit_up}家，炸板{break_rate:.0f}%低位"
    else:
        score = 80.0
        direction = "BUY"
        reason = f"强共识：涨停{limit_up}家，炸板极低{break_rate:.0f}%"

    return AtomicJudgment(
        rule_id="cgyj_02", rule_name="分歧度", thinker="chaoguyangjia",
        dimension="emotion", score=score, confidence=0.7, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"limit_up": limit_up, "break_rate": break_rate, "divergence": divergence},
    )


def cgyj_03_leader_height(ctx: EvalContext) -> AtomicJudgment | None:
    """炒股养家：龙头辨识

    连板高度+板块带动效应
    连板≥7→市场极热（高潮信号），连板≥5→发酵信号，连板≤2→冰点信号
    """
    height = ctx.market_data.get("leader_height", 0)
    quality = 0.8

    if height <= 0:
        return None

    if height >= 7:
        score = 30.0
        direction = "WARNING"
        reason = f"连板{height}板，市场极热，高潮/分歧风险"
    elif height >= 5:
        score = 70.0
        direction = "BUY"
        reason = f"连板{height}板，市场发酵中，龙头效应强"
    elif height >= 3:
        score = 60.0
        direction = "HOLD"
        reason = f"连板{height}板，市场试探中"
    else:
        score = 40.0
        direction = "HOLD"
        reason = f"连板{height}板，市场偏冷"

    return AtomicJudgment(
        rule_id="cgyj_03", rule_name="龙头辨识", thinker="chaoguyangjia",
        dimension="emotion", score=score, confidence=0.7, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"leader_height": height},
    )


def cgyj_04_win_rate_threshold(ctx: EvalContext) -> AtomicJudgment | None:
    """炒股养家：胜率门槛

    估算胜率<60%→不行动，>90%→重仓
    基于情绪阶段+趋势确认综合估算
    """
    phase = ctx.market_data.get("emotion_phase", "")
    macro_regime = ctx.market_data.get("macro_regime", "SIDEWAYS")
    quality = 0.5  # 胜率估算是推断值，质量中等

    # 简化胜率估算
    base_win = 50.0
    if phase in ("发酵", "冰点"):
        base_win += 15.0
    elif phase in ("高潮", "退潮"):
        base_win -= 15.0

    if macro_regime == "BULL":
        base_win += 10.0
    elif macro_regime == "BEAR":
        base_win -= 10.0

    estimated_win = max(0.0, min(100.0, base_win))

    if estimated_win >= 90:
        score = 90.0
        direction = "BUY"
        reason = f"估算胜率{estimated_win:.0f}%>90%，炒股养家：可以重仓"
    elif estimated_win >= 60:
        score = 60.0
        direction = "HOLD"
        reason = f"估算胜率{estimated_win:.0f}%，可以轻仓参与"
    else:
        score = 20.0
        direction = "WARNING"
        reason = f"估算胜率{estimated_win:.0f}%<60%，炒股养家：控制手，等待"

    return AtomicJudgment(
        rule_id="cgyj_04", rule_name="胜率门槛", thinker="chaoguyangjia",
        dimension="emotion", score=score, confidence=0.5, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"estimated_win_rate": estimated_win},
    )


def crowd_01_contagion(ctx: EvalContext) -> AtomicJudgment | None:
    """乌合之众：群体传染度

    涨停板块扩散度（板块数越多→传染越广）
    """
    limit_up_pool = ctx.market_data.get("limit_up_pool")
    quality = 0.8

    if limit_up_pool is None or limit_up_pool.empty:
        return None

    # 统计涨停股票涉及的板块数
    sector_col = None
    for col in ["所属行业", "行业", "板块", "概念"]:
        if col in limit_up_pool.columns:
            sector_col = col
            break

    if sector_col is None:
        return None

    unique_sectors = limit_up_pool[sector_col].nunique()
    total_stocks = len(limit_up_pool)

    if unique_sectors >= 15:
        score = 85.0
        reason = f"涨停扩散到{unique_sectors}个板块，群体传染极广"
    elif unique_sectors >= 10:
        score = 65.0
        reason = f"涨停扩散到{unique_sectors}个板块，传染较广"
    elif unique_sectors >= 5:
        score = 45.0
        reason = f"涨停集中在{unique_sectors}个板块，传染有限"
    else:
        score = 25.0
        reason = f"涨停仅{unique_sectors}个板块，传染极窄"

    return AtomicJudgment(
        rule_id="crowd_01", rule_name="群体传染度", thinker="le_bon",
        dimension="emotion", score=score, confidence=0.65, data_quality=quality,
        direction="BUY" if score >= 60 else "HOLD",
        reason=reason,
        metadata={"unique_sectors": unique_sectors, "total_stocks": total_stocks},
    )


def crowd_02_consensus_level(ctx: EvalContext) -> AtomicJudgment | None:
    """乌合之众+Marks：共识一致性

    北向资金+涨停扩散+媒体→共识过高=反转风险
    共识一致时，群体过度乐观/悲观 = 反转信号
    """
    north_flow = ctx.market_data.get("north_flow")
    limit_up = ctx.market_data.get("limit_up_count", 0)
    phase = ctx.market_data.get("emotion_phase", "")
    quality = 0.7

    # 计算共识度
    consensus_score = 50.0

    # 北向资金方向一致性
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
                    consensus_score += 20.0  # 外资一致看多
                elif flow_val < -100:
                    consensus_score -= 20.0  # 外资一致看空
        except Exception:
            pass

    # 涨停数作为共识度代理
    if limit_up >= 100:
        consensus_score += 15.0  # 大面积涨停 = 强共识
    elif limit_up <= 20:
        consensus_score -= 15.0  # 涨停极少 = 弱共识

    # 情绪阶段修正
    if phase == "高潮":
        consensus_score += 15.0  # 高潮 = 极端共识
    elif phase == "冰点":
        consensus_score -= 15.0  # 冰点 = 极端共识（反向）

    consensus_score = max(0.0, min(100.0, consensus_score))

    # 共识过高 = 反转风险，共识过低 = 机会
    if consensus_score >= 80:
        score = 30.0
        direction = "WARNING"
        reason = f"共识度{consensus_score:.0f}极高，勒庞：群体过度一致，反转临近"
    elif consensus_score >= 60:
        score = 55.0
        direction = "HOLD"
        reason = f"共识度{consensus_score:.0f}偏高，注意风险"
    elif consensus_score >= 40:
        score = 65.0
        direction = "BUY"
        reason = f"共识度{consensus_score:.0f}适中，分歧中有机会"
    else:
        score = 75.0
        direction = "BUY"
        reason = f"共识度{consensus_score:.0f}极低，群体分歧，可能是布局窗口"

    return AtomicJudgment(
        rule_id="crowd_02", rule_name="共识一致性", thinker="le_bon",
        dimension="emotion", score=score, confidence=0.6, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"consensus_score": consensus_score},
    )


def crowd_03_sentiment_miss(ctx: EvalContext) -> AtomicJudgment | None:
    """乌合之众：情绪错杀

    板块退潮中被连带下跌但基本面无变化的个股
    需要个股近期跌幅数据和基本面数据对比
    """
    kline = ctx.stock_data.get("kline")
    q = ctx.stock_data.get("quarterly", {})
    phase = ctx.market_data.get("emotion_phase", "")
    quality = 0.5

    if kline is None or kline.empty or not q:
        return None

    # 检查近期跌幅
    try:
        close_col = None
        for col in ["close", "收盘", "收盘价"]:
            if col in kline.columns:
                close_col = col
                break
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 5:
            return None

        recent_5d_return = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
        net_margin = q.get("net_margin", 0)
        roe_list = q.get("roe_list", [])

        # 情绪错杀条件：近期下跌 + 基本面没恶化
        if recent_5d_return < -5 and net_margin > 10 and (not roe_list or roe_list[-1] > 15):
            score = 80.0
            direction = "BUY"
            reason = f"近5日{recent_5d_return:+.1f}%但净利率{net_margin:.1f}%+ROE良好，可能被错杀"
        elif recent_5d_return < -3 and net_margin > 5:
            score = 60.0
            direction = "HOLD"
            reason = f"近5日{recent_5d_return:+.1f}%，基本面尚可，观察是否错杀"
        else:
            score = 50.0
            direction = "HOLD"
            reason = f"近5日{recent_5d_return:+.1f}%，无明显错杀信号"

        return AtomicJudgment(
            rule_id="crowd_03", rule_name="情绪错杀", thinker="le_bon",
            dimension="emotion", score=score, confidence=0.5, data_quality=quality,
            direction=direction, reason=reason,
            metadata={"recent_5d_return": recent_5d_return, "net_margin": net_margin},
        )
    except Exception:
        return None


def emo_01_emotion_composite(ctx: EvalContext) -> AtomicJudgment | None:
    """综合：情绪 = cgyj_01×0.3 + cgyj_02×0.2 + crowd_01×0.2 + crowd_02×0.3"""
    scores = {}
    for rule_id, weight in [("cgyj_01", 0.3), ("cgyj_02", 0.2), ("crowd_01", 0.2), ("crowd_02", 0.3)]:
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
        rule_id="emo_01", rule_name="情绪综合", thinker="composite",
        dimension="emotion", score=round(composite, 1),
        confidence=0.65, data_quality=0.7,
        direction="BUY" if composite >= 60 else ("SELL" if composite < 30 else "HOLD"),
        reason=f"情绪综合评分{composite:.0f}",
    )


def register_all(registry) -> None:
    """注册所有情绪判断规则"""
    rules = [
        (cgyj_01_emotion_phase, "chaoguyangjia"),
        (cgyj_02_divergence, "chaoguyangjia"),
        (cgyj_03_leader_height, "chaoguyangjia"),
        (cgyj_04_win_rate_threshold, "chaoguyangjia"),
        (crowd_01_contagion, "le_bon"),
        (crowd_02_consensus_level, "le_bon"),
        (crowd_03_sentiment_miss, "le_bon"),
        (emo_01_emotion_composite, "composite"),
    ]
    for fn, thinker in rules:
        registry.register(fn.__name__, fn, "emotion", thinker)
