"""
风险判断原子规则 — 8 条

来源大师：Klarman（卡拉曼）、交易系统资料
核心问题：这笔交易的风险是否可控？

Klarman核心理论：
- 「安全边际」：以低于内在价值的价格买入
- 「永久性亏损」：不是波动，是基本面恶化导致的永久损失
- 「仓位=置信度」：不确定性越高，押注越小

交易系统核心：
- 「数学期望」：E = (胜率×平均盈利) - (败率×平均亏损)，E<0拒绝
- 「三层熔断」：日-3%/周-5%/月-10%→仓位压缩
- 「情绪防火墙」：当日回撤>2%/连续3次止损→强制冷静
- 「避雷针」：退市风险/监管风险/减持风险
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.core.signal import AtomicJudgment, EvalContext
from src.core.utils import find_col

logger = logging.getLogger(__name__)

# lazy import arch (GARCH)
_arch_model = None

def _get_arch():
    global _arch_model
    if _arch_model is None:
        from arch import arch_model
        _arch_model = arch_model
    return _arch_model


def klm_04_permanent_loss(ctx: EvalContext) -> AtomicJudgment | None:
    """Klarman：永久性亏损概率

    基本面恶化信号→风险分
    检查：营收下滑+毛利率下降+现金流恶化
    """
    q = ctx.stock_data.get("quarterly", {})
    quality = ctx.quality.assess_dict(q, "quarterly") if ctx.quality else 0.5

    if not q:
        return None

    revenue_growth = q.get("revenue_growth", 0)
    gross_margin = q.get("gross_margin", 0)
    fcf_list = q.get("fcf_list", [])
    roe_list = q.get("roe_list", [])

    risk_signals = []
    risk_score = 0.0

    # 营收下滑
    if revenue_growth < -10:
        risk_signals.append(f"营收{revenue_growth:.1f}%大幅下滑")
        risk_score += 30.0
    elif revenue_growth < -3:
        risk_signals.append(f"营收{revenue_growth:.1f}%下滑")
        risk_score += 15.0

    # 毛利率下降
    if gross_margin < 15:
        risk_signals.append(f"毛利率{gross_margin:.1f}%极低")
        risk_score += 20.0

    # FCF恶化
    if fcf_list and len(fcf_list) >= 2:
        if fcf_list[-1] < 0 and fcf_list[-2] > 0:
            risk_signals.append("自由现金流转负")
            risk_score += 25.0
        elif fcf_list[-1] < 0:
            risk_signals.append("自由现金流持续为负")
            risk_score += 15.0

    # ROE恶化
    if roe_list and len(roe_list) >= 2:
        if roe_list[-1] < roe_list[-2] * 0.7:
            risk_signals.append("ROE大幅下降")
            risk_score += 20.0

    risk_score = min(100.0, risk_score)

    if risk_score >= 60:
        score = 15.0
        direction = "SELL"
        reason = f"Klarman：永久性亏损风险高——{'、'.join(risk_signals)}"
    elif risk_score >= 30:
        score = 35.0
        direction = "WARNING"
        reason = f"基本面风险信号：{'、'.join(risk_signals)}"
    elif risk_signals:
        score = 50.0
        direction = "HOLD"
        reason = f"轻度风险信号：{'、'.join(risk_signals)}"
    else:
        score = 80.0
        direction = "BUY"
        reason = "无永久性亏损信号，基本面健康"

    return AtomicJudgment(
        rule_id="klm_04", rule_name="永久性亏损概率", thinker="seth_klarman",
        dimension="risk", score=score, confidence=0.75, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"risk_signals": risk_signals, "risk_score": risk_score},
    )


def klm_05_position_by_confidence(ctx: EvalContext) -> AtomicJudgment | None:
    """Klarman：仓位=置信度（Kelly公式版）

    Kelly公式：f* = (p×b - q) / b
    p=胜率, b=盈亏比, q=1-p
    实际使用半Kelly（更保守）
    """
    composite_confidence = ctx.stock_data.get("composite_confidence", 0.5)
    win_rate = ctx.stock_data.get("estimated_win_rate", 0.5)
    avg_win = ctx.stock_data.get("avg_win_pct", 5.0)
    avg_loss = ctx.stock_data.get("avg_loss_pct", 3.0)
    quality = 0.7

    # Kelly公式计算
    max_position_pct = 25.0
    if avg_loss > 0:
        b = avg_win / avg_loss  # 盈亏比
    else:
        b = 1.0
    p = win_rate
    q = 1.0 - p
    kelly_full = (p * b - q) / b if b > 0 else 0
    kelly_full = max(0.0, min(kelly_full, 1.0))  # 限制在[0, 1]
    kelly_half = kelly_full / 2.0  # 半Kelly更保守

    suggested_position = kelly_half * max_position_pct

    # 评分仍基于置信度（Kelly影响仓位，唔影响评分）
    if composite_confidence >= 0.75:
        score = 75.0
        direction = "BUY"
    elif composite_confidence >= 0.5:
        score = 55.0
        direction = "HOLD"
    elif composite_confidence >= 0.3:
        score = 35.0
        direction = "WARNING"
    else:
        score = 15.0
        direction = "SELL"

    reason = (f"置信度{composite_confidence:.0%}，"
              f"Kelly最优{kelly_full:.0%}（半Kelly={kelly_half:.0%}），"
              f"建议仓位{suggested_position:.0f}%")

    return AtomicJudgment(
        rule_id="klm_05", rule_name="仓位建议", thinker="seth_klarman",
        dimension="risk", score=score, confidence=0.7, data_quality=quality,
        direction=direction, reason=reason,
        metadata={
            "composite_confidence": composite_confidence,
            "suggested_position_pct": round(suggested_position, 1),
            "kelly_full": round(kelly_full, 4),
            "kelly_half": round(kelly_half, 4),
            "win_rate": win_rate,
            "win_loss_ratio": round(b, 3),
        },
    )


def sys_01_math_expectation(ctx: EvalContext) -> AtomicJudgment | None:
    """交易系统：数学期望

    E = (胜率×平均盈利) - (败率×平均亏损)
    E<0→拒绝交易
    """
    win_rate = ctx.stock_data.get("estimated_win_rate", 0.5)
    avg_win = ctx.stock_data.get("avg_win_pct", 5.0)
    avg_loss = ctx.stock_data.get("avg_loss_pct", 3.0)
    quality = 0.5  # 期望值基于历史估算

    loss_rate = 1.0 - win_rate
    expectation = (win_rate * avg_win) - (loss_rate * avg_loss)

    if expectation > 2.0:
        score = 85.0
        direction = "BUY"
        reason = f"数学期望E=+{expectation:.1f}%，正期望值强"
    elif expectation > 0.5:
        score = 65.0
        direction = "BUY"
        reason = f"数学期望E=+{expectation:.1f}%，正期望值"
    elif expectation > 0:
        score = 50.0
        direction = "HOLD"
        reason = f"数学期望E=+{expectation:.1f}%，正但微弱"
    else:
        score = 15.0
        direction = "SELL"
        reason = f"数学期望E={expectation:.1f}%为负，交易系统：拒绝"

    return AtomicJudgment(
        rule_id="sys_01", rule_name="数学期望", thinker="trading_system",
        dimension="risk", score=score, confidence=0.6, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"expectation": expectation, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss},
    )


def sys_02_circuit_breaker(ctx: EvalContext) -> AtomicJudgment | None:
    """交易系统：三层熔断

    日回撤>-3%→仓位压缩50%
    周回撤>-5%→仓位压缩至20%
    月回撤>-10%→清仓观望
    """
    daily_drawdown = ctx.stock_data.get("daily_drawdown", 0)
    weekly_drawdown = ctx.stock_data.get("weekly_drawdown", 0)
    monthly_drawdown = ctx.stock_data.get("monthly_drawdown", 0)
    quality = 0.8  # 实时数据

    if monthly_drawdown < -10:
        score = 5.0
        direction = "SELL"
        reason = f"月回撤{monthly_drawdown:.1f}%>-10%，熔断：清仓观望"
    elif weekly_drawdown < -5:
        score = 15.0
        direction = "SELL"
        reason = f"周回撤{weekly_drawdown:.1f}%>-5%，熔断：仓位压缩至20%"
    elif daily_drawdown < -3:
        score = 30.0
        direction = "WARNING"
        reason = f"日回撤{daily_drawdown:.1f}%>-3%，熔断：仓位压缩50%"
    else:
        score = 70.0
        direction = "HOLD"
        reason = "未触发熔断"

    return AtomicJudgment(
        rule_id="sys_02", rule_name="三层熔断", thinker="trading_system",
        dimension="risk", score=score, confidence=0.85, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"daily_dd": daily_drawdown, "weekly_dd": weekly_drawdown, "monthly_dd": monthly_drawdown},
    )


def sys_03_emotion_firewall(ctx: EvalContext) -> AtomicJudgment | None:
    """交易系统：情绪防火墙

    当日回撤>2%→强制冷静2小时
    连续3次止损→强制停手1天
    """
    daily_drawdown = ctx.stock_data.get("daily_drawdown", 0)
    consecutive_stops = ctx.stock_data.get("consecutive_stop_losses", 0)
    quality = 0.8

    triggers = []

    if daily_drawdown < -2:
        triggers.append(f"日回撤{daily_drawdown:.1f}%>-2%")

    if consecutive_stops >= 3:
        triggers.append(f"连续{consecutive_stops}次止损")

    if triggers:
        score = 20.0
        direction = "SELL"
        reason = f"情绪防火墙触发：{'、'.join(triggers)}，强制冷静"
    else:
        score = 65.0
        direction = "HOLD"
        reason = "情绪防火墙未触发"

    return AtomicJudgment(
        rule_id="sys_03", rule_name="情绪防火墙", thinker="trading_system",
        dimension="risk", score=score, confidence=0.8, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"triggers": triggers, "consecutive_stops": consecutive_stops},
    )


def sys_04_delist_risk(ctx: EvalContext) -> AtomicJudgment | None:
    """交易系统：避雷针-退市风险

    连续亏损+营收不足+净资产为负→退市风险
    """
    q = ctx.stock_data.get("quarterly", {})
    f10 = ctx.stock_data.get("f10", {})
    quality = ctx.quality.assess_dict(q, "quarterly") if ctx.quality else 0.5

    if not q:
        return None

    net_margin = q.get("net_margin", 0)
    revenue = q.get("revenue", 0)
    net_assets = q.get("net_assets", 0)

    risk_signals = []

    # 连续亏损
    if net_margin < 0:
        risk_signals.append(f"净利率{net_margin:.1f}%<0")

    # 营收不足（退市新规：营收<3亿）
    if 0 < revenue < 3:
        risk_signals.append(f"营收{revenue:.1f}亿<3亿退市线")

    # 净资产为负
    if net_assets < 0:
        risk_signals.append(f"净资产{net_assets:.1f}亿<0")

    if len(risk_signals) >= 2:
        score = 5.0
        direction = "SELL"
        reason = f"退市风险极高：{'、'.join(risk_signals)}"
    elif risk_signals:
        score = 25.0
        direction = "WARNING"
        reason = f"退市风险信号：{'、'.join(risk_signals)}"
    else:
        score = 80.0
        direction = "HOLD"
        reason = "无退市风险信号"

    return AtomicJudgment(
        rule_id="sys_04", rule_name="退市风险", thinker="trading_system",
        dimension="risk", score=score, confidence=0.8, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"risk_signals": risk_signals},
    )


def sys_05_regulatory_risk(ctx: EvalContext) -> AtomicJudgment | None:
    """交易系统：避雷针-监管风险

    监管函/立案调查/大股东减持→风险信号
    """
    f10 = ctx.stock_data.get("f10", {})
    quality = ctx.quality.assess_dict(f10, "f10") if ctx.quality else 0.3

    if not f10:
        return None

    risk_signals = []
    announcements = f10.get("announcements", [])

    # 检查公告中的风险关键词
    risk_keywords = ["立案", "调查", "监管函", "警示函", "处罚", "违规", "减持", "质押"]
    for ann in announcements:
        title = str(ann.get("title", ""))
        for kw in risk_keywords:
            if kw in title:
                risk_signals.append(f"{kw}: {title[:30]}")
                break

    if len(risk_signals) >= 2:
        score = 10.0
        direction = "SELL"
        reason = f"多重监管风险：{'、'.join(risk_signals[:2])}"
    elif risk_signals:
        score = 30.0
        direction = "WARNING"
        reason = f"监管风险信号：{risk_signals[0]}"
    else:
        score = 75.0
        direction = "HOLD"
        reason = "无监管风险信号"

    return AtomicJudgment(
        rule_id="sys_05", rule_name="监管风险", thinker="trading_system",
        dimension="risk", score=score, confidence=0.6, data_quality=quality,
        direction=direction, reason=reason,
        metadata={"risk_signals": risk_signals},
    )


def sys_06_volatility_regime(ctx: EvalContext) -> AtomicJudgment | None:
    """交易系统：GARCH波动率预测

    用GARCH(1,1)模型预测未来5日波动率
    波动率聚集效应：大波动后大概率跟大波动
    高波动→收紧仓位/止损，低波动→正常操作
    """
    kline = ctx.stock_data.get("kline")
    quality = ctx.quality.assess_dataframe(kline, "kline_daily") if ctx.quality else 0.5

    if kline is None or kline.empty:
        return None

    try:
        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 30:
            return None

        # 日对数收益率（百分比）
        returns = np.log(prices / prices.shift(1)).dropna() * 100

        if len(returns) < 30:
            return None

        # 检查收益率方差是否为零（常数价格）
        if returns.std() < 1e-10:
            return None

        # 拟合GARCH(1,1)
        arch_model = _get_arch()
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        result = model.fit(disp='off', show_warning=False)

        # 当前条件波动率
        current_vol = result.conditional_volatility.iloc[-1]

        # 预测未来5日波动率
        forecasts = result.forecast(horizon=5)
        predicted_vol = np.sqrt(forecasts.variance.iloc[-1].mean())

        # GARCH参数
        omega = result.params.get('omega', 0)
        alpha = result.params.get('alpha[1]', 0)
        beta = result.params.get('beta[1]', 0)

        # 评分映射（阈值从config读取）
        garch_cfg = ctx.config.get("models", {}).get("garch", {})
        vol_high = garch_cfg.get("vol_high_threshold", 4.0)
        vol_med = garch_cfg.get("vol_med_threshold", 2.5)
        vol_low = garch_cfg.get("vol_low_threshold", 1.5)

        if predicted_vol > vol_high:
            score = 15.0
            direction = "SELL"
            reason = f"GARCH预测5日波动率{predicted_vol:.2f}%极高，风险骤增"
        elif predicted_vol > vol_med:
            score = 35.0
            direction = "WARNING"
            reason = f"GARCH预测5日波动率{predicted_vol:.2f}%偏高，注意风控"
        elif predicted_vol > vol_low:
            score = 55.0
            direction = "HOLD"
            reason = f"GARCH预测5日波动率{predicted_vol:.2f}%正常"
        else:
            score = 75.0
            direction = "BUY"
            reason = f"GARCH预测5日波动率{predicted_vol:.2f}%低，波动率收缩"

        return AtomicJudgment(
            rule_id="sys_06", rule_name="波动率预测", thinker="trading_system",
            dimension="risk", score=score, confidence=0.7, data_quality=quality,
            direction=direction, reason=reason,
            metadata={
                "predicted_vol_5d": round(float(predicted_vol), 4),
                "current_vol": round(float(current_vol), 4),
                "garch_omega": round(float(omega), 6),
                "garch_alpha": round(float(alpha), 4),
                "garch_beta": round(float(beta), 4),
            },
        )
    except Exception as e:
        logger.debug(f"GARCH拟合失败: {e}")
        return None


def risk_01_risk_composite(ctx: EvalContext) -> AtomicJudgment | None:
    """综合：风险 = max(klm_04, sys_01, sys_02, sys_03) — 取最高风险

    风险维度用max而非加权：木桶原理，最弱的环节决定风险
    """
    risk_rule_ids = ["klm_04", "sys_01", "sys_02", "sys_03", "sys_04", "sys_05", "sys_06"]
    scores = []
    for rule_id in risk_rule_ids:
        existing = ctx.stock_data.get("_judgments", {}).get(rule_id)
        if existing:
            scores.append(existing.score)

    if not scores:
        return None

    # 风险取最低分（最高风险）
    composite = min(scores)

    return AtomicJudgment(
        rule_id="risk_01", rule_name="风险综合", thinker="composite",
        dimension="risk", score=round(composite, 1),
        confidence=0.7, data_quality=0.6,
        direction="BUY" if composite >= 65 else ("SELL" if composite < 25 else "HOLD"),
        reason=f"风险综合评分{composite:.0f}（取最高风险）",
    )


def register_all(registry) -> None:
    """注册所有风险判断规则"""
    rules = [
        (klm_04_permanent_loss, "seth_klarman"),
        (klm_05_position_by_confidence, "seth_klarman"),
        (sys_01_math_expectation, "trading_system"),
        (sys_02_circuit_breaker, "trading_system"),
        (sys_03_emotion_firewall, "trading_system"),
        (sys_04_delist_risk, "trading_system"),
        (sys_05_regulatory_risk, "trading_system"),
        (sys_06_volatility_regime, "trading_system"),
        (risk_01_risk_composite, "composite"),
    ]
    for fn, thinker in rules:
        registry.register(fn.__name__, fn, "risk", thinker)
