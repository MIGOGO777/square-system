"""
趋势判断原子规则 — 6 条

来源大师：Livermore（利弗莫尔）
核心问题：趋势方向是否支持这个操作？

Livermore核心理论：
- 「最小阻力线」：价格沿阻力最小的方向移动
- 「关键点」：突破整数关口/前高 = 趋势确认
- 「金字塔加仓」：确认后逐步加仓，不一次满仓
- 「止损」：亏损>5%必须止损，不抱幻想
"""

from __future__ import annotations

import logging

import numpy as np

from src.core.signal import AtomicJudgment, EvalContext
from src.core.utils import find_col

logger = logging.getLogger(__name__)

# lazy import scipy.stats (Jarque-Bera test)
_scipy_stats = None

def _get_scipy_stats():
    global _scipy_stats
    if _scipy_stats is None:
        from scipy import stats as _s
        _scipy_stats = _s
    return _scipy_stats


def lvr_01_least_resistance(ctx: EvalContext) -> AtomicJudgment | None:
    """Livermore：最小阻力线

    价格沿某方向移动时，成交量变化→阻力方向判断
    上涨放量→阻力向上（顺势），上涨缩量→阻力向下（假突破）
    """
    kline = ctx.stock_data.get("kline")
    quality = ctx.quality.assess_dataframe(kline, "kline_daily") if ctx.quality else 0.5

    if kline is None or kline.empty:
        return None

    try:
        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        vol_col = find_col(kline, ["volume", "成交量", "vol"])
        if close_col is None or vol_col is None:
            return None

        prices = kline[close_col].dropna()
        volumes = kline[vol_col].dropna()
        if len(prices) < 10 or len(volumes) < 10:
            return None

        # 最近10日方向
        recent_10 = prices.iloc[-10:]
        price_change = (recent_10.iloc[-1] / recent_10.iloc[0] - 1) * 100

        # 成交量趋势
        vol_recent = volumes.iloc[-5:].mean()
        vol_prior = volumes.iloc[-10:-5].mean()
        vol_ratio = vol_recent / vol_prior if vol_prior > 0 else 1.0

        # 最小阻力判断
        if price_change > 3 and vol_ratio > 1.2:
            score = 75.0
            direction = "BUY"
            reason = f"10日涨{price_change:.1f}%+放量{vol_ratio:.1f}x，最小阻力向上"
        elif price_change > 3 and vol_ratio < 0.8:
            score = 40.0
            direction = "WARNING"
            reason = f"10日涨{price_change:.1f}%但缩量，可能假突破"
        elif price_change < -3 and vol_ratio > 1.2:
            score = 25.0
            direction = "SELL"
            reason = f"10日跌{price_change:.1f}%+放量，最小阻力向下"
        elif price_change < -3 and vol_ratio < 0.8:
            score = 45.0
            direction = "HOLD"
            reason = f"10日跌{price_change:.1f}%但缩量，抛压减弱"
        else:
            score = 50.0
            direction = "HOLD"
            reason = f"10日变动{price_change:.1f}%，方向不明"

        return AtomicJudgment(
            rule_id="lvr_01", rule_name="最小阻力线", thinker="livermore",
            dimension="trend", score=score, confidence=0.7, data_quality=quality,
            direction=direction, reason=reason,
            metadata={"price_change_10d": price_change, "vol_ratio": vol_ratio},
        )
    except Exception:
        return None


def lvr_02_key_point(ctx: EvalContext) -> AtomicJudgment | None:
    """Livermore：关键点突破

    价格突破整数关口（100/200/300/500）→趋势确认信号
    价格突破近期前高→趋势延续信号
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
        if len(prices) < 20:
            return None

        current = prices.iloc[-1]

        # 检查是否突破整数关口
        key_levels = [10, 20, 30, 50, 100, 150, 200, 300, 500, 1000]
        near_key = False
        for level in key_levels:
            if level * 0.98 <= current <= level * 1.02:
                near_key = True
                break

        # 检查是否突破20日前高
        recent_20_high = prices.iloc[-20:].max()
        broke_high = current >= recent_20_high * 0.99

        if near_key and broke_high:
            score = 85.0
            direction = "BUY"
            reason = f"价格{current:.2f}突破整数关口+20日前高，Livermore：关键点确认"
        elif broke_high:
            score = 70.0
            direction = "BUY"
            reason = f"价格{current:.2f}突破20日前高{recent_20_high:.2f}，趋势延续"
        elif near_key:
            score = 60.0
            direction = "HOLD"
            reason = f"价格{current:.2f}接近整数关口，等待突破确认"
        else:
            score = 50.0
            direction = "HOLD"
            reason = f"价格{current:.2f}无关键点信号"

        return AtomicJudgment(
            rule_id="lvr_02", rule_name="关键点突破", thinker="livermore",
            dimension="trend", score=score, confidence=0.75, data_quality=quality,
            direction=direction, reason=reason,
            metadata={"current_price": current, "recent_20_high": recent_20_high, "near_key": near_key},
        )
    except Exception:
        return None


def lvr_03_trend_confirm(ctx: EvalContext) -> AtomicJudgment | None:
    """Livermore：趋势确认

    价格 > N日MA 且成交量放大 → 趋势确认
    多头排列（MA5>MA10>MA20>MA60）→ 强势确认
    """
    kline = ctx.stock_data.get("kline")
    quality = ctx.quality.assess_dataframe(kline, "kline_daily") if ctx.quality else 0.5

    if kline is None or kline.empty:
        return None

    try:
        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        vol_col = find_col(kline, ["volume", "成交量", "vol"])
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 60:
            return None

        current = prices.iloc[-1]

        # 计算均线
        ma5 = prices.iloc[-5:].mean()
        ma10 = prices.iloc[-10:].mean()
        ma20 = prices.iloc[-20:].mean()
        ma60 = prices.iloc[-60:].mean()

        # 多头排列检查
        bullish_align = ma5 > ma10 > ma20 > ma60
        above_ma20 = current > ma20

        # 成交量确认
        vol_confirm = False
        if vol_col:
            volumes = kline[vol_col].dropna()
            if len(volumes) >= 10:
                vol_recent = volumes.iloc[-5:].mean()
                vol_prior = volumes.iloc[-10:-5].mean()
                vol_confirm = vol_recent > vol_prior * 1.1

        if bullish_align and vol_confirm:
            score = 85.0
            direction = "BUY"
            reason = f"多头排列(MA5>MA10>MA20>MA60)+放量确认，趋势强"
        elif bullish_align:
            score = 75.0
            direction = "BUY"
            reason = "多头排列，趋势确认，但缺量能配合"
        elif above_ma20 and vol_confirm:
            score = 65.0
            direction = "HOLD"
            reason = f"价格在MA20上方+放量，趋势偏多"
        elif above_ma20:
            score = 55.0
            direction = "HOLD"
            reason = "价格在MA20上方，趋势中性偏多"
        else:
            score = 35.0
            direction = "WARNING"
            reason = f"价格在MA20({ma20:.2f})下方，趋势偏弱"

        return AtomicJudgment(
            rule_id="lvr_03", rule_name="趋势确认", thinker="livermore",
            dimension="trend", score=score, confidence=0.75, data_quality=quality,
            direction=direction, reason=reason,
            metadata={"ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60, "bullish_align": bullish_align},
        )
    except Exception:
        return None


def lvr_04_stop_loss(ctx: EvalContext) -> AtomicJudgment | None:
    """Livermore：止损信号

    亏损>5%→SELL信号
    价格跌破关键支撑（MA60或前低）→SELL信号
    """
    kline = ctx.stock_data.get("kline")
    buy_price = ctx.stock_data.get("buy_price", 0)
    quality = ctx.quality.assess_dataframe(kline, "kline_daily") if ctx.quality else 0.5

    if kline is None or kline.empty:
        return None

    try:
        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 60:
            return None

        current = prices.iloc[-1]

        # 亏损计算
        loss_pct = 0.0
        if buy_price > 0:
            loss_pct = (current / buy_price - 1) * 100

        # 跌破MA60
        ma60 = prices.iloc[-60:].mean()
        below_ma60 = current < ma60

        # 跌破20日前低
        recent_20_low = prices.iloc[-20:].min()
        broke_low = current <= recent_20_low * 1.01

        if loss_pct < -5:
            score = 10.0
            direction = "SELL"
            reason = f"亏损{loss_pct:.1f}%>-5%，Livermore：必须止损"
        elif below_ma60 and broke_low:
            score = 20.0
            direction = "SELL"
            reason = f"跌破MA60({ma60:.2f})+破前低{recent_20_low:.2f}，趋势破坏"
        elif below_ma60:
            score = 35.0
            direction = "WARNING"
            reason = f"价格在MA60({ma60:.2f})下方，趋势偏弱"
        elif broke_low:
            score = 40.0
            direction = "WARNING"
            reason = f"触及20日前低{recent_20_low:.2f}，支撑受考验"
        else:
            score = 60.0
            direction = "HOLD"
            reason = "未触发止损信号"

        return AtomicJudgment(
            rule_id="lvr_04", rule_name="止损信号", thinker="livermore",
            dimension="trend", score=score, confidence=0.85, data_quality=quality,
            direction=direction, reason=reason,
            metadata={"loss_pct": loss_pct, "below_ma60": below_ma60, "buy_price": buy_price},
        )
    except Exception:
        return None


def lvr_05_pyramid(ctx: EvalContext) -> AtomicJudgment | None:
    """Livermore：金字塔加仓信号

    确认趋势后，每次加仓<初始仓位50%
    当前盈利+趋势确认→可以加仓信号
    """
    kline = ctx.stock_data.get("kline")
    buy_price = ctx.stock_data.get("buy_price", 0)
    current_position = ctx.stock_data.get("position_pct", 0)
    quality = ctx.quality.assess_dataframe(kline, "kline_daily") if ctx.quality else 0.5

    if kline is None or kline.empty or buy_price <= 0:
        return None

    try:
        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 20:
            return None

        current = prices.iloc[-1]
        profit_pct = (current / buy_price - 1) * 100

        # 趋势确认：价格在MA20上方
        ma20 = prices.iloc[-20:].mean()
        trend_ok = current > ma20

        # 盈利>5%且趋势确认→可以加仓
        if profit_pct > 5 and trend_ok:
            score = 70.0
            direction = "BUY"
            reason = f"盈利{profit_pct:.1f}%+趋势确认，Livermore：可金字塔加仓"
        elif profit_pct > 3 and trend_ok:
            score = 60.0
            direction = "HOLD"
            reason = f"盈利{profit_pct:.1f}%，趋势尚可，观望为主"
        else:
            score = 50.0
            direction = "HOLD"
            reason = f"盈利{profit_pct:.1f}%，不满足加仓条件"

        return AtomicJudgment(
            rule_id="lvr_05", rule_name="金字塔加仓", thinker="livermore",
            dimension="trend", score=score, confidence=0.6, data_quality=quality,
            direction=direction, reason=reason,
            metadata={"profit_pct": profit_pct, "buy_price": buy_price, "current_position": current_position},
        )
    except Exception:
        return None


def lvr_06_gbm_drift(ctx: EvalContext) -> AtomicJudgment | None:
    """Livermore：GBM漂移率估计

    用几何布朗运动模型估算价格漂移率μ和波动率σ
    μ = E[ln(P_t/P_{t-1})] × 252（年化）
    σ = std[ln(P_t/P_{t-1})] × sqrt(252)（年化）
    同时做Jarque-Bera正态性检验，判断GBM假设是否成立
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

        # 日对数收益率
        log_returns = np.log(prices / prices.shift(1)).dropna()

        if len(log_returns) < 30:
            return None

        # 年化漂移率和波动率
        mu = float(log_returns.mean() * 252)
        sigma = float(log_returns.std(ddof=1) * np.sqrt(252))

        # Jarque-Bera正态性检验
        stats = _get_scipy_stats()
        jb_stat, jb_pvalue = stats.jarque_bera(log_returns)
        returns_normal = jb_pvalue >= 0.05

        # 评分映射（阈值从config读取）
        gbm_cfg = ctx.config.get("models", {}).get("gbm", {})
        mu_strong = gbm_cfg.get("mu_strong", 0.15)
        mu_moderate = gbm_cfg.get("mu_moderate", 0.05)

        if mu > mu_strong and returns_normal:
            score = 80.0
            direction = "BUY"
            reason = f"GBM漂移率μ={mu:.1%}强上升，收益率正态分布"
        elif mu > mu_moderate:
            score = 65.0
            direction = "BUY"
            reason = f"GBM漂移率μ={mu:.1%}温和上升"
        elif mu > -mu_moderate:
            score = 50.0
            direction = "HOLD"
            reason = f"GBM漂移率μ={mu:.1%}接近零，方向不明"
        elif mu > -mu_strong:
            score = 35.0
            direction = "WARNING"
            reason = f"GBM漂移率μ={mu:.1%}偏负"
        else:
            score = 20.0
            direction = "SELL"
            reason = f"GBM漂移率μ={mu:.1%}强下降"

        # 置信度调整：收益率不正态→GBM假设不成立，降低置信度
        base_confidence = 0.65
        if not returns_normal:
            base_confidence -= 0.15

        return AtomicJudgment(
            rule_id="lvr_06", rule_name="GBM漂移率", thinker="livermore",
            dimension="trend", score=score, confidence=base_confidence, data_quality=quality,
            direction=direction, reason=reason,
            metadata={
                "mu_annualized": round(mu, 4),
                "sigma_annualized": round(sigma, 4),
                "jb_stat": round(float(jb_stat), 4),
                "jb_pvalue": round(float(jb_pvalue), 4),
                "returns_normal": returns_normal,
            },
        )
    except Exception as e:
        logger.debug(f"GBM漂移率计算失败: {e}")
        return None


def trend_01_trend_composite(ctx: EvalContext) -> AtomicJudgment | None:
    """综合：趋势 = lvr_01×0.2 + lvr_02×0.2 + lvr_03×0.25 + lvr_04×0.15 + lvr_06×0.2"""
    scores = {}
    for rule_id, weight in [("lvr_01", 0.2), ("lvr_02", 0.2), ("lvr_03", 0.25), ("lvr_04", 0.15), ("lvr_06", 0.2)]:
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
        rule_id="trend_01", rule_name="趋势综合", thinker="composite",
        dimension="trend", score=round(composite, 1),
        confidence=0.7, data_quality=0.7,
        direction="BUY" if composite >= 65 else ("SELL" if composite < 35 else "HOLD"),
        reason=f"趋势综合评分{composite:.0f}",
    )


def register_all(registry) -> None:
    """注册所有趋势判断规则"""
    rules = [
        (lvr_01_least_resistance, "livermore"),
        (lvr_02_key_point, "livermore"),
        (lvr_03_trend_confirm, "livermore"),
        (lvr_04_stop_loss, "livermore"),
        (lvr_05_pyramid, "livermore"),
        (lvr_06_gbm_drift, "livermore"),
        (trend_01_trend_composite, "composite"),
    ]
    for fn, thinker in rules:
        registry.register(fn.__name__, fn, "trend", thinker)
