"""
Black-Scholes期权定价 — IV vs RV偏离检测

A股只有ETF期权（50ETF/300ETF等），无个股期权。
本模块的价值：提取隐含波动率(IV)与已实现波动率(RV)的偏离。

IV > RV → 期权定价偏高（做空波动率机会）
IV < RV → 期权定价偏低（做多波动率机会）

简化实现：
- RV从GARCH或历史波动率计算
- IV从期权价格反推（需要期权数据源）
- 当无期权数据时，退化为波动率分析
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils import find_col

logger = logging.getLogger(__name__)


def norm_cdf(x: float) -> float:
    """标准正态分布CDF（用scipy或近似公式）"""
    try:
        from scipy.stats import norm
        return float(norm.cdf(x))
    except ImportError:
        # Abramowitz-Stegun近似
        a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2)
        return 0.5 * (1.0 + sign * y)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes看涨期权价格"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes看跌期权价格"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def implied_volatility(market_price: float, S: float, K: float, T: float,
                       r: float, is_call: bool = True,
                       max_iter: int = 100, tol: float = 1e-6) -> float | None:
    """
    Newton-Raphson法反推隐含波动率

    Args:
        market_price: 期权市场价格
        S: 标的价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        is_call: 是否看涨

    Returns:
        float: 隐含波动率，None if fails
    """
    sigma = 0.3  # 初始猜测

    for _ in range(max_iter):
        if is_call:
            price = bs_call_price(S, K, T, r, sigma)
        else:
            price = bs_put_price(S, K, T, r, sigma)

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        # Vega (价格对sigma的导数)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)

        if abs(vega) < 1e-10:
            break

        sigma -= diff / vega
        sigma = max(0.01, min(sigma, 5.0))  # 限制范围

    return sigma if abs(diff) < tol * 10 else None


class BlackScholesAnalyzer:
    """Black-Scholes波动率分析器"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.r = risk_free_rate

    def analyze(self, kline: pd.DataFrame,
                option_price: float | None = None,
                strike: float | None = None,
                expiry_days: int | None = None) -> dict[str, Any] | None:
        """
        分析波动率状态

        当无期权数据时，仅输出已实现波动率分析。
        当有期权数据时，计算IV并与RV比较。
        """
        if kline is None or kline.empty:
            return None

        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 20:
            return None

        # 已实现波动率（20日）
        log_returns = np.log(prices / prices.shift(1)).dropna()
        rv_20 = float(log_returns.tail(20).std() * np.sqrt(252) * 100)
        rv_60 = float(log_returns.tail(60).std() * np.sqrt(252) * 100) if len(log_returns) >= 60 else rv_20

        result = {
            "rv_20d": round(rv_20, 2),
            "rv_60d": round(rv_60, 2),
            "current_price": round(float(prices.iloc[-1]), 2),
            "model": "black_scholes",
        }

        # 如果有期权数据，计算IV
        if option_price and strike and expiry_days:
            S = float(prices.iloc[-1])
            T = expiry_days / 365.0
            iv = implied_volatility(option_price, S, strike, T, self.r)
            if iv:
                iv_pct = iv * 100
                iv_rv_spread = iv_pct - rv_20
                result["implied_vol"] = round(iv_pct, 2)
                result["iv_rv_spread"] = round(iv_rv_spread, 2)

                if iv_rv_spread > 5:
                    result["signal"] = "overpriced"
                    result["reason"] = f"IV({iv_pct:.1f}%) > RV({rv_20:.1f}%)，期权偏贵"
                elif iv_rv_spread < -5:
                    result["signal"] = "underpriced"
                    result["reason"] = f"IV({iv_pct:.1f}%) < RV({rv_20:.1f}%)，期权偏便宜"
                else:
                    result["signal"] = "fair"
                    result["reason"] = f"IV({iv_pct:.1f}%) ≈ RV({rv_20:.1f}%)，定价合理"
            else:
                result["implied_vol"] = None
                result["signal"] = "no_iv"
                result["reason"] = "无法计算隐含波动率"
        else:
            result["signal"] = "rv_only"
            result["reason"] = f"20日已实现波动率{rv_20:.1f}%，无期权数据比较"

        return result


def analyze_all(candidates: list, kline_map: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """对候选池做波动率分析"""
    analyzer = BlackScholesAnalyzer()
    results = {}
    for c in candidates:
        kline = kline_map.get(c.symbol)
        if kline is None:
            continue
        result = analyzer.analyze(kline)
        if result:
            results[c.symbol] = result
    return results
