"""
Monte Carlo价格模拟 — GBM路径模拟

基于几何布朗运动(GBM)模拟价格路径：
S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
Z ~ N(0,1)

输出：
- VaR 95%: 5日后最大亏损
- 盈利概率: 5日后价格>当前价格的概率
- 最大回撤P95: 95%分位的最大回撤
- 预期收益: 5日平均收益
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils import find_col

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo价格路径模拟器"""

    def __init__(self, n_simulations: int = 10000, horizon: int = 5,
                 confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.confidence_level = confidence_level

    def simulate(self, kline: pd.DataFrame) -> dict[str, Any] | None:
        """
        对单只股票做Monte Carlo模拟

        Args:
            kline: K线DataFrame

        Returns:
            dict with var_95, profit_prob, max_drawdown_p95, expected_return
            None if data insufficient
        """
        if kline is None or kline.empty:
            return None

        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 30:
            return None

        try:
            return self._run_simulation(prices)
        except Exception as e:
            logger.debug(f"Monte Carlo模拟失败: {e}")
            return None

    def _run_simulation(self, prices: pd.Series) -> dict[str, Any]:
        """执行GBM Monte Carlo模拟"""
        # 计算对数收益率的mu和sigma
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu = float(log_returns.mean())  # 日均值
        sigma = float(log_returns.std())  # 日标准差

        if sigma < 1e-10:
            return None

        S0 = float(prices.iloc[-1])
        dt = 1  # 1天

        # 生成随机路径: (n_simulations, horizon)
        np.random.seed(42)
        Z = np.random.standard_normal((self.n_simulations, self.horizon))

        # GBM: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        increments = drift + diffusion

        # 累积收益
        cumulative_returns = np.cumsum(increments, axis=1)
        # 最终价格
        final_prices = S0 * np.exp(cumulative_returns[:, -1])
        # 所有路径上的价格矩阵
        price_paths = S0 * np.exp(cumulative_returns)

        # 统计指标
        final_returns = (final_prices - S0) / S0  # 5日收益率
        expected_return = float(np.mean(final_returns))

        # VaR: 最大亏损的分位数
        var_alpha = 1 - self.confidence_level
        var_95 = float(np.percentile(final_returns, var_alpha * 100))

        # 盈利概率
        profit_prob = float(np.mean(final_prices > S0))

        # 最大回撤（每条路径）
        max_drawdowns = []
        for i in range(self.n_simulations):
            path = price_paths[i]
            running_max = np.maximum.accumulate(path)
            drawdowns = (path - running_max) / running_max
            max_drawdowns.append(float(np.min(drawdowns)))
        max_drawdown_p95 = float(np.percentile(max_drawdowns, 5))  # 5%分位=最差的5%

        return {
            "expected_return": round(expected_return * 100, 2),  # 百分比
            "var_95": round(var_95 * 100, 2),  # 百分比
            "profit_prob": round(profit_prob, 3),
            "max_drawdown_p95": round(max_drawdown_p95 * 100, 2),  # 百分比
            "mu_annualized": round(mu * 252 * 100, 2),
            "sigma_annualized": round(sigma * np.sqrt(252) * 100, 2),
            "n_simulations": self.n_simulations,
            "horizon": self.horizon,
        }


def simulate_all(candidates: list, kline_map: dict[str, pd.DataFrame],
                 n_simulations: int = 10000, horizon: int = 5) -> dict[str, dict]:
    """
    对候选池所有标的做Monte Carlo模拟

    Args:
        candidates: CandidateStock列表
        kline_map: {symbol: kline_DataFrame}
        n_simulations: 模拟次数
        horizon: 模拟天数

    Returns:
        dict: {symbol: simulation_result}
    """
    sim = MonteCarloSimulator(n_simulations=n_simulations, horizon=horizon)
    results = {}

    for c in candidates:
        kline = kline_map.get(c.symbol)
        if kline is None:
            continue
        result = sim.simulate(kline)
        if result:
            results[c.symbol] = result

    return results
