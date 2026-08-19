"""
Markowitz组合优化器

输入：候选池的kline收益率序列
输出：每个候选标的的推荐仓位权重

步骤：
1. 从候选标的的60日K线计算日收益率
2. 构建协方差矩阵
3. 求解最优权重（最大夏普比率）
4. 输出推荐仓位

失败时fallback到等权重分配。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils import find_col

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Markowitz组合优化器"""

    def __init__(self, risk_free_rate: float = 0.02, max_weight: float = 0.30):
        self.risk_free_rate = risk_free_rate / 252  # 日化
        self.max_weight = max_weight

    def optimize(self, candidates: list,
                 kline_map: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """
        对候选池进行Markowitz优化

        Args:
            candidates: CandidateStock列表
            kline_map: {symbol: kline_DataFrame}

        Returns:
            dict: {symbol: {"weight": float, "expected_return": float,
                           "volatility": float, "sharpe": float}}
        """
        if not candidates or not kline_map:
            return {}

        # 提取各股日收益率
        returns_map = {}
        for c in candidates:
            kline = kline_map.get(c.symbol)
            if kline is None or kline.empty:
                continue
            close_col = find_col(kline, ["close", "收盘", "收盘价"])
            if close_col is None:
                continue
            prices = kline[close_col].dropna()
            if len(prices) < 20:
                continue
            daily_returns = prices.pct_change().dropna()
            if len(daily_returns) < 10:
                continue
            returns_map[c.symbol] = daily_returns

        if len(returns_map) < 2:
            # 不足2只，无法优化，返回等权重
            return self._equal_weight_fallback(list(returns_map.keys()))

        try:
            return self._optimize_sharpe(returns_map)
        except Exception as e:
            logger.warning(f"Markowitz优化失败: {e}，使用等权重")
            return self._equal_weight_fallback(list(returns_map.keys()))

    def _optimize_sharpe(self, returns_map: dict[str, pd.Series]) -> dict[str, dict]:
        """最大夏普比率优化"""
        from scipy.optimize import minimize

        symbols = list(returns_map.keys())
        n = len(symbols)

        # 构建收益率矩阵（对齐日期）
        df_returns = pd.DataFrame(returns_map)
        df_returns = df_returns.dropna()
        if len(df_returns) < 10:
            return self._equal_weight_fallback(symbols)

        mean_returns = df_returns.mean().values
        cov_matrix = df_returns.cov().values

        # 目标函数：负夏普比率
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol < 1e-10:
                return 1e6  # 惩罚：零波动率意味着退化组合
            return -(port_return - self.risk_free_rate) / port_vol

        # 约束：权重之和=1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((0, self.max_weight) for _ in range(n))
        x0 = np.ones(n) / n  # 初始等权重

        result = minimize(neg_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-10})

        if not result.success:
            logger.warning(f"优化未收敛: {result.message}")
            return self._equal_weight_fallback(symbols)

        weights = result.x
        port_return = float(np.dot(weights, mean_returns) * 252)
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252))
        sharpe = float((port_return - self.risk_free_rate * 252) / port_vol) if port_vol > 0 else 0

        output = {}
        for i, sym in enumerate(symbols):
            output[sym] = {
                "weight": round(float(weights[i]), 4),
                "expected_return": round(float(mean_returns[i] * 252), 4),
                "volatility": round(float(np.sqrt(cov_matrix[i, i]) * np.sqrt(252)), 4),
                "sharpe": round(sharpe, 3),
            }

        logger.info(f"Markowitz优化完成: {n}只标的，组合夏普={sharpe:.3f}")
        return output

    def _equal_weight_fallback(self, symbols: list[str]) -> dict[str, dict]:
        """等权重fallback"""
        if not symbols:
            return {}
        w = 1.0 / len(symbols)
        return {sym: {"weight": round(w, 4), "expected_return": 0,
                       "volatility": 0, "sharpe": 0} for sym in symbols}
