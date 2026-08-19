"""
Fama-French 5因子模型 — Alpha归因

回归方程：
R_i - R_f = alpha + beta*(R_m - R_f) + s*SMB + h*HML + r*RMW + c*CMA + epsilon

因子：
- R_m - R_f: 市场超额收益
- SMB (Small Minus Big): 规模因子，小盘-大盘
- HML (High Minus Low): 价值因子，高PB-低PB
- RMW (Robust Minus Weak): 盈利因子，高ROE-低ROE
- CMA (Conservative Minus Aggressive): 投资因子，低投资-高投资

输出：
- Alpha（年化超额收益）
- 各因子暴露（Beta, SMB, HML, RMW, CMA）
- R²（因子能解释多少收益）

简化实现：
- 用市场数据近似构建因子收益率序列
- 当因子数据不足时，退化为CAPM单因子模型
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils import find_col

logger = logging.getLogger(__name__)


class FamaFrenchModel:
    """Fama-French 5因子模型"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.rf_daily = risk_free_rate / 252

    def analyze(self, stock_kline: pd.DataFrame,
                market_kline: pd.DataFrame | None = None,
                factor_returns: pd.DataFrame | None = None) -> dict[str, Any] | None:
        """
        对单只股票做因子归因分析

        Args:
            stock_kline: 个股K线
            market_kline: 市场指数K线（如沪深300）
            factor_returns: 因子收益率DataFrame（可选，含mkt_rf, smb, hml, rmw, cma列）

        Returns:
            dict with alpha, beta, smb, hml, rmw, cma, r_squared
        """
        if stock_kline is None or stock_kline.empty:
            return None

        close_col = find_col(stock_kline, ["close", "收盘", "收盘价"])
        if close_col is None:
            return None

        stock_prices = stock_kline[close_col].dropna()
        if len(stock_prices) < 30:
            return None

        stock_returns = stock_prices.pct_change().dropna()

        # 如果有市场K线，计算市场超额收益
        if market_kline is not None and not market_kline.empty:
            mkt_close = find_col(market_kline, ["close", "收盘", "收盘价"])
            if mkt_close:
                mkt_prices = market_kline[mkt_close].dropna()
                mkt_returns = mkt_prices.pct_change().dropna()
                # 对齐
                common_len = min(len(stock_returns), len(mkt_returns))
                stock_returns = stock_returns.iloc[-common_len:]
                mkt_excess = mkt_returns.iloc[-common_len:] - self.rf_daily
            else:
                mkt_excess = None
        else:
            mkt_excess = None

        try:
            if factor_returns is not None and not factor_returns.empty:
                return self._regress_5factor(stock_returns, factor_returns)
            elif mkt_excess is not None:
                return self._regress_capm(stock_returns, mkt_excess)
            else:
                return self._estimate_from_kline(stock_returns)
        except Exception as e:
            logger.debug(f"因子回归失败: {e}")
            return None

    def _regress_5factor(self, stock_returns: pd.Series,
                         factor_returns: pd.DataFrame) -> dict[str, Any]:
        """5因子回归"""
        # 对齐长度
        common_len = min(len(stock_returns), len(factor_returns))
        y = stock_returns.iloc[-common_len:].values
        X_df = factor_returns.iloc[-common_len:].copy()

        # 构建设计矩阵
        factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma"]
        available_cols = [c for c in factor_cols if c in X_df.columns]
        if not available_cols:
            return self._estimate_from_kline(stock_returns)

        X = X_df[available_cols].values
        X = np.column_stack([np.ones(len(X)), X])  # 截距项

        # OLS回归
        try:
            beta_hat, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return self._estimate_from_kline(stock_returns)

        alpha_daily = float(beta_hat[0])
        factor_betas = {}
        for i, col in enumerate(available_cols):
            factor_betas[col] = float(beta_hat[i + 1])

        # R²
        y_pred = X @ beta_hat
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 年化Alpha
        alpha_annual = alpha_daily * 252

        return {
            "alpha_annual": round(alpha_annual * 100, 2),
            "alpha_daily": round(alpha_daily * 100, 4),
            "beta": round(factor_betas.get("mkt_rf", 0), 3),
            "smb": round(factor_betas.get("smb", 0), 3),
            "hml": round(factor_betas.get("hml", 0), 3),
            "rmw": round(factor_betas.get("rmw", 0), 3),
            "cma": round(factor_betas.get("cma", 0), 3),
            "r_squared": round(float(r_squared), 3),
            "model": "fama_french_5factor",
        }

    def _regress_capm(self, stock_returns: pd.Series,
                      mkt_excess: pd.Series) -> dict[str, Any]:
        """CAPM单因子回归（fallback）"""
        common_len = min(len(stock_returns), len(mkt_excess))
        y = stock_returns.iloc[-common_len:].values - self.rf_daily
        x = mkt_excess.iloc[-common_len:].values

        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return None

        alpha_daily = float(beta_hat[0])
        beta = float(beta_hat[1])

        y_pred = X @ beta_hat
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "alpha_annual": round(alpha_daily * 252 * 100, 2),
            "alpha_daily": round(alpha_daily * 100, 4),
            "beta": round(beta, 3),
            "smb": 0, "hml": 0, "rmw": 0, "cma": 0,
            "r_squared": round(float(r_squared), 3),
            "model": "capm_single_factor",
        }

    def _estimate_from_kline(self, stock_returns: pd.Series) -> dict[str, Any]:
        """无市场数据时的估算（仅返回收益统计）"""
        mu = float(stock_returns.mean()) * 252
        sigma = float(stock_returns.std()) * np.sqrt(252)
        sharpe = (mu - 0.02) / sigma if sigma > 0 else 0

        return {
            "alpha_annual": round((mu - 0.02) * 100, 2),
            "alpha_daily": round(float(stock_returns.mean()) * 100, 4),
            "beta": 1.0,  # 默认
            "smb": 0, "hml": 0, "rmw": 0, "cma": 0,
            "r_squared": 0.0,
            "model": "no_market_data",
            "note": "无市场数据，Alpha为超额收益估算",
        }


def analyze_all(candidates: list, kline_map: dict[str, pd.DataFrame],
                market_kline: pd.DataFrame | None = None,
                factor_returns: pd.DataFrame | None = None) -> dict[str, dict]:
    """
    对候选池所有标的做因子归因

    Args:
        candidates: CandidateStock列表
        kline_map: {symbol: kline_DataFrame}
        market_kline: 市场指数K线
        factor_returns: 因子收益率数据

    Returns:
        dict: {symbol: factor_result}
    """
    model = FamaFrenchModel()
    results = {}

    for c in candidates:
        kline = kline_map.get(c.symbol)
        if kline is None:
            continue
        result = model.analyze(kline, market_kline, factor_returns)
        if result:
            results[c.symbol] = result

    return results
