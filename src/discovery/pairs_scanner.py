"""
配对交易扫描器 — 协整检验

扫描逻辑：
1. 按行业分组候选股票
2. 对同行业股票两两做Engle-Granger协整检验
3. 协整对的价差偏离>2sigma时发出信号

Engle-Granger两步法：
Step 1: OLS回归 Y = alpha + beta * X + epsilon
Step 2: 对残差epsilon做ADF单位根检验
若残差平稳 → 协整关系成立

半衰期：Ornstein-Uhlenbeck均值回归速度
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PairsScanner:
    """配对交易扫描器"""

    def __init__(self, p_threshold: float = 0.05, spread_threshold: float = 2.0,
                 min_overlap: int = 30, half_life_max: float = 30.0):
        self.p_threshold = p_threshold
        self.spread_threshold = spread_threshold
        self.min_overlap = min_overlap
        self.half_life_max = half_life_max

    def scan(self, stock_list: list[dict], market_data: dict) -> list[dict]:
        """
        扫描协整配对

        Args:
            stock_list: 股票数据列表（需含kline, industry, symbol, name）
            market_data: 市场数据

        Returns:
            list[dict]: 配对信号列表
        """
        # 检查statsmodels是否可用
        try:
            from statsmodels.tsa.stattools import adfuller  # noqa: F401
        except ImportError:
            logger.warning("statsmodels不可用，配对交易扫描跳过")
            return []

        # 按行业分组
        industry_groups: dict[str, list[dict]] = {}
        for stock in stock_list:
            industry = stock.get("industry", "")
            kline = stock.get("kline")
            if not industry or kline is None or (hasattr(kline, 'empty') and kline.empty):
                continue
            industry_groups.setdefault(industry, []).append(stock)

        signals = []
        for industry, stocks in industry_groups.items():
            if len(stocks) < 2:
                continue
            for a, b in combinations(stocks, 2):
                try:
                    result = self._test_pair(a, b, industry)
                    if result:
                        signals.append(result)
                except Exception as e:
                    logger.debug(f"配对检验 {a.get('symbol')}/{b.get('symbol')} 失败: {e}")

        # 按|z-score|排序
        signals.sort(key=lambda x: abs(x.get("spread_zscore", 0)), reverse=True)
        return signals

    def _test_pair(self, stock_a: dict, stock_b: dict, industry: str) -> dict | None:
        """检验一对股票的协整关系"""
        from statsmodels.tsa.stattools import adfuller

        kline_a = stock_a.get("kline")
        kline_b = stock_b.get("kline")

        close_a = self._get_close(kline_a)
        close_b = self._get_close(kline_b)
        if close_a is None or close_b is None:
            return None

        # 按日期对齐（优先用DatetimeIndex，否则回退到位置对齐）
        y, x = self._align_series(close_a, close_b)
        if len(y) < self.min_overlap:
            return None

        # Step 1: OLS回归 Y = alpha + beta * X
        X_matrix = np.column_stack([np.ones(len(x)), x])
        try:
            beta_hat, *_ = np.linalg.lstsq(X_matrix, y, rcond=None)
        except np.linalg.LinAlgError:
            return None

        alpha, hedge_ratio = beta_hat[0], beta_hat[1]
        residuals = y - (alpha + hedge_ratio * x)

        # Step 2: ADF检验残差平稳性
        try:
            adf_result = adfuller(residuals, maxlag=5, autolag='AIC')
            adf_pvalue = adf_result[1]
        except Exception:
            return None

        if adf_pvalue >= self.p_threshold:
            return None  # 不协整

        # 价差z-score
        spread_mean = np.mean(residuals)
        spread_std = np.std(residuals, ddof=1)
        if spread_std < 1e-10:
            return None
        z_score = (residuals[-1] - spread_mean) / spread_std

        if abs(z_score) < self.spread_threshold:
            return None  # 价差未偏离

        # 半衰期
        half_life = self._calc_half_life(residuals)
        if half_life > self.half_life_max or half_life <= 0:
            return None  # 回归太慢

        symbol_a = stock_a.get("symbol", "")
        symbol_b = stock_b.get("symbol", "")
        name_a = stock_a.get("name", symbol_a)
        name_b = stock_b.get("name", symbol_b)

        # 判断哪只被低估
        if z_score > 0:
            # Y偏高 → X被低估
            signal_desc = f"{name_b}可能被低估"
            undervalued = symbol_b
        else:
            # Y偏低 → X被高估，Y被低估
            signal_desc = f"{name_a}可能被低估"
            undervalued = symbol_a

        return {
            "symbol": f"{symbol_a}/{symbol_b}",
            "name": f"{name_a} vs {name_b}",
            "industry": industry,
            "route": "pairs_trading",
            "reason": f"协整配对 ADF p={adf_pvalue:.4f}, 价差z={z_score:.2f}偏离>{self.spread_threshold}σ, {signal_desc}",
            "pair_a": symbol_a,
            "pair_b": symbol_b,
            "pair_a_name": name_a,
            "pair_b_name": name_b,
            "hedge_ratio": round(float(hedge_ratio), 4),
            "adf_pvalue": round(float(adf_pvalue), 6),
            "spread_zscore": round(float(z_score), 3),
            "half_life": round(float(half_life), 1),
            "undervalued": undervalued,
            "signal": "pair_divergence",
        }

    def _calc_half_life(self, residuals: np.ndarray) -> float:
        """
        计算Ornstein-Uhlenbeck半衰期

        回归 delta_R = theta * (mu - R_{t-1})
        half_life = ln(2) / theta
        """
        r = residuals[:-1]
        delta_r = np.diff(residuals)
        mu = np.mean(residuals)

        # OLS: delta_r = theta * (mu - r)
        X = (mu - r).reshape(-1, 1)
        try:
            theta, *_ = np.linalg.lstsq(X, delta_r, rcond=None)
            theta = float(theta[0])
        except np.linalg.LinAlgError:
            return float('inf')

        if theta <= 0:
            return float('inf')  # 不均值回归

        return np.log(2) / theta

    def _get_close(self, kline) -> pd.Series | None:
        """从kline提取收盘价序列"""
        if kline is None:
            return None
        if isinstance(kline, pd.DataFrame):
            for col in ("close", "收盘", "收盘价"):
                if col in kline.columns:
                    return kline[col].dropna()
        return None

    def _align_series(self, s1: pd.Series, s2: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """按日期对齐两个序列，返回对齐后的numpy数组"""
        # 尝试DatetimeIndex对齐
        if isinstance(s1.index, pd.DatetimeIndex) and isinstance(s2.index, pd.DatetimeIndex):
            common = s1.index.intersection(s2.index)
            if len(common) >= self.min_overlap:
                return s1.loc[common].values, s2.loc[common].values

        # 尝试用date列对齐（kline可能有date列作为普通列）
        # 回退：位置对齐（取较短长度的尾部）
        min_len = min(len(s1), len(s2))
        logger.debug("日期索引不可用，回退到位置对齐")
        return s1.values[-min_len:], s2.values[-min_len:]
