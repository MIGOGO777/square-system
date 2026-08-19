"""
Almgren-Chriss最优执行模型 — 大单拆单算法

核心思想：交易越快，市场冲击越大；交易越慢，价格漂移风险越大。
最优执行路径在两者之间取得平衡。

模型参数：
- sigma: 日波动率
- eta: 临时冲击系数（每单位成交量对价格的瞬时影响）
- gamma: 永久冲击系数（交易对价格的持久影响）
- X: 总交易量
- T: 执行时间窗口（天）
- N: 拆单笔数

简化实现：
- 用历史成交量和波动率估计冲击参数
- 输出建议拆单方案
- 适用于机构大单，散户价值有限
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils import find_col

logger = logging.getLogger(__name__)


class AlmgrenChriss:
    """Almgren-Chriss最优执行模型"""

    def __init__(self, risk_aversion: float = 1e-6):
        """
        Args:
            risk_aversion: 风险厌恶系数（越大越保守，执行越快）
        """
        self.risk_aversion = risk_aversion

    def optimize(self, kline: pd.DataFrame, trade_volume: float,
                 adv: float | None = None, T: int = 2) -> dict[str, Any] | None:
        """
        计算最优执行方案

        Args:
            kline: K线数据
            trade_volume: 交易量（股数或金额）
            adv: 日均成交量（可选，从kline估算）
            T: 执行天数

        Returns:
            dict with execution plan
        """
        if kline is None or kline.empty:
            return None

        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        vol_col = find_col(kline, ["volume", "成交量"])
        if close_col is None:
            return None

        prices = kline[close_col].dropna()
        if len(prices) < 20:
            return None

        # 估算参数
        sigma = float(np.log(prices / prices.shift(1)).dropna().tail(20).std())  # 日波动率

        if vol_col and vol_col in kline.columns:
            volumes = kline[vol_col].dropna()
            adv_est = float(volumes.tail(20).mean()) if len(volumes) >= 20 else float(volumes.mean())
        else:
            adv_est = adv if adv else 1e6  # 默认

        if adv_est <= 0:
            return None

        # 交易量占ADV的比例
        participation = trade_volume / (adv_est * T)

        if participation > 0.5:
            # 交易量太大，建议延长执行时间
            T_suggested = int(np.ceil(trade_volume / (adv_est * 0.2)))
            T_suggested = min(T_suggested, 10)
        else:
            T_suggested = T

        # 冲击参数（简化估计）
        # eta: 临时冲击系数 ≈ sigma / (ADV * sqrt(N))
        # gamma: 永久冲击系数 ≈ sigma * participation
        eta = sigma / (np.sqrt(adv_est) * 0.1)
        gamma = sigma * participation * 0.1

        # 最优执行轨迹（Almgren-Chriss closed-form）
        N = T_suggested * 4  # 每天4个时段（开盘/上午/下午/收盘）
        tau = T_suggested / N

        # 计算最优持有量轨迹
        kappa = np.sqrt(self.risk_aversion * sigma**2 / eta)
        if kappa * T_suggested > 50:
            kappa = 50 / T_suggested  # 防止溢出

        holdings = np.zeros(N + 1)
        holdings[0] = trade_volume

        if kappa > 1e-10:
            for j in range(N):
                t_remaining = (N - j) * tau
                holdings[j + 1] = trade_volume * np.sinh(kappa * t_remaining) / np.sinh(kappa * T_suggested)
        else:
            # 线性执行
            for j in range(N):
                holdings[j + 1] = trade_volume * (N - j - 1) / N

        # 每笔交易量
        trades = np.diff(-holdings)
        trades = np.maximum(trades, 0)  # 确保非负

        # 估算成本
        temp_impact = eta * np.sum(trades**2) / tau
        perm_impact = gamma * trade_volume
        total_impact = temp_impact + perm_impact
        impact_pct = total_impact / (trade_volume * float(prices.iloc[-1])) * 100

        # 建议分几笔
        n_trades = max(2, min(N, int(np.ceil(trade_volume / (adv_est * 0.15)))))

        return {
            "suggested_days": T_suggested,
            "n_trades": n_trades,
            "trade_per_period": round(trade_volume / n_trades, 0),
            "expected_impact_pct": round(impact_pct, 3),
            "participation_rate": round(participation * 100, 2),
            "sigma_daily": round(sigma * 100, 3),
            "adv": round(adv_est, 0),
            "execution_schedule": self._format_schedule(trades, n_trades),
            "model": "almgren_chriss",
        }

    def _format_schedule(self, trades: np.ndarray, n_trades: int) -> list[str]:
        """格式化执行计划"""
        total = np.sum(trades)
        if total <= 0:
            return []

        # 分成n_trades笔
        per_trade = total / n_trades
        schedule = []
        labels = ["开盘", "上午盘中", "下午盘中", "收盘"]
        for i in range(n_trades):
            label = labels[i % len(labels)]
            day = i // len(labels) + 1
            schedule.append(f"第{day}天{label}: {per_trade:.0f}股")

        return schedule[:6]  # 最多显示6笔


def analyze_candidates(candidates: list, kline_map: dict[str, pd.DataFrame],
                       position_value: float = 100000) -> dict[str, dict]:
    """
    对候选池做最优执行分析

    Args:
        candidates: CandidateStock列表
        kline_map: {symbol: kline_DataFrame}
        position_value: 假设的持仓金额（元）

    Returns:
        dict: {symbol: execution_plan}
    """
    ac = AlmgrenChriss()
    results = {}

    for c in candidates:
        kline = kline_map.get(c.symbol)
        if kline is None:
            continue

        # 用position_value估算交易量
        close_col = find_col(kline, ["close", "收盘", "收盘价"])
        if close_col is None:
            continue
        prices = kline[close_col].dropna()
        if prices.empty:
            continue

        current_price = float(prices.iloc[-1])
        if current_price <= 0:
            continue

        trade_volume = position_value / current_price  # 股数

        result = ac.optimize(kline, trade_volume)
        if result:
            results[c.symbol] = result

    return results
