"""
路线一：逆向猎手（Marks + Klarman）

扫描逻辑：
1. 近20日跌幅最大的行业中，找出基本面没恶化的个股
2. 北向资金连续流出但公司ROE/现金流改善的标的
3. 解禁潮刚过但股价企稳的标的

逆向核心：「别人恐惧时贪婪」— 但必须确认基本面没恶化
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.core.signal import CandidateStock
from src.data.quality import DataQualityAssessor
from src.rules.registry import RuleRegistry
from src.pool.builder import quick_score_stock

logger = logging.getLogger(__name__)


class ContrarianScanner:
    """逆向猎手扫描器"""

    def __init__(self, registry: RuleRegistry, quality: DataQualityAssessor):
        self.registry = registry
        self.quality = quality

    def scan(self, stock_list: list[dict], market_data: dict) -> list[dict]:
        """
        扫描逆向机会

        Returns:
            list[dict]: 发现的标的列表，每项包含symbol, name, reason, score等
        """
        findings = []

        # 扫描1：近期大跌但基本面没恶化
        findings.extend(self._scan_fallen_angels(stock_list, market_data))

        # 扫描2：北向流出但公司改善
        findings.extend(self._scan_north_divergence(stock_list, market_data))

        return findings

    def _scan_fallen_angels(self, stock_list: list[dict],
                            market_data: dict) -> list[dict]:
        """
        近期大跌但基本面没恶化的个股
        条件：近20日跌幅>10% + ROE稳定 + FCF>0
        """
        findings = []

        for stock in stock_list:
            kline = stock.get("kline")
            q = stock.get("quarterly", {})

            if kline is None or kline.empty or not q:
                continue

            try:
                close_col = None
                for col in ["close", "收盘", "收盘价"]:
                    if col in kline.columns:
                        close_col = col
                        break
                if close_col is None:
                    continue

                prices = kline[close_col].dropna()
                if len(prices) < 20:
                    continue

                # 近20日跌幅
                drop_pct = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100

                if drop_pct > -8:
                    continue  # 跌幅不够

                # 基本面检查
                roe_list = q.get("roe_list", [])
                fcf_list = q.get("fcf_list", [])
                net_margin = q.get("net_margin", 0)

                # ROE稳定（最近一年>10%）
                if not roe_list or roe_list[-1] < 10:
                    continue

                # FCF为正
                if fcf_list and fcf_list[-1] < 0:
                    continue

                # 净利率为正
                if net_margin < 5:
                    continue

                findings.append({
                    "symbol": stock.get("symbol", ""),
                    "name": stock.get("name", ""),
                    "industry": stock.get("industry", ""),
                    "route": "contrarian",
                    "reason": f"近20日跌{drop_pct:.1f}%但ROE={roe_list[-1]:.1f}%+净利率{net_margin:.1f}%，基本面未恶化",
                    "drop_pct": drop_pct,
                    "roe": roe_list[-1],
                })

            except Exception:
                continue

        return findings

    def _scan_north_divergence(self, stock_list: list[dict],
                               market_data: dict) -> list[dict]:
        """
        北向资金连续流出但公司ROE/现金流改善的标的
        """
        north_flow = market_data.get("north_flow")
        if north_flow is None or north_flow.empty:
            return []

        # 检查北向是否连续流出
        try:
            flow_col = None
            for col in ("net_flow", "净流入", "north_net"):
                if col in north_flow.columns:
                    flow_col = col
                    break
            if flow_col is None:
                return []

            recent = north_flow[flow_col].dropna().tail(5)
            if len(recent) < 3:
                return []

            consecutive_out = sum(1 for v in recent if v < 0)
            if consecutive_out < 3:
                return []  # 北向没有连续流出

        except Exception:
            return []

        # 在北向流出期间，找ROE改善的个股
        findings = []
        for stock in stock_list:
            q = stock.get("quarterly", {})
            if not q:
                continue

            roe_list = q.get("roe_list", [])
            if len(roe_list) < 2:
                continue

            # ROE改善
            if roe_list[-1] > roe_list[-2] and roe_list[-1] > 12:
                findings.append({
                    "symbol": stock.get("symbol", ""),
                    "name": stock.get("name", ""),
                    "industry": stock.get("industry", ""),
                    "route": "contrarian",
                    "reason": f"北向连续流出期间，ROE从{roe_list[-2]:.1f}%升至{roe_list[-1]:.1f}%",
                    "roe_improvement": roe_list[-1] - roe_list[-2],
                })

        return findings[:10]  # 最多10只
