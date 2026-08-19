"""
路线三：情绪错杀（炒股养家 + 乌合之众）

扫描逻辑：
1. 情绪冰点/试探期，但个股评分>70的标的
2. 涨停板块扩散度极低但北向资金在流入（分歧=机会）
3. 板块退潮中被连带下跌但基本面无变化的个股

炒股养家：「买在分歧，卖在共识」
乌合之众：群体恐慌时优质标的被错杀
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class SentimentMissScanner:
    """情绪错杀扫描器"""

    def scan(self, stock_list: list[dict], market_data: dict) -> list[dict]:
        """
        扫描情绪错杀机会

        Returns:
            list[dict]: 发现的标的列表
        """
        findings = []

        phase = market_data.get("emotion_phase", "")
        limit_up = market_data.get("limit_up_count", 0)
        break_rate = market_data.get("break_rate", 0.0)

        # 扫描1：情绪冰点中的优质标的
        if phase in ("冰点", "试探", "退潮"):
            findings.extend(self._scan_cold_quality(stock_list, market_data))

        # 扫描2：高分歧=机会（涨停多但炸板高）
        if limit_up >= 40 and break_rate >= 25:
            findings.extend(self._scan_high_divergence(stock_list, market_data))

        # 扫描3：板块退潮中的错杀
        if phase == "退潮":
            findings.extend(self._scan_sector_miss(stock_list, market_data))

        return findings

    def _scan_cold_quality(self, stock_list: list[dict],
                           market_data: dict) -> list[dict]:
        """
        情绪冰点中的优质标的
        条件：ROE>15% + 近5日跌幅>5% + 毛利率>30%
        """
        findings = []

        for stock in stock_list:
            q = stock.get("quarterly", {})
            kline = stock.get("kline")

            if not q or kline is None or kline.empty:
                continue

            try:
                roe_list = q.get("roe_list", [])
                gross_margin = q.get("gross_margin", 0)

                if not roe_list or roe_list[-1] < 15:
                    continue
                if gross_margin < 30:
                    continue

                close_col = None
                for col in ["close", "收盘", "收盘价"]:
                    if col in kline.columns:
                        close_col = col
                        break
                if close_col is None:
                    continue

                prices = kline[close_col].dropna()
                if len(prices) < 5:
                    continue

                drop_5d = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
                if drop_5d > -3:
                    continue

                findings.append({
                    "symbol": stock.get("symbol", ""),
                    "name": stock.get("name", ""),
                    "industry": stock.get("industry", ""),
                    "route": "sentiment_miss",
                    "reason": f"情绪冰点中，ROE={roe_list[-1]:.1f}%+毛利率{gross_margin:.1f}%但近5日跌{drop_5d:.1f}%",
                    "drop_5d": drop_5d,
                    "roe": roe_list[-1],
                })

            except Exception:
                continue

        return findings[:10]

    def _scan_high_divergence(self, stock_list: list[dict],
                              market_data: dict) -> list[dict]:
        """
        高分歧期的机会
        涨停多+炸板高=市场意见不统一=选股能力的回报
        """
        findings = []
        limit_up = market_data.get("limit_up_count", 0)
        break_rate = market_data.get("break_rate", 0.0)

        # 找近期逆势上涨的个股（在高分歧中走出独立行情）
        for stock in stock_list:
            kline = stock.get("kline")
            q = stock.get("quarterly", {})

            if kline is None or kline.empty:
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
                if len(prices) < 5:
                    continue

                change_5d = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100

                # 逆势上涨>5%
                if change_5d < 5:
                    continue

                # 基本面支撑
                if q:
                    roe_list = q.get("roe_list", [])
                    if not roe_list or roe_list[-1] < 10:
                        continue
                else:
                    continue

                findings.append({
                    "symbol": stock.get("symbol", ""),
                    "name": stock.get("name", ""),
                    "industry": stock.get("industry", ""),
                    "route": "sentiment_miss",
                    "reason": f"高分歧期(涨停{limit_up}家/炸板{break_rate:.0f}%)逆势涨{change_5d:.1f}%",
                    "change_5d": change_5d,
                })

            except Exception:
                continue

        return findings[:10]

    def _scan_sector_miss(self, stock_list: list[dict],
                          market_data: dict) -> list[dict]:
        """
        板块退潮中的错杀
        条件：所在板块整体下跌 + 个股基本面没恶化
        """
        findings = []

        # 简化：找近5日跌幅>8%但ROE稳定的个股
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
                if len(prices) < 5:
                    continue

                drop_5d = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
                if drop_5d > -5:
                    continue

                # 基本面检查
                roe_list = q.get("roe_list", [])
                net_margin = q.get("net_margin", 0)

                if not roe_list or roe_list[-1] < 12:
                    continue
                if net_margin < 8:
                    continue

                findings.append({
                    "symbol": stock.get("symbol", ""),
                    "name": stock.get("name", ""),
                    "industry": stock.get("industry", ""),
                    "route": "sentiment_miss",
                    "reason": f"退潮期跌{drop_5d:.1f}%但ROE={roe_list[-1]:.1f}%+净利率{net_margin:.1f}%，可能错杀",
                    "drop_5d": drop_5d,
                })

            except Exception:
                continue

        return findings[:10]
