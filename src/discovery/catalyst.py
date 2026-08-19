"""
路线四：催化剂猎手（Klarman）

扫描逻辑：
1. 即将发布财报且上季度超预期的标的
2. 近期有重大合同/订单公告但股价未涨
3. 大股东/高管近期增持
4. 行业并购重组，同行业标的

Klarman：便宜但无催化剂可能是价值陷阱，有催化剂才是真机会
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class CatalystScanner:
    """催化剂猎手扫描器"""

    def scan(self, stock_list: list[dict], market_data: dict) -> list[dict]:
        """
        扫描催化剂机会

        Returns:
            list[dict]: 发现的标的列表
        """
        findings = []

        for stock in stock_list:
            f10 = stock.get("f10", {})
            q = stock.get("quarterly", {})
            kline = stock.get("kline")

            if not f10:
                continue

            stock_findings = []

            # 检查公告中的催化剂
            announcements = f10.get("announcements", [])
            catalyst_keywords = ["增持", "回购", "纳入", "超预期", "业绩预增",
                                 "重大合同", "战略合作", "中标", "并购"]
            for ann in announcements:
                title = str(ann.get("title", ""))
                for kw in catalyst_keywords:
                    if kw in title:
                        stock_findings.append(f"{kw}: {title[:40]}")
                        break

            # 检查业绩改善
            if q:
                roe_list = q.get("roe_list", [])
                if len(roe_list) >= 2 and roe_list[-1] > roe_list[-2] * 1.2:
                    stock_findings.append(f"ROE改善: {roe_list[-2]:.1f}%→{roe_list[-1]:.1f}%")

                revenue_growth = q.get("revenue_growth", 0)
                if revenue_growth > 20:
                    stock_findings.append(f"营收增长{revenue_growth:.0f}%")

            # 检查股价未涨（催化剂未被定价）
            if kline is not None and not kline.empty and stock_findings:
                try:
                    close_col = None
                    for col in ["close", "收盘", "收盘价"]:
                        if col in kline.columns:
                            close_col = col
                            break
                    if close_col:
                        prices = kline[close_col].dropna()
                        if len(prices) >= 5:
                            change_5d = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
                            if change_5d > 5:
                                continue  # 已经涨了，催化剂可能已定价
                except Exception:
                    pass

            if stock_findings:
                findings.append({
                    "symbol": stock.get("symbol", ""),
                    "name": stock.get("name", ""),
                    "industry": stock.get("industry", ""),
                    "route": "catalyst",
                    "reason": f"发现{len(stock_findings)}个催化剂：{stock_findings[0]}",
                    "catalysts": stock_findings,
                })

        return findings[:15]
