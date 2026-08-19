"""
主动发现引擎 — 六条寻宝路线

六条路线：
1. 逆向猎手（Marks+Klarman）：大跌但基本面没恶化
2. 行业拐点（邱国鹭）：从分散→集中的行业
3. 情绪错杀（炒股养家+乌合之众）：群体恐慌中的优质标的
4. 催化剂猎手（Klarman）：有催化剂但股价未反应
5. 龙虎追踪（炒股养家）：龙头辨识+知名席位
6. 配对交易（协整套利）：同行业协整对的价差偏离

交叉验证：多条路线指向同一标的 → 高置信度
"""

from __future__ import annotations

import logging
from typing import Any

from src.data.quality import DataQualityAssessor
from src.rules.registry import RuleRegistry

from .catalyst import CatalystScanner
from .contrarian import ContrarianScanner
from .dragon_tiger import DragonTigerScanner
from .industry_shift import IndustryShiftScanner
from .pairs_scanner import PairsScanner
from .sentiment_miss import SentimentMissScanner

logger = logging.getLogger(__name__)


class DiscoveryEngine:
    """
    主动发现引擎 — 五路线交叉验证
    """

    def __init__(self, registry: RuleRegistry, quality: DataQualityAssessor):
        self.contrarian = ContrarianScanner(registry, quality)
        self.industry_shift = IndustryShiftScanner()
        self.sentiment_miss = SentimentMissScanner()
        self.catalyst = CatalystScanner()
        self.dragon_tiger = DragonTigerScanner()
        self.pairs_scanner = PairsScanner()

    def discover(self, stock_list: list[dict],
                 market_data: dict) -> list[dict]:
        """
        执行五条路线扫描 + 交叉验证

        Args:
            stock_list: 股票基础数据列表
            market_data: 市场数据

        Returns:
            list[dict]: 交叉验证后的发现列表，按置信度排序
        """
        all_findings = {
            "contrarian": [],
            "industry_shift": [],
            "sentiment_miss": [],
            "catalyst": [],
            "dragon_tiger": [],
            "pairs_trading": [],
        }

        # 执行各路线扫描
        try:
            all_findings["contrarian"] = self.contrarian.scan(stock_list, market_data)
        except Exception as e:
            logger.warning(f"逆向猎手扫描出错: {e}")

        try:
            all_findings["industry_shift"] = self.industry_shift.scan(market_data)
        except Exception as e:
            logger.warning(f"行业拐点扫描出错: {e}")

        try:
            all_findings["sentiment_miss"] = self.sentiment_miss.scan(stock_list, market_data)
        except Exception as e:
            logger.warning(f"情绪错杀扫描出错: {e}")

        try:
            all_findings["catalyst"] = self.catalyst.scan(stock_list, market_data)
        except Exception as e:
            logger.warning(f"催化剂扫描出错: {e}")

        try:
            all_findings["dragon_tiger"] = self.dragon_tiger.scan(stock_list, market_data)
        except Exception as e:
            logger.warning(f"龙虎追踪扫描出错: {e}")

        try:
            all_findings["pairs_trading"] = self.pairs_scanner.scan(stock_list, market_data)
        except Exception as e:
            logger.warning(f"配对交易扫描出错: {e}")

        # 交叉验证
        validated = self._cross_validate(all_findings)

        # 统计
        total = sum(len(v) for v in all_findings.values())
        logger.info(f"主动发现: 扫描{total}条线索，交叉验证后{len(validated)}条")

        return validated

    def _cross_validate(self, findings: dict[str, list[dict]]) -> list[dict]:
        """
        交叉验证：多条路线指向同一标的 → 高置信度

        同一只股票被越多路线发现，置信度越高。
        """
        # 统计每只股票被多少条路线发现
        symbol_routes: dict[str, list[dict]] = {}

        for route, items in findings.items():
            for item in items:
                symbol = item.get("symbol", "")
                if not symbol:
                    continue
                if symbol not in symbol_routes:
                    symbol_routes[symbol] = []
                symbol_routes[symbol].append(item)

        # 构建交叉验证结果
        validated = []
        base_keys = {"symbol", "name", "industry", "route", "reason", "routes", "route_count", "confidence", "reasons"}
        for symbol, items in symbol_routes.items():
            routes = set(item.get("route", "") for item in items)
            route_count = len(routes)

            # 合并原因
            reasons = []
            for item in items:
                r = item.get("reason", "")
                if r and r not in reasons:
                    reasons.append(r)

            # 置信度 = 路线数 / 总路线数
            confidence = route_count / 6.0

            # 保留原始items中的结构化数据（如配对交易的hedge_ratio等）
            extra = {}
            for item in items:
                for k, v in item.items():
                    if k not in base_keys and k not in extra:
                        extra[k] = v

            validated.append({
                "symbol": symbol,
                "name": items[0].get("name", ""),
                "industry": items[0].get("industry", ""),
                "routes": list(routes),
                "route_count": route_count,
                "confidence": confidence,
                "reasons": reasons[:3],  # 最多3条原因
                "reason": f"被{route_count}条路线发现: {reasons[0]}" if reasons else "",
                **extra,
            })

        # 按路线数（置信度）排序
        validated.sort(key=lambda x: (x["route_count"], x["confidence"]), reverse=True)

        return validated
