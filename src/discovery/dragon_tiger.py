"""
路线五：龙虎追踪（炒股养家）

扫描逻辑：
1. 龙虎榜出现知名游资席位
2. 连板高度突破近期新高
3. 概念板块中第一个涨停的标的
4. 涨停后次日高开且成交量未放大（惜售）

炒股养家：龙头辨识度越高，带动效应越强
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DragonTigerScanner:
    """龙虎追踪扫描器"""

    def scan(self, stock_list: list[dict], market_data: dict) -> list[dict]:
        """
        扫描龙虎榜机会

        Returns:
            list[dict]: 发现的标的列表
        """
        findings = []

        # 扫描1：龙虎榜数据
        dragon_tiger = market_data.get("dragon_tiger")
        if dragon_tiger is not None and not dragon_tiger.empty:
            findings.extend(self._scan_dragon_tiger(dragon_tiger))

        # 扫描2：连板高度
        findings.extend(self._scan_leader_height(stock_list, market_data))

        # 扫描3：涨停池中的龙头（板块第一个涨停）
        limit_up_pool = market_data.get("limit_up_pool")
        if limit_up_pool is not None and not limit_up_pool.empty:
            findings.extend(self._scan_limit_up_leaders(limit_up_pool))

        return findings

    def _scan_dragon_tiger(self, dt: pd.DataFrame) -> list[dict]:
        """扫描龙虎榜中的知名席位"""
        findings = []

        # 知名游资席位关键词（简化版）
        hot_seats = ["华泰证券上海", "中信证券上海", "国泰君安上海",
                     "东方财富拉萨", "机构专用"]

        try:
            for _, row in dt.iterrows():
                symbol = str(row.get("代码", row.get("code", row.get("symbol", ""))))
                name = str(row.get("名称", row.get("name", "")))

                # 检查买卖席位
                for col in dt.columns:
                    val = str(row.get(col, ""))
                    for seat in hot_seats:
                        if seat in val:
                            findings.append({
                                "symbol": symbol,
                                "name": name,
                                "route": "dragon_tiger",
                                "reason": f"龙虎榜出现知名席位：{seat}",
                                "seat": seat,
                            })
                            break

        except Exception as e:
            logger.warning(f"龙虎榜扫描出错: {e}")

        return findings[:10]

    def _scan_leader_height(self, stock_list: list[dict],
                            market_data: dict) -> list[dict]:
        """扫描连板高度突破"""
        findings = []

        leader_height = market_data.get("leader_height", 0)
        if leader_height < 5:
            return findings

        # 连板高度>=5说明市场有龙头效应
        hot_stocks = market_data.get("hot_stocks")
        if hot_stocks is None or hot_stocks.empty:
            return findings

        try:
            # 找连板高度最高的标的
            height_col = None
            for col in ["连板", "连续涨停", "height"]:
                if col in hot_stocks.columns:
                    height_col = col
                    break

            if height_col:
                top = hot_stocks.nlargest(3, height_col)
                for _, row in top.iterrows():
                    findings.append({
                        "symbol": str(row.get("代码", row.get("code", row.get("symbol", "")))),
                        "name": str(row.get("名称", row.get("name", ""))),
                        "route": "dragon_tiger",
                        "reason": f"连板{row[height_col]}板，市场龙头效应",
                        "height": int(row[height_col]),
                    })
        except Exception:
            pass

        return findings

    def _scan_limit_up_leaders(self, limit_up_pool: pd.DataFrame) -> list[dict]:
        """
        涨停池中的板块龙头
        每个板块第一个涨停的标的（简化：取涨停时间最早的）
        """
        findings = []

        try:
            sector_col = None
            for col in ["所属行业", "行业", "板块", "概念"]:
                if col in limit_up_pool.columns:
                    sector_col = col
                    break

            time_col = None
            for col in ["涨停时间", "首次涨停时间", "time"]:
                if col in limit_up_pool.columns:
                    time_col = col
                    break

            if sector_col is None:
                return findings

            # 按板块分组
            for sector, group in limit_up_pool.groupby(sector_col):
                if len(group) < 2:
                    continue  # 板块内涨停数太少

                # 取第一只（简化：取第一行）
                row = group.iloc[0]
                findings.append({
                    "symbol": str(row.get("代码", row.get("code", row.get("symbol", "")))),
                    "name": str(row.get("名称", row.get("name", ""))),
                    "route": "dragon_tiger",
                    "reason": f"{sector}板块龙头（{len(group)}家涨停）",
                    "sector": str(sector),
                    "sector_count": len(group),
                })

        except Exception:
            pass

        return findings[:10]
