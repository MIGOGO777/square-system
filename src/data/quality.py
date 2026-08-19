"""
数据质量评估器

给每条数据打可信度标签，可信度贯穿整个决策链：
数据质量 → 规则置信度 → 决策置信度 → 仓位

可信度等级：
- REALTIME (0.9): 当日实时数据（涨停池、北向资金、龙虎榜）
- RECENT (0.7): 近期数据（本周宏观指标、月度CPI）
- HISTORICAL (0.5): 历史数据（季度财报、年度ROE）
- ESTIMATED (0.3): 估算/缺失数据（炸板率估算、无行业均值时的默认值）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityAssessor:
    """
    数据质量评估器

    根据数据类型、时效性、完整性三个维度评估可信度。
    """

    # 可信度等级
    REALTIME: float = 0.9
    RECENT: float = 0.7
    HISTORICAL: float = 0.5
    ESTIMATED: float = 0.3

    # 数据类型 → 默认时效性等级
    DATA_TYPE_DEFAULTS: dict[str, tuple[float, float]] = {
        # (默认可信度, 最大有效小时数)
        "limit_up_pool": (REALTIME, 8),       # 涨停池，当日有效
        "north_flow": (REALTIME, 4),           # 北向资金，盘中有效
        "hot_stocks": (REALTIME, 8),           # 强势股，当日有效
        "dragon_tiger": (REALTIME, 24),        # 龙虎榜，当日有效
        "fund_flow": (REALTIME, 8),            # 资金流向，当日有效
        "kline_daily": (REALTIME, 8),          # 日K线，当日有效
        "valuation": (RECENT, 48),             # 估值数据，2天有效
        "industry_comparison": (RECENT, 168),  # 行业对比，1周有效
        "macro_m2": (HISTORICAL, 720),         # M2，1月有效
        "macro_pmi": (HISTORICAL, 720),        # PMI，1月有效
        "macro_cpi": (HISTORICAL, 720),        # CPI，1月有效
        "pe_band": (HISTORICAL, 168),          # PE分位，1周有效
        "quarterly": (HISTORICAL, 2160),       # 季报，3月有效
        "f10": (HISTORICAL, 720),              # F10，1月有效
        "lockup_expiry": (RECENT, 168),        # 解禁日历，1周有效
    }

    def assess(self, data_type: str, data_age_hours: float | None = None,
               completeness: float = 1.0) -> float:
        """
        评估数据可信度

        Args:
            data_type: 数据类型（如 "limit_up_pool", "quarterly"）
            data_age_hours: 数据年龄（小时），None 则使用默认值
            completeness: 数据完整性 0-1（缺失字段越多越低）

        Returns:
            float: 可信度 0-1
        """
        default_quality, max_age = self.DATA_TYPE_DEFAULTS.get(
            data_type, (self.ESTIMATED, 24)
        )

        if data_age_hours is None:
            quality = default_quality
        else:
            # 时效性衰减：超过最大有效时间后，可信度线性衰减
            if data_age_hours <= max_age:
                quality = default_quality
            else:
                decay = (data_age_hours - max_age) / max_age
                quality = max(0.1, default_quality * (1.0 - decay * 0.5))

        # 完整性修正
        quality *= completeness

        return round(max(0.1, min(1.0, quality)), 3)

    def assess_dataframe(self, df: pd.DataFrame, data_type: str) -> float:
        """
        评估 DataFrame 的整体可信度

        基于非空值比例计算完整性。

        Args:
            df: 数据
            data_type: 数据类型

        Returns:
            float: 可信度 0-1
        """
        if df is None or df.empty:
            return self.ESTIMATED

        # 完整性 = 非空值占比
        total_cells = df.size
        if total_cells == 0:
            return self.ESTIMATED
        non_null = df.notna().sum().sum()
        completeness = non_null / total_cells

        return self.assess(data_type, completeness=completeness)

    def assess_dict(self, data: dict, data_type: str) -> float:
        """
        评估 dict 数据的可信度

        基于非空值比例。
        """
        if not data:
            return self.ESTIMATED

        total = len(data)
        non_empty = sum(1 for v in data.values() if v is not None and v != "" and v != 0)
        completeness = non_empty / total if total > 0 else 0.0

        return self.assess(data_type, completeness=completeness)

    def get_quality_label(self, quality: float) -> str:
        """返回可读的质量标签"""
        if quality >= 0.85:
            return "高"
        elif quality >= 0.65:
            return "中"
        elif quality >= 0.45:
            return "低"
        else:
            return "极低"
