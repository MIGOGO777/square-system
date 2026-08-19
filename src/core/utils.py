"""公共工具函数"""

from __future__ import annotations

import pandas as pd


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """查找DataFrame中第一个匹配的列名"""
    for col in candidates:
        if col in df.columns:
            return col
    return None
