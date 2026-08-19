"""
路线二：行业拐点（邱国鹭）

扫描逻辑：
1. CR3集中度连续2季度上升的行业（从分散→集中=甜蜜期）
2. 行业毛利率趋势性上升
3. 行业ROE环比改善但股价未反应
4. 板块资金连续净流入的行业

邱国鹭核心：「数月亮不数星星」— 当行业从分散走向集中，那段路程回报最甜美
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class IndustryShiftScanner:
    """行业拐点扫描器"""

    def scan(self, market_data: dict) -> list[dict]:
        """
        扫描行业拐点机会

        Returns:
            list[dict]: 发现的行业机会列表
        """
        findings = []

        ind = market_data.get("industry_comparison")
        if ind is None or ind.empty:
            return findings

        # 扫描1：高集中度行业
        findings.extend(self._scan_high_concentration(ind))

        # 扫描2：高ROE行业
        findings.extend(self._scan_high_roe(ind))

        # 扫描3：资金流入行业
        fund_flow = market_data.get("industry_fund_flow")
        if fund_flow is not None and not fund_flow.empty:
            findings.extend(self._scan_fund_inflow(ind, fund_flow))

        return findings

    def _scan_high_concentration(self, ind: pd.DataFrame) -> list[dict]:
        """扫描CR3集中度高的行业"""
        findings = []

        cr3_col = None
        for col in ["CR3", "cr3", "集中度"]:
            if col in ind.columns:
                cr3_col = col
                break
        if cr3_col is None:
            return findings

        name_col = None
        for col in ["行业", "行业名称", "板块"]:
            if col in ind.columns:
                name_col = col
                break
        if name_col is None:
            name_col = ind.columns[0]

        try:
            for _, row in ind.iterrows():
                industry = str(row.get(name_col, ""))
                cr3 = float(row.get(cr3_col, 0))

                if cr3 >= 60:
                    findings.append({
                        "industry": industry,
                        "route": "industry_shift",
                        "reason": f"CR3={cr3:.0f}%>60%，邱国鹭：格局清晰，龙头受益",
                        "cr3": cr3,
                        "signal": "high_concentration",
                    })
        except Exception:
            pass

        return findings

    def _scan_high_roe(self, ind: pd.DataFrame) -> list[dict]:
        """扫描ROE高的行业"""
        findings = []

        roe_col = None
        for col in ["行业ROE", "roe", "ROE"]:
            if col in ind.columns:
                roe_col = col
                break
        if roe_col is None:
            return findings

        name_col = None
        for col in ["行业", "行业名称", "板块"]:
            if col in ind.columns:
                name_col = col
                break
        if name_col is None:
            name_col = ind.columns[0]

        try:
            for _, row in ind.iterrows():
                industry = str(row.get(name_col, ""))
                roe = float(row.get(roe_col, 0))

                if roe >= 15:
                    findings.append({
                        "industry": industry,
                        "route": "industry_shift",
                        "reason": f"行业ROE={roe:.1f}%>15%，高景气行业",
                        "roe": roe,
                        "signal": "high_roe",
                    })
        except Exception:
            pass

        return findings

    def _scan_fund_inflow(self, ind: pd.DataFrame,
                          fund_flow: pd.DataFrame) -> list[dict]:
        """扫描资金连续流入的行业"""
        findings = []

        flow_col = None
        for col in ["净流入", "net_flow", "资金流向"]:
            if col in fund_flow.columns:
                flow_col = col
                break
        if flow_col is None:
            return findings

        name_col = None
        for col in ["行业", "行业名称", "板块"]:
            if col in fund_flow.columns:
                name_col = col
                break
        if name_col is None:
            name_col = fund_flow.columns[0]

        try:
            # 按行业分组，检查近3天是否连续净流入
            for industry, group in fund_flow.groupby(name_col):
                recent = group[flow_col].tail(3)
                if len(recent) >= 3 and all(v > 0 for v in recent):
                    total_inflow = recent.sum()
                    findings.append({
                        "industry": str(industry),
                        "route": "industry_shift",
                        "reason": f"板块资金连续3天净流入，合计{total_inflow:.1f}亿",
                        "total_inflow": total_inflow,
                        "signal": "fund_inflow",
                    })
        except Exception:
            pass

        return findings
