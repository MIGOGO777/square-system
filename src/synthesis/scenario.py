"""
情景推演 — 输出条件化结论

三种情景：
- 情景A：大盘高开>1%
- 情景B：平开震荡
- 情景C：低开>1%

每种情景给出具体操作建议（候选标的中哪些可以操作，仓位多少）
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.signal import CandidateStock, MarketState

logger = logging.getLogger(__name__)


class ScenarioPlanner:
    """情景推演器"""

    def plan(self, market_state: MarketState | None,
             candidates: list[CandidateStock]) -> dict:
        """
        生成情景推演计划

        Returns:
            dict: {
                "scenario_a": {"condition": str, "action": str, "details": list},
                "scenario_b": {"condition": str, "action": str, "details": list},
                "scenario_c": {"condition": str, "action": str, "details": list},
                "summary": str,
            }
        """
        regime = market_state.regime if market_state else "SIDEWAYS"
        temp_pct = market_state.temperature_percentile if market_state else 50.0

        scenario_a = self._plan_gap_up(candidates, regime, temp_pct)
        scenario_b = self._plan_flat(candidates, regime, temp_pct)
        scenario_c = self._plan_gap_down(candidates, regime, temp_pct)

        # 综合建议
        summary = self._generate_summary(regime, temp_pct, candidates)

        return {
            "scenario_a": scenario_a,
            "scenario_b": scenario_b,
            "scenario_c": scenario_c,
            "summary": summary,
        }

    def _plan_gap_up(self, candidates: list[CandidateStock],
                     regime: str, temp_pct: float) -> dict:
        """情景A：大盘高开>1%"""
        details = []
        actions = []

        if regime == "BULL":
            actions.append("牛市高开，趋势型标的可轻仓跟随")
            for c in candidates[:3]:
                trend_score = c.get_dimension_score("trend")
                if trend_score and trend_score >= 65:
                    details.append(f"{c.symbol} {c.name}: 趋势{trend_score:.0f}分，可跟随")
        elif regime == "BEAR":
            actions.append("熊市高开警惕诱多，不追高")
            actions.append("已有持仓可考虑减仓锁定利润")
        else:
            actions.append("震荡市高开，不追高，等分歧确认")
            actions.append("优质标的若回调到位可逆向加仓")

        if not details:
            details.append("无适合高开追入的标的")

        return {
            "condition": "大盘高开>1%",
            "action": "；".join(actions),
            "details": details,
        }

    def _plan_flat(self, candidates: list[CandidateStock],
                   regime: str, temp_pct: float) -> dict:
        """情景B：平开震荡"""
        details = []
        actions = []

        actions.append("维持原判断，按候选池操作")

        # 找出评分最高且置信度最高的标的
        sorted_candidates = sorted(candidates,
                                   key=lambda c: c.composite_score * c.composite_confidence,
                                   reverse=True)

        for c in sorted_candidates[:3]:
            score = c.composite_score
            conf = c.composite_confidence
            if score >= 65 and conf >= 0.5:
                details.append(f"{c.symbol} {c.name}: 综合{score:.0f}分 置信{conf:.0%}，可操作")
            elif score >= 55:
                details.append(f"{c.symbol} {c.name}: 综合{score:.0f}分，观望为主")

        if not details:
            details.append("无高置信度标的，建议观望")

        # 仓位建议
        if regime == "BULL":
            actions.append("建议总仓位30-40%")
        elif regime == "BEAR":
            actions.append("建议总仓位10%以下")
        else:
            actions.append("建议总仓位15-25%")

        return {
            "condition": "平开震荡",
            "action": "；".join(actions),
            "details": details,
        }

    def _plan_gap_down(self, candidates: list[CandidateStock],
                       regime: str, temp_pct: float) -> dict:
        """情景C：低开>1%"""
        details = []
        actions = []

        if regime == "BEAR":
            actions.append("熊市低开，严格控制仓位")
            actions.append("只保留高置信度标的，其余减仓")
        elif temp_pct < 30:
            actions.append("市场温度已低+低开，可能是恐慌性杀跌")
            actions.append("Marks/炒股养家：极度恐慌=逆向布局窗口")
            # 找逆向标的
            for c in candidates:
                value_score = c.get_dimension_score("value")
                if value_score and value_score >= 70:
                    details.append(f"{c.symbol} {c.name}: 价值{value_score:.0f}分，恐慌中可逆向")
        else:
            actions.append("低开观察，优质标的跌到位可逆向加仓")
            actions.append("不要恐慌性抛售，检查基本面是否变化")

        if not details:
            details.append("低开时优先观察，不急于操作")

        return {
            "condition": "低开>1%",
            "action": "；".join(actions),
            "details": details,
        }

    def _generate_summary(self, regime: str, temp_pct: float,
                          candidates: list[CandidateStock]) -> str:
        """生成综合建议"""
        high_conf = [c for c in candidates if c.composite_confidence >= 0.6 and c.composite_score >= 65]

        parts = []
        parts.append(f"当前市场{regime}，温度百分位{temp_pct:.0f}%")

        if high_conf:
            names = [f"{c.symbol}" for c in high_conf[:3]]
            parts.append(f"高置信度标的: {', '.join(names)}")
        else:
            parts.append("无高置信度标的，建议观望")

        if regime == "BULL":
            parts.append("牛市策略：顺势持仓，不轻易减仓")
        elif regime == "BEAR":
            parts.append("熊市策略：防守为主，等待恐慌性杀跌后的逆向机会")
        else:
            parts.append("震荡策略：精选个股，控制仓位，耐心等待")

        return "；".join(parts)
