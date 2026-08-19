"""
反事实推理 — 每个候选标的必须回答三个问题

1. 如果买错了，最可能错在哪？（failure_mode）
2. 市场共识可能在哪里是错的？（consensus_error）— Marks第二层思维
3. 如果不买，错过的机会成本？（opportunity_cost）

设计原则：
- 无法识别风险的标的 → 排除
- 发现强风险信号的 → 排除或降级
- 反事实分析结果附加到 CandidateStock.counterfactual
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.signal import CandidateStock, MarketState

logger = logging.getLogger(__name__)


class CounterfactualAnalyzer:
    """反事实推理器"""

    def analyze(self, candidate: CandidateStock,
                market_state: MarketState | None = None) -> dict:
        """
        对候选标的执行反事实分析

        Returns:
            dict: {
                "failure_mode": str,     # 如果错了，最可能错在哪
                "consensus_error": str,  # 共识可能错在哪
                "opportunity_cost": str, # 不买的机会成本
                "risk_level": str,       # "low"/"medium"/"high"
                "verdict": str,          # "pass"/"caution"/"reject"
            }
        """
        failure = self._most_likely_failure(candidate, market_state)
        consensus = self._where_consensus_wrong(candidate, market_state)
        opp_cost = self._missed_if_skip(candidate, market_state)

        # 综合风险评级
        risk_level = self._assess_risk_level(candidate, failure)

        # 最终裁决
        verdict = self._make_verdict(candidate, risk_level)

        return {
            "failure_mode": failure,
            "consensus_error": consensus,
            "opportunity_cost": opp_cost,
            "risk_level": risk_level,
            "verdict": verdict,
        }

    def _most_likely_failure(self, candidate: CandidateStock,
                             market_state: MarketState | None) -> str:
        """
        如果买错了，最可能错在哪？

        检查维度：
        - 估值过高？
        - 行业见顶？
        - 情绪过度一致？
        - 基本面恶化？
        - 趋势反转？
        """
        risks = []

        # 检查价值维度
        value_score = candidate.get_dimension_score("value")
        if value_score is not None and value_score < 40:
            risks.append("估值偏高，价值维度得分低")

        # 检查行业维度
        industry_score = candidate.get_dimension_score("industry")
        if industry_score is not None and industry_score < 40:
            risks.append("行业景气度不足")

        # 检查情绪维度
        emotion_score = candidate.get_dimension_score("emotion")
        if emotion_score is not None and emotion_score < 30:
            risks.append("市场情绪极差，可能继续下跌")

        # 检查趋势维度
        trend_score = candidate.get_dimension_score("trend")
        if trend_score is not None and trend_score < 35:
            risks.append("趋势破坏，技术面不支持")

        # 检查风险维度
        risk_score = candidate.get_dimension_score("risk")
        if risk_score is not None and risk_score < 30:
            risks.append("风险信号严重")

        # 检查市场状态
        if market_state:
            if market_state.regime == "BEAR":
                risks.append("宏观环境为熊市，系统性风险")
            if market_state.temperature_percentile and market_state.temperature_percentile > 85:
                risks.append("市场温度过高，可能见顶")

        if not risks:
            return "未识别到明显风险，但市场总有不可预见的因素"

        return "最可能错在: " + "；".join(risks[:3])

    def _where_consensus_wrong(self, candidate: CandidateStock,
                               market_state: MarketState | None) -> str:
        """
        市场共识可能在哪里是错的？

        Marks第二层思维：不是"会怎样"，而是"共识的反面是什么"
        """
        # 检查情绪极端
        emotion_judgments = candidate.get_judgments_by_dimension("emotion")

        # 如果情绪评分很高（共识乐观），可能错在过度乐观
        if emotion_judgments:
            avg_emotion = sum(j.score for j in emotion_judgments) / len(emotion_judgments)
            if avg_emotion > 75:
                return "共识可能过度乐观：情绪评分高，但群体过度一致往往是反转信号（勒庞）"
            elif avg_emotion < 30:
                return "共识可能过度悲观：情绪冰点，但群体恐慌往往是布局窗口（炒股养家）"

        # 检查估值共识
        value_judgments = candidate.get_judgments_by_dimension("value")
        if value_judgments:
            avg_value = sum(j.score for j in value_judgments) / len(value_judgments)
            if avg_value > 75:
                return "共识认为估值便宜，但可能是价值陷阱（增长放缓被忽视）"
            elif avg_value < 35:
                return "共识认为估值贵，但可能忽视了成长性（被错杀）"

        return "共识方向不明确，需更多数据判断"

    def _missed_if_skip(self, candidate: CandidateStock,
                        market_state: MarketState | None) -> str:
        """
        如果不买，错过的机会成本？

        检查：
        - 催化剂时间窗口
        - 估值低点的稀缺性
        - 趋势启动的窗口期
        """
        opportunities = []

        # 检查是否有催化剂
        for j in candidate.judgments:
            if j.rule_id == "klm_03" and j.score >= 65:
                opportunities.append("有催化剂即将释放")
            if j.rule_id == "klm_01" and j.score >= 75:
                opportunities.append("安全边际大，估值低点稀缺")
            if j.rule_id == "lvr_02" and j.score >= 70:
                opportunities.append("关键点突破，趋势启动窗口")

        if not opportunities:
            return "机会成本较低：无明显催化剂或估值优势"

        return "如果不买可能错过: " + "；".join(opportunities[:2])

    def _assess_risk_level(self, candidate: CandidateStock,
                           failure_mode: str) -> str:
        """综合风险评级"""
        risk_score = candidate.get_dimension_score("risk")
        if risk_score is not None and risk_score < 25:
            return "high"
        if "未识别到明显风险" in failure_mode:
            return "low"
        if len(failure_mode) > 80:  # 风险描述较长
            return "medium"
        return "medium"

    def _make_verdict(self, candidate: CandidateStock,
                      risk_level: str) -> str:
        """最终裁决"""
        if risk_level == "high":
            return "reject"
        if risk_level == "low" and candidate.composite_score >= 65:
            return "pass"
        if candidate.composite_score >= 55:
            return "caution"
        return "reject"
