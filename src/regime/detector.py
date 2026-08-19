"""
市场状态检测器 — 动态阈值 + 多大师确认

核心设计：
- 温度阈值从固定值(70/30)改为滚动百分位（>80百分位=BULL，<20百分位=BEAR）
- 市场状态判定需要至少两层确认（宏观+情绪+趋势，2/3通过才确认）
- 输出 MarketState 包含完整决策上下文

温度计算：6维度加权
- 情绪温度（涨停数/炸板率/连板高度）× 0.25
- 北向资金方向 × 0.15
- 涨停板块扩散度 × 0.10
- 宏观指标（M2/PMI）× 0.20
- 估值分位 × 0.20
- 趋势强度 × 0.10
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.signal import AtomicJudgment, MarketState

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    市场状态检测器

    动态阈值 + 多层确认
    """

    # 温度维度权重
    WEIGHTS = {
        "emotion": 0.25,
        "north_flow": 0.15,
        "contagion": 0.10,
        "macro": 0.20,
        "valuation": 0.20,
        "trend": 0.10,
    }

    def detect(self, market_data: dict,
               history_temps: list[float] | None = None) -> MarketState:
        """
        检测市场状态

        Args:
            market_data: 市场数据
            history_temps: 最近60天的温度历史（用于计算百分位）

        Returns:
            MarketState: 市场状态
        """
        # 1. 计算当前温度（6维度加权）
        temperature = self._calc_temperature(market_data)

        # 2. 计算温度百分位（动态阈值）
        if history_temps and len(history_temps) >= 10:
            percentile = self._calc_percentile(temperature, history_temps)
        else:
            percentile = temperature  # 无历史时直接用温度作为百分位

        # 3. 三层确认：宏观+情绪+趋势
        macro_signal = self._check_macro_layer(market_data, percentile)
        emotion_signal = self._check_emotion_layer(market_data)
        trend_signal = self._check_trend_layer(market_data)

        confirmations = {
            "macro": macro_signal,
            "emotion": emotion_signal,
            "trend": trend_signal,
        }

        # 4. 综合判定
        confirmed_by = [k for k, v in confirmations.items() if v != "NEUTRAL"]
        bull_votes = sum(1 for v in confirmations.values() if v == "BULL")
        bear_votes = sum(1 for v in confirmations.values() if v == "BEAR")

        if bull_votes >= 2:
            regime = "BULL"
        elif bear_votes >= 2:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"

        # 5. 钟摆位置
        pendulum = self._calc_pendulum_position(percentile, market_data)

        # 6. 情绪阶段
        emotion_phase = market_data.get("emotion_phase", "未知")

        return MarketState(
            regime=regime,
            temperature=temperature,
            temperature_percentile=percentile,
            pendulum_position=pendulum,
            emotion_phase=emotion_phase,
            confirmed_by=confirmed_by,
        )

    # ──────────────────────────────────────────────────────────────
    # 温度计算
    # ──────────────────────────────────────────────────────────────

    def _calc_temperature(self, market_data: dict) -> float:
        """
        计算市场温度（0-100）

        6维度加权：
        - 情绪温度（涨停数/炸板率/连板高度）× 0.25
        - 北向资金方向 × 0.15
        - 涨停板块扩散度 × 0.10
        - 宏观指标（M2/PMI）× 0.20
        - 估值分位 × 0.20
        - 趋势强度 × 0.10
        """
        scores = {}

        # 情绪维度
        emotion_temp = market_data.get("emotion_temp", 50.0)
        limit_up = market_data.get("limit_up_count", 0)
        break_rate = market_data.get("break_rate", 0.0)
        leader_height = market_data.get("leader_height", 0)

        emotion_score = emotion_temp
        if limit_up >= 80:
            emotion_score = min(100, emotion_score + 15)
        elif limit_up <= 20:
            emotion_score = max(0, emotion_score - 15)
        if break_rate > 30:
            emotion_score = max(0, emotion_score - 10)
        if leader_height >= 7:
            emotion_score = min(100, emotion_score + 10)
        scores["emotion"] = max(0.0, min(100.0, emotion_score))

        # 北向资金
        north_flow = market_data.get("north_flow")
        north_score = 50.0
        if north_flow is not None and not north_flow.empty:
            try:
                flow_col = None
                for col in ("net_flow", "净流入", "north_net"):
                    if col in north_flow.columns:
                        flow_col = col
                        break
                if flow_col:
                    flow_val = float(north_flow[flow_col].iloc[-1])
                    if flow_val > 200:
                        north_score = 85.0
                    elif flow_val > 100:
                        north_score = 70.0
                    elif flow_val > 0:
                        north_score = 55.0
                    elif flow_val > -100:
                        north_score = 40.0
                    elif flow_val > -200:
                        north_score = 25.0
                    else:
                        north_score = 10.0
            except Exception:
                pass
        scores["north_flow"] = north_score

        # 涨停板块扩散度
        limit_up_pool = market_data.get("limit_up_pool")
        contagion_score = 50.0
        if limit_up_pool is not None and not limit_up_pool.empty:
            sector_col = None
            for col in ["所属行业", "行业", "板块", "概念"]:
                if col in limit_up_pool.columns:
                    sector_col = col
                    break
            if sector_col:
                unique_sectors = limit_up_pool[sector_col].nunique()
                if unique_sectors >= 15:
                    contagion_score = 85.0
                elif unique_sectors >= 10:
                    contagion_score = 65.0
                elif unique_sectors >= 5:
                    contagion_score = 45.0
                else:
                    contagion_score = 25.0
        scores["contagion"] = contagion_score

        # 宏观指标
        m2_growth = market_data.get("m2_growth", 8.0)
        pmi = market_data.get("pmi", 50.0)
        macro_score = 50.0
        if m2_growth > 10:
            macro_score += 15.0
        elif m2_growth < 6:
            macro_score -= 10.0
        if pmi > 52:
            macro_score += 15.0
        elif pmi > 50:
            macro_score += 5.0
        elif pmi < 48:
            macro_score -= 15.0
        scores["macro"] = max(0.0, min(100.0, macro_score))

        # 估值分位
        pe_percentile = market_data.get("pe_percentile", 50.0)
        scores["valuation"] = pe_percentile

        # 趋势强度（用涨跌幅代理）
        market_change = market_data.get("market_change_pct", 0)
        trend_score = 50.0 + market_change * 5
        scores["trend"] = max(0.0, min(100.0, trend_score))

        # 加权合成
        total = sum(scores.get(dim, 50.0) * w for dim, w in self.WEIGHTS.items())
        return round(max(0.0, min(100.0, total)), 1)

    # ──────────────────────────────────────────────────────────────
    # 动态百分位
    # ──────────────────────────────────────────────────────────────

    def _calc_percentile(self, current: float, history: list[float]) -> float:
        """
        计算当前温度在历史中的百分位

        >80百分位 = BULL，<20百分位 = BEAR
        """
        if not history:
            return current

        sorted_hist = sorted(history)
        count_below = sum(1 for h in sorted_hist if h < current)
        percentile = (count_below / len(sorted_hist)) * 100

        return round(percentile, 1)

    # ──────────────────────────────────────────────────────────────
    # 三层确认
    # ──────────────────────────────────────────────────────────────

    def _check_macro_layer(self, market_data: dict, percentile: float) -> str:
        """宏观层确认"""
        m2_growth = market_data.get("m2_growth", 8.0)
        pmi = market_data.get("pmi", 50.0)

        if percentile > 80 and m2_growth > 8 and pmi > 50:
            return "BULL"
        elif percentile < 20 and pmi < 49:
            return "BEAR"
        return "NEUTRAL"

    def _check_emotion_layer(self, market_data: dict) -> str:
        """情绪层确认"""
        phase = market_data.get("emotion_phase", "")
        limit_up = market_data.get("limit_up_count", 0)

        if phase in ("发酵", "高潮") and limit_up >= 50:
            return "BULL"
        elif phase in ("冰点", "退潮") and limit_up <= 30:
            return "BEAR"
        return "NEUTRAL"

    def _check_trend_layer(self, market_data: dict) -> str:
        """趋势层确认"""
        market_change = market_data.get("market_change_pct", 0)

        if market_change > 2:
            return "BULL"
        elif market_change < -2:
            return "BEAR"
        return "NEUTRAL"

    # ──────────────────────────────────────────────────────────────
    # 钟摆位置
    # ──────────────────────────────────────────────────────────────

    def _calc_pendulum_position(self, percentile: float, market_data: dict) -> str:
        """
        钟摆位置描述

        极度偏左（极度悲观）← 偏左 ← 中性 → 偏右 → 极度偏右（极度乐观）
        """
        if percentile <= 15:
            return "极度偏左（极度悲观）"
        elif percentile <= 30:
            return "偏左（悲观）"
        elif percentile <= 45:
            return "温和偏左"
        elif percentile <= 55:
            return "中性"
        elif percentile <= 70:
            return "温和偏右"
        elif percentile <= 85:
            return "偏右（乐观）"
        else:
            return "极度偏右（极度乐观）"
