"""
因果推断模型 — 简化版DAG反事实分析

将规则式反事实升级为结构因果模型(SCM)。

核心思想：
- 不只问"如果错了可能错在哪"
- 而是问"如果X变成X'，Y会怎样变化"

简化实现：
- 预定义A股常用因果路径（DAG）
- 用线性结构方程做反事实估计
- 不需要DoWhy/CausalNex依赖

因果路径（专家知识）：
M2增速 → 北向资金 → 情绪温度 → 股价
PMI → 行业景气 → 个股盈利 → 股价
情绪温度 → 涨停高度 → 龙头效应 → 股价
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# 预定义因果图（DAG）
# 节点: 变量名
# 边: (原因, 结果, 系数范围)
CAUSAL_GRAPH = {
    "m2_growth": {"north_flow": (0.3, 0.8), "emotion_temp": (0.1, 0.4)},
    "north_flow": {"emotion_temp": (0.2, 0.6), "stock_return": (0.1, 0.3)},
    "pmi": {"industry_sentiment": (0.3, 0.7), "stock_return": (0.1, 0.3)},
    "emotion_temp": {"leader_height": (0.2, 0.5), "stock_return": (0.2, 0.5)},
    "leader_height": {"dragon_tiger_effect": (0.3, 0.6)},
    "industry_sentiment": {"stock_return": (0.2, 0.5)},
    "garch_vol": {"risk_premium": (0.3, 0.7)},
    "risk_premium": {"stock_return": (-0.5, -0.1)},
}


class CausalInferenceModel:
    """简化版因果推断模型"""

    def __init__(self):
        self.graph = CAUSAL_GRAPH

    def analyze(self, market_data: dict, stock_data: dict,
                dimension_scores: dict[str, float]) -> dict[str, Any] | None:
        """
        对候选标的做因果分析

        Args:
            market_data: 市场宏观数据
            stock_data: 个股数据
            dimension_scores: 各维度分数

        Returns:
            dict with causal paths, counterfactuals, and verdict
        """
        # 提取观测值
        observations = self._extract_observations(market_data, stock_data, dimension_scores)
        if not observations:
            return None

        # 找出主要因果路径
        causal_paths = self._find_causal_paths(observations)

        # 反事实分析：最强因果路径上的干预
        counterfactuals = self._run_counterfactuals(observations, causal_paths)

        # 综合判断
        verdict = self._make_verdict(causal_paths, counterfactuals)

        return {
            "causal_paths": causal_paths,
            "counterfactuals": counterfactuals,
            "verdict": verdict,
            "model": "simplified_scm",
        }

    def _extract_observations(self, market_data: dict, stock_data: dict,
                              dimension_scores: dict[str, float]) -> dict[str, float]:
        """从数据中提取因果图节点的观测值"""
        obs = {}

        # 宏观变量
        obs["m2_growth"] = market_data.get("m2_growth", 0)
        obs["pmi"] = market_data.get("pmi", 50)
        obs["emotion_temp"] = market_data.get("emotion_temp", 50)
        obs["leader_height"] = market_data.get("leader_height", 0)

        # 北向资金
        north_flow = 0
        nf = market_data.get("north_flow")
        if nf is not None and hasattr(nf, 'empty') and not nf.empty:
            for col in ("net_flow", "净流入"):
                if col in nf.columns:
                    north_flow = float(nf[col].iloc[-1])
                    break
        obs["north_flow"] = north_flow

        # 行业景气（用industry维度分数代理）
        obs["industry_sentiment"] = dimension_scores.get("industry", 50)

        # 波动率（用risk维度分数代理，反转）
        risk_score = dimension_scores.get("risk", 50)
        obs["garch_vol"] = 100 - risk_score  # 高risk分=低波动

        return obs

    def _find_causal_paths(self, obs: dict[str, float]) -> list[dict]:
        """找出活跃的因果路径"""
        paths = []

        # 路径1: M2 → 北向 → 情绪 → 股价
        m2 = obs.get("m2_growth", 0)
        north = obs.get("north_flow", 0)
        emotion = obs.get("emotion_temp", 50)

        if m2 > 10 and north > 50:
            strength = min(1.0, (m2 - 5) / 10 * north / 200)
            paths.append({
                "path": "M2增速↑ → 北向资金↑ → 情绪升温 → 股价↑",
                "strength": round(strength, 3),
                "direction": "positive",
                "key_variable": "m2_growth",
                "key_value": m2,
            })
        elif m2 < 5 and north < -50:
            strength = min(1.0, (5 - m2) / 10 * abs(north) / 200)
            paths.append({
                "path": "M2增速↓ → 北向资金↓ → 情绪降温 → 股价↓",
                "strength": round(strength, 3),
                "direction": "negative",
                "key_variable": "m2_growth",
                "key_value": m2,
            })

        # 路径2: PMI → 行业景气 → 股价
        pmi = obs.get("pmi", 50)
        industry = obs.get("industry_sentiment", 50)

        if pmi > 51 and industry > 60:
            strength = min(1.0, (pmi - 50) / 5 * (industry - 50) / 50)
            paths.append({
                "path": "PMI↑ → 行业景气↑ → 股价↑",
                "strength": round(strength, 3),
                "direction": "positive",
                "key_variable": "pmi",
                "key_value": pmi,
            })
        elif pmi < 49 and industry < 40:
            strength = min(1.0, (50 - pmi) / 5 * (50 - industry) / 50)
            paths.append({
                "path": "PMI↓ → 行业景气↓ → 股价↓",
                "strength": round(strength, 3),
                "direction": "negative",
                "key_variable": "pmi",
                "key_value": pmi,
            })

        # 路径3: 情绪 → 龙头高度 → 龙头效应
        leader = obs.get("leader_height", 0)
        if emotion > 65 and leader >= 5:
            strength = min(1.0, (emotion - 50) / 50 * leader / 10)
            paths.append({
                "path": "情绪高涨 → 龙头高度↑ → 龙头效应强",
                "strength": round(strength, 3),
                "direction": "positive",
                "key_variable": "emotion_temp",
                "key_value": emotion,
            })

        # 路径4: 波动率 → 风险溢价 → 股价
        vol = obs.get("garch_vol", 50)
        if vol > 70:
            strength = min(1.0, (vol - 50) / 50)
            paths.append({
                "path": "波动率↑ → 风险溢价↑ → 股价承压",
                "strength": round(strength, 3),
                "direction": "negative",
                "key_variable": "garch_vol",
                "key_value": vol,
            })

        # 按强度排序
        paths.sort(key=lambda x: x["strength"], reverse=True)
        return paths

    def _run_counterfactuals(self, obs: dict[str, float],
                             paths: list[dict]) -> list[dict]:
        """对最强路径做反事实分析"""
        counterfactuals = []

        for path in paths[:2]:  # 最多2条路径
            key_var = path["key_variable"]
            key_val = path["key_value"]

            # 反事实：如果关键变量变成0
            cf_val = 0 if key_val > 0 else 50  # 中性值

            # 估算影响（线性模型）
            impact_estimate = self._estimate_intervention(obs, key_var, cf_val)

            counterfactuals.append({
                "intervention": f"若{key_var}={cf_val}（实际={key_val:.1f}）",
                "estimated_impact": impact_estimate,
                "path": path["path"],
            })

        return counterfactuals

    def _estimate_intervention(self, obs: dict[str, float],
                               var: str, new_val: float) -> dict[str, Any]:
        """估算干预效果（线性SCM）"""
        # 简化的线性因果模型
        # stock_return ≈ sum(beta_i * X_i)
        coeffs = {
            "m2_growth": 0.15,
            "north_flow": 0.08,
            "pmi": 0.12,
            "emotion_temp": 0.10,
            "leader_height": 0.05,
            "garch_vol": -0.08,
            "industry_sentiment": 0.10,
        }

        # 当前估计
        current_effect = sum(obs.get(v, 0) * c for v, c in coeffs.items())

        # 反事实估计
        cf_obs = dict(obs)
        cf_obs[var] = new_val
        cf_effect = sum(cf_obs.get(v, 0) * c for v, c in coeffs.items())

        diff = cf_effect - current_effect

        return {
            "current_estimate": round(current_effect, 2),
            "counterfactual_estimate": round(cf_effect, 2),
            "difference": round(diff, 2),
            "direction": "positive" if diff > 0 else "negative",
        }

    def _make_verdict(self, paths: list[dict],
                      counterfactuals: list[dict]) -> dict[str, Any]:
        """综合因果判断"""
        if not paths:
            return {
                "judgment": "neutral",
                "reason": "无显著因果路径",
                "confidence": 0.3,
            }

        # 正向路径强度
        pos_strength = sum(p["strength"] for p in paths if p["direction"] == "positive")
        neg_strength = sum(p["strength"] for p in paths if p["direction"] == "negative")

        net = pos_strength - neg_strength

        if net > 0.3:
            judgment = "bullish"
            reason = f"正向因果路径占优（强度{pos_strength:.2f} vs {neg_strength:.2f}）"
        elif net < -0.3:
            judgment = "bearish"
            reason = f"负向因果路径占优（强度{neg_strength:.2f} vs {pos_strength:.2f}）"
        else:
            judgment = "neutral"
            reason = f"因果力量均衡（正{pos_strength:.2f} vs 负{neg_strength:.2f}）"

        return {
            "judgment": judgment,
            "reason": reason,
            "confidence": round(min(1.0, abs(net) + 0.3), 2),
            "positive_strength": round(pos_strength, 3),
            "negative_strength": round(neg_strength, 3),
        }
