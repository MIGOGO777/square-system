"""
强化学习规则权重调整器 — 轻量版

不替换规则引擎，只动态调整各维度的权重。

方法：Contextual Bandit（上下文赌博机）
- 状态(context)：市场状态 + 各维度分数
- 动作(action)：6个维度的权重分配
- 奖励(reward)：事后收益率

比完整RL简单，避免过拟合，保留可解释性。

权重调整逻辑：
1. 维度分数高 + 事后涨 → 该维度权重上升
2. 维度分数高 + 事后跌 → 该维度权重下降
3. 用指数移动平均(EMA)平滑权重变化
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RLWeightAdjuster:
    """基于上下文赌博机的规则权重动态调整器"""

    DIMENSIONS = ["value", "industry", "emotion", "trend", "macro", "risk"]
    DEFAULT_WEIGHTS = {d: 1.0 / 6 for d in DIMENSIONS}

    def __init__(self, learning_rate: float = 0.05,
                 decay: float = 0.95, state_path: str | None = None):
        self.lr = learning_rate
        self.decay = decay
        self.state_path = state_path
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.history: list[dict] = []
        self._load_state()

    def _load_state(self):
        """加载历史权重状态"""
        if not self.state_path:
            return
        try:
            path = Path(self.state_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.weights = state.get("weights", self.DEFAULT_WEIGHTS)
                self.history = state.get("history", [])[-100:]  # 保留最近100条
                logger.info(f"加载RL权重状态: {self.weights}")
        except Exception as e:
            logger.debug(f"加载RL状态失败: {e}")

    def _save_state(self):
        """保存权重状态"""
        if not self.state_path:
            return
        try:
            path = Path(self.state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"weights": self.weights, "history": self.history[-100:]},
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"保存RL状态失败: {e}")

    def get_weights(self) -> dict[str, float]:
        """获取当前维度权重（归一化）"""
        total = sum(self.weights.values())
        if total <= 0:
            return dict(self.DEFAULT_WEIGHTS)
        return {k: v / total for k, v in self.weights.items()}

    def update(self, dimension_scores: dict[str, float],
               actual_return: float, market_state: str = "SIDEWAYS"):
        """
        根据事后收益率更新权重

        Args:
            dimension_scores: 各维度分数 (0-100)
            actual_return: 实际收益率（百分比）
            market_state: 市场状态
        """
        # 归一化分数到[0, 1]
        norm_scores = {}
        for d in self.DIMENSIONS:
            s = dimension_scores.get(d, 50)
            norm_scores[d] = max(0, min(1, s / 100))

        # 计算各维度的"贡献信号"
        # 高分维度 + 正收益 → 奖励
        # 高分维度 + 负收益 → 惩罚
        reward_direction = 1 if actual_return > 0 else -1
        reward_magnitude = min(abs(actual_return) / 5.0, 1.0)  # 标准化

        for d in self.DIMENSIONS:
            score = norm_scores[d]
            # 信号：分数与收益方向的一致性
            signal = score * reward_direction * reward_magnitude

            # EMA更新权重
            self.weights[d] = self.weights[d] * self.decay + self.lr * signal
            self.weights[d] = max(0.01, self.weights[d])  # 下限

        # 记录历史
        self.history.append({
            "dimension_scores": dimension_scores,
            "return": actual_return,
            "market_state": market_state,
            "weights": self.get_weights(),
        })

        self._save_state()
        logger.debug(f"RL权重更新: {self.get_weights()}")

    def analyze(self, dimension_scores: dict[str, float]) -> dict[str, Any]:
        """
        分析当前权重配置

        Args:
            dimension_scores: 各维度分数

        Returns:
            dict with adjusted weights and analysis
        """
        weights = self.get_weights()

        # 找出最高和最低权重的维度
        sorted_dims = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_dims[0]
        weakest = sorted_dims[-1]

        # 计算权重与分数的一致性
        consistency = 0
        for d in self.DIMENSIONS:
            score = dimension_scores.get(d, 50) / 100
            weight = weights.get(d, 1 / 6)
            # 高分应该对应高权重
            consistency += score * weight

        return {
            "weights": {k: round(v, 3) for k, v in weights.items()},
            "strongest_dim": strongest[0],
            "strongest_weight": round(strongest[1], 3),
            "weakest_dim": weakest[0],
            "weakest_weight": round(weakest[1], 3),
            "consistency": round(consistency, 3),
            "history_count": len(self.history),
            "model": "contextual_bandit",
        }

    def suggest_composite_score(self, dimension_scores: dict[str, float]) -> float:
        """
        用RL权重计算加权综合分

        Args:
            dimension_scores: 各维度分数

        Returns:
            float: 加权综合分 (0-100)
        """
        weights = self.get_weights()
        total = 0
        weight_sum = 0

        for d in self.DIMENSIONS:
            score = dimension_scores.get(d, 50)
            w = weights.get(d, 1 / 6)
            total += score * w
            weight_sum += w

        return total / weight_sum if weight_sum > 0 else 50.0
