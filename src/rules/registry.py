"""
规则注册表 + 引擎

管理所有原子规则的注册、执行、合成。
每条规则是一个独立函数，接收 EvalContext 返回 AtomicJudgment。

设计原则：
- 规则之间无依赖，可独立执行、并行执行
- 规则通过注册表管理，支持按维度/大师筛选执行
- 合成分数使用置信度加权（confidence × data_quality）
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.core.signal import AtomicJudgment, EvalContext

logger = logging.getLogger(__name__)

# 规则函数类型：接收 EvalContext，返回 AtomicJudgment
RuleFn = Callable[[EvalContext], AtomicJudgment | None]


class RuleRegistry:
    """
    规则注册表

    管理所有原子规则的注册和执行。

    使用方式：
        registry = RuleRegistry()
        registry.register("marks_01", marks_pendulum_rule, "macro", "howard_marks")
        judgments = registry.evaluate_all(context)
    """

    def __init__(self) -> None:
        self._rules: dict[str, dict[str, Any]] = {}

    def register(self, rule_id: str, rule_fn: RuleFn,
                 dimension: str, thinker: str) -> None:
        """
        注册一条规则

        Args:
            rule_id: 规则唯一ID
            rule_fn: 规则函数
            dimension: 所属维度
            thinker: 来源大师
        """
        self._rules[rule_id] = {
            "fn": rule_fn,
            "dimension": dimension,
            "thinker": thinker,
        }
        logger.debug(f"注册规则: {rule_id} ({dimension}/{thinker})")

    def evaluate(self, rule_id: str, context: EvalContext) -> AtomicJudgment | None:
        """执行单条规则"""
        rule = self._rules.get(rule_id)
        if rule is None:
            logger.warning(f"规则 {rule_id} 未注册")
            return None
        try:
            return rule["fn"](context)
        except Exception as e:
            logger.error(f"规则 {rule_id} 执行失败: {e}")
            return None

    def evaluate_dimension(self, dimension: str,
                           context: EvalContext) -> list[AtomicJudgment]:
        """执行某维度的所有规则"""
        results = []
        for rule_id, rule in self._rules.items():
            if rule["dimension"] != dimension:
                continue
            judgment = self.evaluate(rule_id, context)
            if judgment is not None:
                results.append(judgment)
        return results

    def evaluate_thinker(self, thinker: str,
                         context: EvalContext) -> list[AtomicJudgment]:
        """执行某大师的所有规则"""
        results = []
        for rule_id, rule in self._rules.items():
            if rule["thinker"] != thinker:
                continue
            judgment = self.evaluate(rule_id, context)
            if judgment is not None:
                results.append(judgment)
        return results

    def evaluate_all(self, context: EvalContext) -> list[AtomicJudgment]:
        """执行所有已注册规则"""
        results = []
        for rule_id in self._rules:
            judgment = self.evaluate(rule_id, context)
            if judgment is not None:
                results.append(judgment)
        return results

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def dimensions(self) -> list[str]:
        return list(set(r["dimension"] for r in self._rules.values()))

    @property
    def thinkers(self) -> list[str]:
        return list(set(r["thinker"] for r in self._rules.values()))

    def list_rules(self) -> list[dict[str, str]]:
        """列出所有已注册规则"""
        return [
            {"id": rid, "dimension": r["dimension"], "thinker": r["thinker"]}
            for rid, r in self._rules.items()
        ]


def composite_score(judgments: list[AtomicJudgment]) -> tuple[float, float]:
    """
    计算一组判断的加权合成分数

    权重 = confidence × data_quality（置信度越高、数据质量越好，权重越大）

    Returns:
        tuple[float, float]: (合成分数 0-100, 合成置信度 0-1)
    """
    if not judgments:
        return 50.0, 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for j in judgments:
        weight = j.confidence * j.data_quality
        weighted_sum += j.score * weight
        total_weight += weight

    if total_weight <= 0:
        return 50.0, 0.0

    score = weighted_sum / total_weight
    # 合成置信度 = 平均置信度 × 数据覆盖度
    avg_confidence = sum(j.confidence for j in judgments) / len(judgments)
    coverage = len(judgments) / max(1, len(judgments))  # TODO: 与预期规则数比较
    confidence = avg_confidence * coverage

    return round(score, 2), round(min(1.0, confidence), 3)


def score_to_direction(score: float) -> str:
    """分数映射到方向"""
    if score >= 75:
        return "BUY"
    elif score >= 55:
        return "HOLD"
    elif score >= 35:
        return "WARNING"
    else:
        return "SELL"
