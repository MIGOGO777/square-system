"""
正方形系统 v2.3 — 核心数据结构

三大核心数据结构：
- AtomicJudgment: 原子化判断结果（系统最核心的数据单元）
- CandidateStock: 候选标的（包含完整决策上下文）
- MarketState: 市场状态（动态阈值版本）

设计原则：
- 每个判断都附带置信度，置信度来源于数据质量
- 所有结构支持序列化/反序列化，便于缓存和传输
- 保留 v1 的 TradeSignal 用于最终输出兼容
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ──────────────────────────────────────────────────────────────
# 原子化判断结果 — 系统最核心的数据结构
# ──────────────────────────────────────────────────────────────

@dataclass
class AtomicJudgment:
    """
    原子化判断结果

    每条大师规则独立产出一个 AtomicJudgment。
    所有维度的判断都用同一个结构，便于跨维度合成。

    Attributes:
        rule_id: 规则唯一ID，如 "marks_01_pendulum"
        rule_name: 规则可读名，如 "钟摆位置判断"
        thinker: 来源大师，如 "howard_marks"
        dimension: 所属维度 value/industry/emotion/trend/macro/risk
        score: 评分 0-100
        confidence: 置信度 0-1（基于数据质量和规则可靠性）
        data_quality: 数据质量 0-1
        direction: BUY/SELL/HOLD/WARNING
        reason: 判断理由（≤100字）
        metadata: 附加信息（自由格式）
    """
    rule_id: str
    rule_name: str
    thinker: str
    dimension: str
    score: float
    confidence: float
    data_quality: float
    direction: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "thinker": self.thinker,
            "dimension": self.dimension,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 3),
            "data_quality": round(self.data_quality, 3),
            "direction": self.direction,
            "reason": self.reason,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────
# 候选标的 — 包含完整决策上下文
# ──────────────────────────────────────────────────────────────

@dataclass
class CandidateStock:
    """
    候选标的

    包含一只股票的所有相关判断、合成分数、反事实分析。
    支持从主动发现引擎产出（discovery_route 非空）。

    Attributes:
        symbol: 股票代码
        name: 股票名称
        industry: 所属行业
        judgments: 所有相关原子判断
        composite_score: 加权合成分数 0-100
        composite_confidence: 合成置信度 0-1
        counterfactual: 反事实分析结果
        discovery_route: 发现路线（如 "contrarian"/"dragon_tiger"，空=常规筛选）
    """
    symbol: str
    name: str
    industry: str
    judgments: list[AtomicJudgment] = field(default_factory=list)
    composite_score: float = 0.0
    composite_confidence: float = 0.0
    counterfactual: dict[str, Any] = field(default_factory=dict)
    discovery_route: str = ""

    def get_judgments_by_dimension(self, dimension: str) -> list[AtomicJudgment]:
        """获取指定维度的所有判断"""
        return [j for j in self.judgments if j.dimension == dimension]

    def get_dimension_score(self, dimension: str) -> float | None:
        """获取指定维度的加权合成分数"""
        dim_judgments = self.get_judgments_by_dimension(dimension)
        if not dim_judgments:
            return None
        total_weight = sum(j.confidence * j.data_quality for j in dim_judgments)
        if total_weight <= 0:
            return None
        return sum(
            j.score * j.confidence * j.data_quality for j in dim_judgments
        ) / total_weight

    def to_dict(self) -> dict[str, Any]:
        # 计算各维度分数
        dimension_scores = {}
        for dim in set(j.dimension for j in self.judgments):
            score = self.get_dimension_score(dim)
            if score is not None:
                dimension_scores[dim] = round(score, 1)

        return {
            "symbol": self.symbol,
            "name": self.name,
            "industry": self.industry,
            "composite_score": round(self.composite_score, 2),
            "composite_confidence": round(self.composite_confidence, 3),
            "discovery_route": self.discovery_route,
            "counterfactual": self.counterfactual,
            "dimension_scores": dimension_scores,
            "judgments": [j.to_dict() for j in self.judgments],
        }


# ──────────────────────────────────────────────────────────────
# 市场状态 — 动态阈值版本
# ──────────────────────────────────────────────────────────────

@dataclass
class MarketState:
    """
    市场状态

    使用动态百分位替代固定阈值。
    温度的 BULL/BEAR 判定基于最近60天的分布，而非固定值。

    Attributes:
        regime: BULL/BEAR/SIDEWAYS
        temperature: 绝对温度 0-100
        temperature_percentile: 在最近60天中的百分位 0-100
        pendulum_position: 钟摆位置描述
        emotion_phase: 情绪阶段（冰点/试探/发酵/高潮/分歧/退潮）
        judgments: 市场级别的原子判断
        confirmed_by: 确认层列表（如 ["macro", "emotion"]）
    """
    regime: str = "SIDEWAYS"
    temperature: float = 50.0
    temperature_percentile: float = 50.0
    pendulum_position: str = "无数据"
    emotion_phase: str = "试探"
    judgments: list[AtomicJudgment] = field(default_factory=list)
    confirmed_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "temperature": round(self.temperature, 1),
            "temperature_percentile": round(self.temperature_percentile, 1),
            "pendulum_position": self.pendulum_position,
            "emotion_phase": self.emotion_phase,
            "confirmed_by": self.confirmed_by,
            "judgments": [j.to_dict() for j in self.judgments],
        }


# ──────────────────────────────────────────────────────────────
# 通用评估上下文 — 规则执行时的数据容器
# ──────────────────────────────────────────────────────────────

@dataclass
class EvalContext:
    """
    规则评估上下文

    每条规则执行时接收的统一数据容器。
    包含个股数据、市场数据、历史数据。

    Attributes:
        fetcher: DataFetcher 实例
        quality: DataQualityAssessor 实例
        stock_data: 个股数据（quarterly/valuation/f10/kline）
        market_data: 市场数据（emotion/north_flow/limit_up等）
        history_data: 历史数据（recent_60d_temps/trades等）
    """
    fetcher: Any = None
    quality: Any = None
    stock_data: dict[str, Any] = field(default_factory=dict)
    market_data: dict[str, Any] = field(default_factory=dict)
    history_data: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
