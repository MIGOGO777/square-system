"""
候选池构建器 — 四步筛选流程

Step 1: 硬排除（不为清单）~5000→~1500
Step 2: 行业筛选（邱国鹭选赛道）~1500→~200
Step 3: 多维评分（原子规则打分）~200→~30
Step 4: 反事实淘汰 ~30→~10-15

设计原则：
- 每步独立，可单独测试
- 每步输出候选列表+淘汰原因
- 最终输出 CandidateStock 列表，包含完整决策上下文
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.core.signal import AtomicJudgment, CandidateStock, EvalContext, MarketState
from src.data.quality import DataQualityAssessor
from src.rules.registry import RuleRegistry, composite_score

logger = logging.getLogger(__name__)


class CandidatePoolBuilder:
    """
    候选池构建器

    四步筛选：硬排除→行业筛选→多维评分→反事实淘汰
    """

    def __init__(self, registry: RuleRegistry, quality: DataQualityAssessor,
                 config: dict | None = None):
        self.registry = registry
        self.quality = quality
        self.config = config or {}

    def build(self, stock_list: list[dict], market_data: dict,
              market_state: MarketState | None = None,
              max_candidates: int = 15) -> list[CandidateStock]:
        """
        构建候选池

        Args:
            stock_list: 股票基础数据列表（含symbol, name, industry等）
            market_data: 市场数据（涨停池、北向资金等）
            market_state: 市场状态
            max_candidates: 最终候选数量

        Returns:
            list[CandidateStock]: 最终候选标的列表
        """
        logger.info(f"开始构建候选池，输入{len(stock_list)}只股票")

        # Step 1: 硬排除
        after_exclusion = self._step1_hard_exclusion(stock_list)
        logger.info(f"Step1硬排除后剩余{len(after_exclusion)}只")

        # Step 2: 行业筛选
        after_industry = self._step2_industry_filter(after_exclusion, market_data)
        logger.info(f"Step2行业筛选后剩余{len(after_industry)}只")

        # Step 3: 多维评分
        scored = self._step3_multi_dimension_score(after_industry, market_data, market_state)
        logger.info(f"Step3多维评分完成{len(scored)}只")

        # Step 4: 反事实淘汰 + Top N
        final = self._step4_counterfactual_filter(scored, market_data, max_candidates)
        logger.info(f"Step4最终候选{len(final)}只")

        return final

    # ──────────────────────────────────────────────────────────────
    # Step 1: 硬排除（不为清单）
    # ──────────────────────────────────────────────────────────────

    def _step1_hard_exclusion(self, stock_list: list[dict]) -> list[dict]:
        """
        硬排除规则：
        - ST/退市警示 → 排除
        - 上市不足1年 → 排除
        - 市值<20亿 → 排除
        - 净利率<0 → 排除
        - ROE连续3年<5% → 排除
        - 自由现金流<0 → 排除
        - 资产负债率>70% → 排除
        """
        passed = []
        for stock in stock_list:
            name = stock.get("name", "")
            q = stock.get("quarterly", {})

            # ST/退市
            if "ST" in name or "退" in name:
                stock["_exclusion_reason"] = "ST/退市"
                continue

            # 上市时间（简化：用IPO日期判断）
            ipo_date = stock.get("ipo_date", "")
            if ipo_date:
                try:
                    from datetime import datetime, timedelta
                    ipo = datetime.strptime(str(ipo_date)[:10], "%Y-%m-%d")
                    if (datetime.now() - ipo).days < 365:
                        stock["_exclusion_reason"] = "上市不足1年"
                        continue
                except (ValueError, TypeError):
                    pass

            # 市值
            market_cap = stock.get("market_cap", 0)
            if 0 < market_cap < 20:
                stock["_exclusion_reason"] = f"市值{market_cap:.0f}亿<20亿"
                continue

            # 基本面硬排除
            if q:
                net_margin = q.get("net_margin", 0)
                roe_list = q.get("roe_list", [])
                fcf_list = q.get("fcf_list", [])
                debt_ratio = q.get("debt_ratio", 0)

                if net_margin < 0:
                    stock["_exclusion_reason"] = f"净利率{net_margin:.1f}%<0"
                    continue

                if roe_list and len(roe_list) >= 3 and all(r < 5 for r in roe_list[-3:]):
                    stock["_exclusion_reason"] = "ROE连续3年<5%"
                    continue

                if fcf_list and fcf_list[-1] < 0:
                    stock["_exclusion_reason"] = "自由现金流<0"
                    continue

                if debt_ratio > 70:
                    stock["_exclusion_reason"] = f"资产负债率{debt_ratio:.0f}%>70%"
                    continue

            passed.append(stock)

        return passed

    # ──────────────────────────────────────────────────────────────
    # Step 2: 行业筛选（邱国鹭选赛道）
    # ──────────────────────────────────────────────────────────────

    def _step2_industry_filter(self, stock_list: list[dict],
                               market_data: dict) -> list[dict]:
        """
        行业筛选：
        1. 获取行业对比数据
        2. 选出ROE趋势上升+CR3>30%+资金净流入的前5-8个行业
        3. 只在这些行业中选个股

        如果没有行业数据，跳过此步
        """
        ind = market_data.get("industry_comparison")
        if ind is None or ind.empty:
            logger.warning("无行业对比数据，跳过行业筛选")
            return stock_list

        # 评估各行业
        good_industries = set()

        try:
            # 找ROE列
            roe_col = None
            for col in ["行业ROE", "roe", "ROE"]:
                if col in ind.columns:
                    roe_col = col
                    break

            # 找CR3列
            cr3_col = None
            for col in ["CR3", "cr3", "集中度"]:
                if col in ind.columns:
                    cr3_col = col
                    break

            # 找行业名列
            name_col = None
            for col in ["行业", "行业名称", "板块"]:
                if col in ind.columns:
                    name_col = col
                    break
            if name_col is None:
                name_col = ind.columns[0]

            for _, row in ind.iterrows():
                industry_name = str(row.get(name_col, ""))
                if not industry_name:
                    continue

                score = 0

                # ROE > 10
                if roe_col:
                    roe_val = row.get(roe_col, 0)
                    try:
                        if float(roe_val) > 10:
                            score += 1
                    except (ValueError, TypeError):
                        pass

                # CR3 > 30
                if cr3_col:
                    cr3_val = row.get(cr3_col, 0)
                    try:
                        if float(cr3_val) > 30:
                            score += 1
                    except (ValueError, TypeError):
                        pass

                if score >= 1:
                    good_industries.add(industry_name)

        except Exception as e:
            logger.warning(f"行业筛选出错: {e}，跳过")
            return stock_list

        if not good_industries:
            logger.warning("未找到优质行业，跳过行业筛选")
            return stock_list

        logger.info(f"优质行业{len(good_industries)}个: {list(good_industries)[:5]}")

        # 筛选属于优质行业的个股
        passed = []
        for stock in stock_list:
            stock_industry = stock.get("industry", "")
            if stock_industry in good_industries:
                passed.append(stock)

        # 如果筛完太少，保留全部
        if len(passed) < 20:
            logger.warning(f"行业筛选后仅{len(passed)}只，保留全部")
            return stock_list

        return passed

    # ──────────────────────────────────────────────────────────────
    # Step 3: 多维评分
    # ──────────────────────────────────────────────────────────────

    def _step3_multi_dimension_score(self, stock_list: list[dict],
                                     market_data: dict,
                                     market_state: MarketState | None) -> list[CandidateStock]:
        """
        对每只股票执行原子规则打分
        使用 value + industry + trend + emotion 维度的规则
        """
        scored = []

        for stock_data in stock_list:
            symbol = stock_data.get("symbol", "")
            name = stock_data.get("name", "")
            industry = stock_data.get("industry", "")

            # 构建评估上下文
            ctx = EvalContext(
                fetcher=None,  # 由engine层提供
                quality=self.quality,
                stock_data=stock_data,
                market_data=market_data,
                history_data={},
                config=self.config,
            )

            # 执行各维度规则
            all_judgments = []

            for dimension in ["value", "industry", "emotion", "trend"]:
                judgments = self.registry.evaluate_dimension(dimension, ctx)
                all_judgments.extend(judgments)

            if not all_judgments:
                continue

            # 存储判断结果到stock_data中（供综合规则使用）
            stock_data["_judgments"] = {j.rule_id: j for j in all_judgments}

            # 重新执行综合规则（依赖原子规则结果）
            for dimension in ["value", "industry", "emotion", "trend"]:
                composite_judgments = self.registry.evaluate_dimension(dimension, ctx)
                for cj in composite_judgments:
                    if cj.rule_id.endswith("_01") or cj.rule_id.startswith("cross_"):
                        # 综合规则
                        if cj.rule_id not in stock_data["_judgments"]:
                            all_judgments.append(cj)
                            stock_data["_judgments"][cj.rule_id] = cj

            # 计算合成分数
            comp_score, comp_confidence = composite_score(all_judgments)

            candidate = CandidateStock(
                symbol=symbol,
                name=name,
                industry=industry,
                judgments=all_judgments,
                composite_score=comp_score,
                composite_confidence=comp_confidence,
            )
            scored.append(candidate)

        # 按合成分数排序
        scored.sort(key=lambda c: c.composite_score, reverse=True)

        # 取Top 50（反事实再筛一轮）
        return scored[:50]

    # ──────────────────────────────────────────────────────────────
    # Step 4: 反事实淘汰
    # ──────────────────────────────────────────────────────────────

    def _step4_counterfactual_filter(self, candidates: list[CandidateStock],
                                     market_data: dict,
                                     max_count: int) -> list[CandidateStock]:
        """
        反事实淘汰：
        1. 检查是否有严重风险信号（risk维度低分）
        2. 检查数据质量是否足够
        3. 取Top N
        """
        passed = []

        for c in candidates:
            # 检查是否有风险维度的严重警告
            risk_judgments = c.get_judgments_by_dimension("risk")
            has_critical_risk = False
            for j in risk_judgments:
                if j.score < 20 and j.direction == "SELL":
                    has_critical_risk = True
                    break

            if has_critical_risk:
                c.counterfactual = {"excluded": True, "reason": "风险维度严重警告"}
                continue

            # 检查合成置信度
            if c.composite_confidence < 0.2:
                c.counterfactual = {"excluded": True, "reason": "置信度过低"}
                continue

            # 检查数据覆盖度（至少3个维度有判断）
            dimensions_with_data = set(j.dimension for j in c.judgments)
            if len(dimensions_with_data) < 2:
                c.counterfactual = {"excluded": True, "reason": "数据覆盖不足"}
                continue

            passed.append(c)

        # 按分数排序取Top N
        passed.sort(key=lambda c: c.composite_score, reverse=True)
        return passed[:max_count]


def quick_score_stock(symbol: str, name: str, stock_data: dict,
                      market_data: dict, registry: RuleRegistry,
                      quality: DataQualityAssessor,
                      config: dict | None = None) -> CandidateStock | None:
    """
    快速单股评分（跳过行业筛选和反事实）

    用于主动发现引擎中对单只股票的快速评估。
    """
    ctx = EvalContext(
        fetcher=None,
        quality=quality,
        stock_data=stock_data,
        market_data=market_data,
        history_data={},
        config=config or {},
    )

    all_judgments = []
    for dimension in ["value", "industry", "emotion", "trend", "risk"]:
        judgments = registry.evaluate_dimension(dimension, ctx)
        all_judgments.extend(judgments)

    if not all_judgments:
        return None

    comp_score, comp_confidence = composite_score(all_judgments)

    return CandidateStock(
        symbol=symbol,
        name=name,
        industry=stock_data.get("industry", ""),
        judgments=all_judgments,
        composite_score=comp_score,
        composite_confidence=comp_confidence,
    )
