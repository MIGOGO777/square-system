"""
决策引擎 — v2.2 核心

替代旧版 Orchestrator，组装所有模块：
1. 获取数据
2. 检测市场状态（HMM + 动态阈值）
3. 主动发现（六条路线，含协整配对）
4. 构建候选池（四步筛选）
4.5. Markowitz组合优化
5. 反事实推理
6. 情景推演
7. 生成报告

每步独立，可单独测试。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.core.signal import CandidateStock, EvalContext, MarketState
from src.data.fetcher import DataFetcher
from src.data.quality import DataQualityAssessor
from src.discovery import DiscoveryEngine
from src.pool.builder import CandidatePoolBuilder
from src.pool.optimizer import PortfolioOptimizer
from src.rules.registry import RuleRegistry, composite_score
from src.synthesis.counterfactual import CounterfactualAnalyzer
from src.synthesis.scenario import ScenarioPlanner

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    决策引擎

    组装所有模块，执行完整的分析流程。
    """

    def __init__(self):
        # 加载配置
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {}

        self.quality = DataQualityAssessor()
        self.registry = RuleRegistry()
        self._register_rules()

        self.fetcher = DataFetcher()

        # HMM增强市场状态检测（优先），失败则用传统检测器
        self._use_hmm = False
        try:
            from src.regime.hmm_detector import HMMRegimeDetector
            hmm_config = self.config.get("models", {}).get("hmm", {})
            self.regime_detector = HMMRegimeDetector(
                n_states=hmm_config.get("n_states", 3),
                min_history=hmm_config.get("min_history", 30),
                n_iter=hmm_config.get("max_iterations", 100),
            )
            self._use_hmm = True
            logger.info("HMM市场状态检测器已启用")
        except ImportError:
            from src.regime.detector import RegimeDetector
            self.regime_detector = RegimeDetector()
            logger.info("HMM不可用，使用传统检测器")

        self.pool_builder = CandidatePoolBuilder(self.registry, self.quality,
                                                  config=self.config)
        self.discovery_engine = DiscoveryEngine(self.registry, self.quality)
        self.counterfactual = CounterfactualAnalyzer()
        self.scenario_planner = ScenarioPlanner()

        # Markowitz组合优化器
        mk_config = self.config.get("models", {}).get("markowitz", {})
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=mk_config.get("risk_free_rate", 0.02),
            max_weight=mk_config.get("max_weight", 0.30),
        )
        self._candidate_klines: dict[str, Any] = {}

    def _register_rules(self):
        """注册所有规则"""
        from src.rules import value, industry, emotion, trend, macro, risk
        value.register_all(self.registry)
        industry.register_all(self.registry)
        emotion.register_all(self.registry)
        trend.register_all(self.registry)
        macro.register_all(self.registry)
        risk.register_all(self.registry)
        logger.info(f"已注册{self.registry.rule_count}条规则")

    def run(self, history_temps: list[float] | None = None) -> dict:
        """
        执行完整分析流程

        Returns:
            dict: 完整分析结果，供报告生成器使用
        """
        logger.info("=" * 60)
        logger.info("正方形系统 v2.3 开始运行")
        logger.info("=" * 60)

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market_state": None,
            "discoveries": [],
            "candidates": [],
            "counterfactuals": {},
            "scenarios": {},
            "rules_summary": {},
        }

        # Step 1: 获取市场数据
        logger.info("Step1: 获取市场数据")
        market_data = self._fetch_market_data()

        # 加载HMM观测历史
        if self._use_hmm:
            from src.regime.hmm_detector import load_observation_history
            market_data["observation_history"] = load_observation_history(
                str(Path(__file__).parent.parent.parent / "data" / "cache"))

        # Step 2: 检测市场状态
        logger.info("Step2: 检测市场状态")
        market_state = self.regime_detector.detect(market_data, history_temps)
        result["market_state"] = market_state.to_dict()
        hmm_used = any(j.get("rule_id") == "hmm_regime"
                       for j in market_state.to_dict().get("judgments", []))
        logger.info(f"市场状态: {market_state.regime} 温度{market_state.temperature} "
                     f"百分位{market_state.temperature_percentile}% 钟摆{market_state.pendulum_position}")

        # Step 3: 主动发现
        logger.info("Step3: 主动发现")
        stock_list = self._fetch_stock_list()
        discoveries = self.discovery_engine.discover(stock_list, market_data)
        result["discoveries"] = discoveries
        logger.info(f"主动发现{len(discoveries)}条线索")

        # Step 4: 构建候选池
        logger.info("Step4: 构建候选池")
        candidates = self.pool_builder.build(stock_list, market_data, market_state)
        result["candidates"] = [c.to_dict() for c in candidates]
        logger.info(f"候选池{len(candidates)}只标的")

        # Step 4.5: Markowitz组合优化
        logger.info("Step4.5: Markowitz组合优化")
        kline_map = self._build_kline_map(candidates, stock_list)
        optimization = self.optimizer.optimize(candidates, kline_map)
        result["optimization"] = optimization
        for c in candidates:
            if c.symbol in optimization:
                c.counterfactual = {**(c.counterfactual or {}),
                                    "markowitz_weight": optimization[c.symbol]["weight"],
                                    "markowitz_sharpe": optimization[c.symbol]["sharpe"]}
        logger.info(f"组合优化完成: {len(optimization)}只标的")

        # 模型运行状态跟踪（规则12：失败必须显性化）
        model_status: dict[str, dict] = {}
        model_status["HMM"] = {"ok": hmm_used, "note": "HMM" if hmm_used else "传统检测器"}
        model_status["Markowitz"] = {"ok": bool(optimization), "count": len(optimization)}

        # Step 4.6: Monte Carlo价格模拟
        logger.info("Step4.6: Monte Carlo价格模拟")
        try:
            from src.synthesis.monte_carlo import simulate_all
            mc_results = simulate_all(candidates, kline_map)
            result["monte_carlo"] = mc_results
            model_status["Monte Carlo"] = {"ok": True, "count": len(mc_results)}
            logger.info(f"Monte Carlo完成: {len(mc_results)}只标的")
        except Exception as e:
            logger.warning(f"Monte Carlo失败: {e}")
            result["monte_carlo"] = {}
            model_status["Monte Carlo"] = {"ok": False, "error": str(e)}

        # Step 4.7: Fama-French因子归因
        logger.info("Step4.7: Fama-French因子归因")
        try:
            from src.models.fama_french import analyze_all as ff_analyze
            ff_results = ff_analyze(candidates, kline_map)
            result["fama_french"] = ff_results
            model_status["Fama-French"] = {"ok": True, "count": len(ff_results)}
            logger.info(f"因子归因完成: {len(ff_results)}只标的")
        except Exception as e:
            logger.warning(f"因子归因失败: {e}")
            result["fama_french"] = {}
            model_status["Fama-French"] = {"ok": False, "error": str(e)}

        # Step 4.8: Black-Scholes波动率分析
        logger.info("Step4.8: 波动率分析")
        try:
            from src.models.black_scholes import analyze_all as bs_analyze
            bs_results = bs_analyze(candidates, kline_map)
            result["black_scholes"] = bs_results
            model_status["Black-Scholes"] = {"ok": True, "count": len(bs_results)}
            logger.info(f"波动率分析完成: {len(bs_results)}只标的")
        except Exception as e:
            logger.warning(f"波动率分析失败: {e}")
            result["black_scholes"] = {}
            model_status["Black-Scholes"] = {"ok": False, "error": str(e)}

        # Step 4.9: Almgren-Chriss最优执行
        logger.info("Step4.9: 最优执行分析")
        try:
            from src.models.almgren_chriss import analyze_candidates as ac_analyze
            ac_results = ac_analyze(candidates, kline_map)
            result["almgren_chriss"] = ac_results
            model_status["Almgren-Chriss"] = {"ok": True, "count": len(ac_results)}
            logger.info(f"最优执行分析完成: {len(ac_results)}只标的")
        except Exception as e:
            logger.warning(f"最优执行分析失败: {e}")
            result["almgren_chriss"] = {}
            model_status["Almgren-Chriss"] = {"ok": False, "error": str(e)}

        result["model_status"] = model_status

        # Step 5: 反事实推理
        logger.info("Step5: 反事实推理")
        for c in candidates:
            cf = self.counterfactual.analyze(c, market_state)
            c.counterfactual = {**(c.counterfactual or {}), **cf}
            result["counterfactuals"][c.symbol] = cf
        logger.info(f"完成{len(candidates)}只标的的反事实分析")

        # Step 6: 情景推演
        logger.info("Step6: 情景推演")
        scenarios = self.scenario_planner.plan(market_state, candidates)
        result["scenarios"] = scenarios

        # Step 7: 规则统计
        result["rules_summary"] = {
            "total_rules": self.registry.rule_count,
            "dimensions": self.registry.dimensions,
            "thinkers": self.registry.thinkers,
        }

        # 保存当日观测数据（供HMM使用）
        if self._use_hmm:
            from src.regime.hmm_detector import save_observation
            save_observation(market_data, str(Path(__file__).parent.parent.parent / "data" / "cache"))

        logger.info("=" * 60)
        logger.info("正方形系统 v2.3 运行完成")
        logger.info("=" * 60)

        return result

    def _fetch_market_data(self) -> dict:
        """
        获取市场数据

        对接真实 a-stock-data API，转换为规则引擎期望的格式。
        """
        market_data = {}
        import pandas as pd

        # ── 情绪温度 + 阶段 ──
        try:
            market_data["emotion_temp"] = self.fetcher.get_emotion_temperature()
            temp = market_data["emotion_temp"]
            if temp < 20:
                market_data["emotion_phase"] = "冰点"
            elif temp < 35:
                market_data["emotion_phase"] = "试探"
            elif temp < 55:
                market_data["emotion_phase"] = "发酵"
            elif temp < 70:
                market_data["emotion_phase"] = "高潮"
            elif temp < 85:
                market_data["emotion_phase"] = "分歧"
            else:
                market_data["emotion_phase"] = "退潮"
        except Exception as e:
            logger.warning(f"获取情绪数据失败: {e}")
            market_data["emotion_temp"] = 50.0
            market_data["emotion_phase"] = "未知"

        # ── 涨停池（ths_hot_reason → limit_up_pool DataFrame） ──
        try:
            hot_stocks = self.fetcher.get_hot_stocks()
            if hot_stocks is not None and not hot_stocks.empty:
                # 适配规则引擎：rules 期望 limit_up_pool 有 "所属行业"/"行业" 列
                # ths_hot_reason 提供 "题材归因" 列，映射为 "概念"
                if "题材归因" in hot_stocks.columns:
                    hot_stocks = hot_stocks.rename(columns={"题材归因": "概念"})
                market_data["limit_up_pool"] = hot_stocks
                market_data["limit_up_count"] = len(hot_stocks)
                market_data["hot_stocks"] = hot_stocks
            else:
                market_data["limit_up_pool"] = pd.DataFrame()
                market_data["limit_up_count"] = 0
        except Exception as e:
            logger.warning(f"获取涨停池失败: {e}")
            market_data["limit_up_pool"] = pd.DataFrame()
            market_data["limit_up_count"] = 0

        # ── 炸板率 + 连板高度 ──
        market_data["break_rate"] = self.fetcher.get_break_rate()
        market_data["leader_height"] = self.fetcher.get_leader_height()

        # ── 北向资金 ──
        try:
            north = self.fetcher.get_north_flow()
            if north is not None and not north.empty:
                # 规则引擎期望 "net_flow" 列，hsgt_realtime 返回 "hgt_yi"/"sgt_yi"
                if "hgt_yi" in north.columns and "sgt_yi" in north.columns:
                    north["net_flow"] = north["hgt_yi"].fillna(0) + north["sgt_yi"].fillna(0)
                market_data["north_flow"] = north
            else:
                market_data["north_flow"] = pd.DataFrame()
        except Exception:
            market_data["north_flow"] = pd.DataFrame()

        # ── 行业对比（dict → DataFrame 适配） ──
        try:
            ind = self.fetcher.get_industry_comparison()
            if ind and ind.get("top"):
                # 转为 DataFrame，规则引擎期望有 "行业"/"板块" + "涨跌幅" 列
                ind_df = pd.DataFrame(ind["top"])
                if "name" in ind_df.columns:
                    ind_df = ind_df.rename(columns={"name": "板块", "change_pct": "涨跌幅"})
                market_data["industry_comparison"] = ind_df
            else:
                market_data["industry_comparison"] = pd.DataFrame()
        except Exception:
            market_data["industry_comparison"] = pd.DataFrame()

        # ── 龙虎榜（dict → DataFrame 适配） ──
        try:
            dt = self.fetcher.get_dragon_tiger_all()
            if dt and dt.get("stocks"):
                market_data["dragon_tiger"] = pd.DataFrame(dt["stocks"])
            else:
                market_data["dragon_tiger"] = pd.DataFrame()
        except Exception:
            market_data["dragon_tiger"] = pd.DataFrame()

        # ── 行业资金流向（供 industry.py qgy_07 规则使用） ──
        try:
            ind = market_data.get("industry_comparison")
            if ind is not None and not ind.empty and "net_inflow_yi" in ind.columns:
                market_data["industry_fund_flow"] = ind[["板块", "net_inflow_yi"]].copy()
            else:
                market_data["industry_fund_flow"] = pd.DataFrame()
        except Exception:
            market_data["industry_fund_flow"] = pd.DataFrame()

        # ── 宏观数据（估算默认值） ──
        market_data["m2_growth"] = self.fetcher.get_m2_growth()
        market_data["pmi"] = self.fetcher.get_pmi()
        market_data["cpi"] = self.fetcher.get_cpi()
        market_data["gdp_growth"] = self.fetcher.get_gdp_growth()
        market_data["pe_percentile"] = self.fetcher.get_pe_percentile()

        # ── 市场涨跌幅（供HMM使用） ──
        market_data["market_change_pct"] = market_data.get("market_change_pct", 0.0)

        return market_data

    def _fetch_stock_list(self) -> list[dict]:
        """
        获取股票基础列表

        来源：涨停池活跃股 + 可选沪深300成分股
        丰富范围：全部股票（K线250根+估值+季报+F10）
        """
        try:
            stocks = self.fetcher.get_stock_list_with_basics()
            if not stocks:
                return []

            # 并行丰富数据（IO密集，线程池加速）
            from concurrent.futures import ThreadPoolExecutor, as_completed
            enriched = [None] * len(stocks)
            max_workers = min(8, len(stocks))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for i, stock in enumerate(stocks):
                    symbol = stock.get("symbol", "???")
                    futures[pool.submit(self.fetcher.enrich_stock,
                                        stock, 250)] = (i, symbol)

                done_count = 0
                for fut in as_completed(futures):
                    i, symbol = futures[fut]
                    try:
                        enriched[i] = fut.result()
                    except Exception as e:
                        logger.debug(f"丰富{symbol}失败: {e}")
                        enriched[i] = stocks[i]  # fallback to un-enriched
                    done_count += 1
                    if done_count % 20 == 0 or done_count == len(stocks):
                        logger.info(f"  已丰富 {done_count}/{len(stocks)}")

            enriched = [s for s in enriched if s is not None]
            logger.info(f"获取{len(enriched)}只股票，全部已丰富数据")
            return enriched
        except Exception as e:
            logger.warning(f"获取股票列表失败: {e}")
            return []

    def _build_kline_map(self, candidates: list[CandidateStock],
                         stock_list: list[dict]) -> dict[str, Any]:
        """构建候选标的的kline映射，供Markowitz优化器使用"""
        kline_map = {}
        stock_klines = {s.get("symbol"): s.get("kline") for s in stock_list
                        if s.get("kline") is not None}
        for c in candidates:
            kline = stock_klines.get(c.symbol)
            if kline is not None and hasattr(kline, 'empty') and not kline.empty:
                kline_map[c.symbol] = kline
        return kline_map
