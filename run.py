"""
正方形系统 v2.3 — 入口脚本

使用方式：
    python run.py              # 完整运行，输出报告
    python run.py --dry-run    # 测试模式，只检查模块
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("square-v2")


def main():
    parser = argparse.ArgumentParser(description="正方形系统 v2.3")
    parser.add_argument("--dry-run", action="store_true", help="测试模式")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== 测试模式 ===")
        _dry_run()
        return

    logger.info("=== 正方形系统 v2.3 启动 ===")

    from src.core.engine import DecisionEngine
    from src.output.report import ReportGenerator

    # 运行引擎
    engine = DecisionEngine()
    result = engine.run()

    # 生成报告
    reporter = ReportGenerator()
    report = reporter.generate(result)

    # 输出报告
    output_path = args.output or str(PROJECT_ROOT / "data" / "reports" / "LATEST.md")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"报告已输出到: {output_path}")
    logger.info("=== 正方形系统 v2.3 完成 ===")


def _dry_run():
    """测试模式：检查所有模块是否正常"""
    try:
        from src.core.signal import AtomicJudgment, CandidateStock, MarketState, EvalContext
        logger.info("✅ core.signal 模块正常")
    except Exception as e:
        logger.error(f"❌ core.signal 模块异常: {e}")

    try:
        from src.data.quality import DataQualityAssessor
        q = DataQualityAssessor()
        logger.info(f"✅ data.quality 模块正常 (REALTIME={q.REALTIME})")
    except Exception as e:
        logger.error(f"❌ data.quality 模块异常: {e}")

    try:
        from src.rules.registry import RuleRegistry
        reg = RuleRegistry()
        from src.rules import value, industry, emotion, trend, macro, risk
        value.register_all(reg)
        industry.register_all(reg)
        emotion.register_all(reg)
        trend.register_all(reg)
        macro.register_all(reg)
        risk.register_all(reg)
        logger.info(f"✅ rules 模块正常 ({reg.rule_count}条规则, 维度: {reg.dimensions})")
    except Exception as e:
        logger.error(f"❌ rules 模块异常: {e}")

    try:
        from src.regime.detector import RegimeDetector
        logger.info("✅ regime.detector 模块正常")
    except Exception as e:
        logger.error(f"❌ regime.detector 模块异常: {e}")

    try:
        from src.discovery import DiscoveryEngine
        logger.info("✅ discovery 模块正常")
    except Exception as e:
        logger.error(f"❌ discovery 模块异常: {e}")

    try:
        from src.synthesis.counterfactual import CounterfactualAnalyzer
        from src.synthesis.scenario import ScenarioPlanner
        logger.info("✅ synthesis 模块正常")
    except Exception as e:
        logger.error(f"❌ synthesis 模块异常: {e}")

    try:
        from src.output.report import ReportGenerator
        logger.info("✅ output.report 模块正常")
    except Exception as e:
        logger.error(f"❌ output.report 模块异常: {e}")

    # ── 数学模型依赖检查 ──
    logger.info("--- 数学模型依赖 ---")

    try:
        from arch import arch_model
        logger.info("✅ arch (GARCH) 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ arch (GARCH) 不可用: {e}")

    try:
        from hmmlearn.hmm import GaussianHMM
        logger.info("✅ hmmlearn (HMM) 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ hmmlearn (HMM) 不可用: {e}")

    try:
        from scipy.optimize import minimize
        from scipy import stats
        logger.info("✅ scipy 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ scipy 不可用: {e}")

    try:
        from statsmodels.tsa.stattools import adfuller
        logger.info("✅ statsmodels 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ statsmodels 不可用: {e}")

    # ── 新模块检查 ──
    logger.info("--- 新增模块 ---")

    try:
        from src.regime.hmm_detector import HMMRegimeDetector
        logger.info("✅ regime.hmm_detector 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ regime.hmm_detector 不可用: {e}")

    try:
        from src.pool.optimizer import PortfolioOptimizer
        logger.info("✅ pool.optimizer 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ pool.optimizer 不可用: {e}")

    try:
        from src.discovery.pairs_scanner import PairsScanner
        logger.info("✅ discovery.pairs_scanner 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ discovery.pairs_scanner 不可用: {e}")

    # ── 第二批模型检查 ──
    logger.info("--- 第二批模型 ---")

    try:
        from src.synthesis.monte_carlo import MonteCarloSimulator
        logger.info("✅ synthesis.monte_carlo 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ synthesis.monte_carlo 不可用: {e}")

    try:
        from src.models.fama_french import FamaFrenchModel
        logger.info("✅ models.fama_french 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ models.fama_french 不可用: {e}")

    try:
        from src.models.black_scholes import BlackScholesAnalyzer
        logger.info("✅ models.black_scholes 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ models.black_scholes 不可用: {e}")

    try:
        from src.models.almgren_chriss import AlmgrenChriss
        logger.info("✅ models.almgren_chriss 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ models.almgren_chriss 不可用: {e}")

    try:
        from src.models.rl_weight_adjuster import RLWeightAdjuster
        logger.info("✅ models.rl_weight_adjuster 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ models.rl_weight_adjuster 不可用: {e}")

    try:
        from src.models.causal_inference import CausalInferenceModel
        logger.info("✅ models.causal_inference 模块正常")
    except Exception as e:
        logger.warning(f"⚠️ models.causal_inference 不可用: {e}")

    logger.info("=== 测试完成 ===")


if __name__ == "__main__":
    main()
