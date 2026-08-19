"""
正方形系统 v2.3 — 每日运行脚本（带实时股价）

输出：
- data/reports/LATEST.md — 最新报告（含实时股价）
- data/reports/YYYY-MM-DD.md — 历史报告存档
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daily-runner")


def fetch_prices(candidates: list[dict]) -> dict[str, dict]:
    """获取候选标的的实时股价

    Returns:
        {symbol: {"price": float, "change_pct": float, "pe_ttm": float, "pb": float}}
    """
    if not candidates:
        return {}

    codes = [c.get("symbol", "") for c in candidates if c.get("symbol")]
    if not codes:
        return {}

    try:
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher()
        df = fetcher.get_valuation(codes)
        if df is None or df.empty:
            logger.warning("获取股价失败: 返回空")
            return {}

        prices = {}
        for _, row in df.iterrows():
            symbol = str(row.get("symbol", ""))
            prices[symbol] = {
                "price": float(row.get("price", 0)),
                "change_pct": float(row.get("change_pct", 0)),
                "pe_ttm": float(row.get("pe_ttm", 0)),
                "pb": float(row.get("pb", 0)),
                "mcap_yi": float(row.get("mcap_yi", 0)),
            }
        logger.info(f"获取 {len(prices)} 只标的实时股价")
        return prices
    except Exception as e:
        logger.warning(f"获取股价失败: {e}")
        return {}


def inject_prices_to_report(report: str, candidates: list[dict],
                             price_map: dict[str, dict]) -> str:
    """在候选池表格的代码列后插入实时股价"""
    if not price_map:
        return report

    lines = report.split("\n")
    in_candidate_table = False
    new_lines = []

    for line in lines:
        # 检测候选池表格表头
        if line.startswith("| 代码 | 名称"):
            in_candidate_table = True
            # 在原表头后加一列"现价"
            new_lines.append("| 代码 | 名称 | 现价 | 商业质量 | 估值安全 | 趋势 | 情绪 | 综合分 | 置信度 |")
            continue
        if in_candidate_table and line.startswith("|------"):
            new_lines.append("|------|------|------|---------|---------|------|------|--------|--------|")
            continue

        # 检测表格行
        if in_candidate_table and line.startswith("| "):
            parts = line.split("|")
            if len(parts) >= 8:
                symbol = parts[1].strip()
                name = parts[2].strip()
                value_s = parts[3].strip()
                industry_s = parts[4].strip()
                trend_s = parts[5].strip()
                emotion_s = parts[6].strip()
                comp_score = parts[7].strip()
                comp_conf = parts[8].strip() if len(parts) > 8 else ""

                # 插入股价列
                if symbol in price_map:
                    p = price_map[symbol]
                    price_str = f"{p['price']:.2f}"
                    if p['change_pct'] > 0:
                        price_str += f" 📈+{p['change_pct']:.2f}%"
                    elif p['change_pct'] < 0:
                        price_str += f" 📉{p['change_pct']:.2f}%"
                    else:
                        price_str += f" {p['change_pct']:.2f}%"
                else:
                    price_str = "—"

                new_lines.append(f"| {symbol} | {name} | {price_str} | {value_s} | {industry_s} | "
                                 f"{trend_s} | {emotion_s} | {comp_score} | {comp_conf} |")
                continue
            in_candidate_table = False

        new_lines.append(line)

    return "\n".join(new_lines)


def run_daily():
    """每日运行入口"""
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== 每日运行 {today} ===")

    from src.core.engine import DecisionEngine
    from src.output.report import ReportGenerator

    # 运行引擎
    engine = DecisionEngine()
    result = engine.run()

    # 获取实时股价
    candidates = result.get("candidates", [])
    price_map = fetch_prices(candidates)

    # 生成基础报告
    reporter = ReportGenerator()
    report = reporter.generate(result)

    # 注入股价到候选池表格
    if price_map:
        report = inject_prices_to_report(report, candidates, price_map)
        logger.info(f"已注入 {len(price_map)} 只股价到报告")
    else:
        logger.warning("无股价数据，报告不含实时价格")

    # 输出路径
    reports_dir = PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    latest_path = reports_dir / "LATEST.md"
    archive_path = reports_dir / f"{today}.md"

    # 写入 LATEST.md
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"已更新: {latest_path}")

    # 存档
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"已存档: {archive_path}")

    # 输出摘要
    ms = result.get("market_state", {})
    discoveries = result.get("discoveries", [])

    logger.info(f"市场: {ms.get('regime', '?')} 温度{ms.get('temperature', 0):.0f} "
                f"百分位{ms.get('temperature_percentile', 0):.0f}%")
    logger.info(f"主动发现: {len(discoveries)}条")
    logger.info(f"候选池: {len(candidates)}只")
    logger.info(f"=== 每日运行完成 ===")

    return report


if __name__ == "__main__":
    run_daily()
