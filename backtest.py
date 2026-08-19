"""
正方形系统 v2.2 回测脚本

回测逻辑：
1. 用akshare获取沪深300成分股过去2年的日线数据
2. 每月月初用规则引擎评分，选出Top5
3. 等权买入，持有到下月初换仓
4. 计算：年化收益、最大回撤、胜率、夏普比率
5. 与沪深300基准对比
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)


def fetch_index_daily(symbol: str = "sh000300",
                      start: str = "20240101") -> pd.DataFrame:
    """获取指数日线（基准）"""
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol=symbol)
    df = df.rename(columns={"date": "日期", "close": "收盘"})
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] >= start].copy()
    df = df.sort_values("日期").reset_index(drop=True)
    return df


def fetch_stock_daily(symbol: str, start: str = "20240101") -> pd.DataFrame | None:
    """获取个股日线（用新浪数据源）"""
    import akshare as ak
    try:
        # 转换代码格式：600519 → sh600519
        if symbol.startswith("6"):
            ak_symbol = f"sh{symbol}"
        else:
            ak_symbol = f"sz{symbol}"

        df = ak.stock_zh_a_daily(symbol=ak_symbol, start_date=start, adjust="qfq")
        if df is None or df.empty:
            return None
        df = df.rename(columns={"date": "日期", "close": "收盘", "open": "开盘",
                                "high": "最高", "low": "最低", "volume": "成交量"})
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)
        return df
    except Exception:
        return None


def get_sample_stocks() -> list[str]:
    """获取回测用的股票池（沪深300成分股的子集）"""
    import akshare as ak
    try:
        df = ak.index_stock_cons(symbol="000300")
        codes = df["品种代码"].tolist()
        return codes[:50]  # 取前50只降低回测时间
    except Exception:
        # fallback: 用一些常见大盘股
        return ["600519", "000858", "601318", "600036", "000333",
                "600276", "601166", "000001", "600000", "601398",
                "600900", "000651", "601888", "300750", "002475",
                "600887", "601012", "600309", "000568", "002304"]


def score_stock_simple(kline: pd.DataFrame) -> float | None:
    """
    简化评分：基于技术指标的快速打分

    完整规则引擎需要太多数据（季报/F10等），回测用简化版：
    - 趋势分：MA5>MA20 + 价格在MA20上方
    - 动量分：近5日涨幅
    - 量价分：成交量放大
    - 波动分：波动率适中（不过大也不过小）
    """
    if kline is None or len(kline) < 30:
        return None

    close = kline["收盘"].values
    volume = kline["成交量"].values if "成交量" in kline.columns else np.ones(len(close))

    score = 50.0  # 基准分

    # 趋势：MA5 vs MA20
    ma5 = np.mean(close[-5:])
    ma20 = np.mean(close[-20:])
    if ma5 > ma20:
        score += 15  # 短期均线在上
    else:
        score -= 10

    # 价格位置：当前价 vs MA20
    current = close[-1]
    if current > ma20:
        score += 10
    else:
        score -= 5

    # 动量：近5日涨幅
    ret5 = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
    if ret5 > 3:
        score += 10
    elif ret5 > 0:
        score += 5
    elif ret5 < -3:
        score -= 10

    # 量价配合：近5日成交量 vs 20日均量
    if len(volume) >= 20:
        vol5 = np.mean(volume[-5:])
        vol20 = np.mean(volume[-20:])
        if vol20 > 0 and vol5 / vol20 > 1.3:
            score += 5  # 放量

    # 波动率：20日标准差
    if len(close) >= 20:
        returns = np.diff(np.log(close[-20:]))
        vol = np.std(returns) * np.sqrt(252) * 100
        if vol > 50:
            score -= 10  # 波动太大
        elif vol < 15:
            score += 5   # 波动适中偏小

    return max(0, min(100, score))


def run_backtest(start_date: str = "20240101",
                 top_n: int = 5) -> dict:
    """
    执行回测

    Returns:
        dict: 回测结果
    """
    logger.info("=== 正方形系统 v2.2 回测 ===")
    logger.info(f"回测起始: {start_date}, 每月选{top_n}只")

    # 1. 获取股票池
    logger.info("获取股票池...")
    stock_codes = get_sample_stocks()
    logger.info(f"股票池: {len(stock_codes)}只")

    # 2. 批量获取历史数据
    logger.info("获取历史数据（可能需要几分钟）...")
    stock_data: dict[str, pd.DataFrame] = {}
    for i, code in enumerate(stock_codes):
        df = fetch_stock_daily(code, start=start_date)
        if df is not None and len(df) >= 60:
            stock_data[code] = df
        if (i + 1) % 10 == 0:
            logger.info(f"  已获取 {i+1}/{len(stock_codes)}")

    logger.info(f"有效股票: {len(stock_data)}只")

    if len(stock_data) < 10:
        logger.error("数据不足，无法回测")
        return {}

    # 3. 获取基准（沪深300）
    logger.info("获取沪深300基准...")
    benchmark = fetch_index_daily(start=start_date)

    # 4. 按月回测
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df["日期"].tolist())
    all_dates = sorted(all_dates)

    if not all_dates:
        logger.error("无有效交易日")
        return {}

    # 生成每月第一个交易日
    date_series = pd.Series(all_dates)
    monthly_dates = date_series.groupby([date_series.dt.year, date_series.dt.month]).first().tolist()

    # 限制回测范围
    start_dt = pd.Timestamp(start_date)
    monthly_dates = [d for d in monthly_dates if d >= start_dt]

    if len(monthly_dates) < 3:
        logger.error("回测期间太短")
        return {}

    logger.info(f"回测期间: {monthly_dates[0].strftime('%Y-%m-%d')} ~ {monthly_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"换仓次数: {len(monthly_dates) - 1}")

    # 5. 逐月评分选股
    portfolio_returns = []
    monthly_picks = []
    win_months = 0
    total_months = 0

    for i in range(len(monthly_dates) - 1):
        rebalance_date = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        # 对每只股票评分
        scores = {}
        for code, df in stock_data.items():
            # 截取到换仓日的数据
            mask = df["日期"] <= rebalance_date
            df_slice = df[mask]
            s = score_stock_simple(df_slice)
            if s is not None:
                scores[code] = s

        if not scores:
            continue

        # 选Top N
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_codes = [code for code, _ in sorted_stocks[:top_n]]
        top_scores = {code: s for code, s in sorted_stocks[:top_n]}

        # 计算持有期收益
        period_returns = []
        for code in top_codes:
            df = stock_data[code]
            buy_row = df[df["日期"] >= rebalance_date].head(1)
            sell_row = df[df["日期"] >= next_date].head(1)

            if buy_row.empty or sell_row.empty:
                continue

            buy_price = float(buy_row["收盘"].iloc[0])
            sell_price = float(sell_row["收盘"].iloc[0])

            if buy_price > 0:
                ret = (sell_price / buy_price - 1)
                period_returns.append(ret)

        if period_returns:
            avg_ret = np.mean(period_returns)
            portfolio_returns.append(avg_ret)
            monthly_picks.append({
                "date": rebalance_date.strftime("%Y-%m"),
                "stocks": top_codes,
                "scores": [f"{top_codes[j]}({top_scores[top_codes[j]]:.0f})" for j in range(len(top_codes))],
                "return": f"{avg_ret:.2%}",
            })
            total_months += 1
            if avg_ret > 0:
                win_months += 1

    if not portfolio_returns:
        logger.error("无有效回测数据")
        return {}

    # 6. 计算基准收益
    benchmark_returns = []
    for i in range(len(monthly_dates) - 1):
        rebalance_date = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        bm_buy = benchmark[benchmark["日期"] >= rebalance_date].head(1)
        bm_sell = benchmark[benchmark["日期"] >= next_date].head(1)

        if not bm_buy.empty and not bm_sell.empty:
            buy_p = float(bm_buy["收盘"].iloc[0])
            sell_p = float(bm_sell["收盘"].iloc[0])
            if buy_p > 0:
                benchmark_returns.append(sell_p / buy_p - 1)

    # 7. 统计
    portfolio_arr = np.array(portfolio_returns)
    benchmark_arr = np.array(benchmark_returns[:len(portfolio_returns)])

    cum_port = np.cumprod(1 + portfolio_arr)
    cum_bench = np.cumprod(1 + benchmark_arr) if len(benchmark_arr) > 0 else np.ones_like(cum_port)

    total_return = cum_port[-1] - 1
    bench_total = cum_bench[-1] - 1 if len(cum_bench) > 0 else 0
    excess = total_return - bench_total

    # 年化
    n_months = len(portfolio_returns)
    annual_return = (1 + total_return) ** (12 / n_months) - 1 if n_months > 0 else 0
    bench_annual = (1 + bench_total) ** (12 / n_months) - 1 if n_months > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(cum_port)
    drawdown = (cum_port - peak) / peak
    max_drawdown = float(np.min(drawdown))

    # 胜率
    win_rate = win_months / total_months if total_months > 0 else 0

    # 夏普（月度）
    if np.std(portfolio_returns) > 0:
        sharpe = float(np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(12))
    else:
        sharpe = 0

    result = {
        "total_return": total_return,
        "benchmark_return": bench_total,
        "excess_return": excess,
        "annual_return": annual_return,
        "benchmark_annual": bench_annual,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "n_months": n_months,
        "monthly_picks": monthly_picks,
    }

    return result


def print_result(result: dict):
    """打印回测结果"""
    print("\n" + "=" * 50)
    print("正方形系统 v2.2 回测结果")
    print("=" * 50)
    print(f"回测月数: {result['n_months']}")
    print(f"总收益:   {result['total_return']:+.2%}")
    print(f"基准收益: {result['benchmark_return']:+.2%}")
    print(f"超额收益: {result['excess_return']:+.2%}")
    print(f"年化收益: {result['annual_return']:+.2%}")
    print(f"基准年化: {result['benchmark_annual']:+.2%}")
    print(f"最大回撤: {result['max_drawdown']:.2%}")
    print(f"月胜率:   {result['win_rate']:.0%}")
    print(f"夏普比率: {result['sharpe']:.2f}")
    print()

    print("每月选股:")
    print("-" * 50)
    for pick in result.get("monthly_picks", []):
        stocks_str = ", ".join(pick["scores"])
        print(f"  {pick['date']}: {stocks_str} → {pick['return']}")

    print()
    if result['excess_return'] > 0:
        print(f"结论: 跑赢基准 {result['excess_return']:+.2%}，系统有正期望值")
    else:
        print(f"结论: 跑输基准 {result['excess_return']:+.2%}，规则阈值需要调参")


if __name__ == "__main__":
    result = run_backtest(start_date="20240101", top_n=5)
    if result:
        print_result(result)
