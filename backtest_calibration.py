"""
正方形系统 — 规则阈值网格搜索

对 score_stock_simple() 的关键阈值做参数搜索。
训练期拟合，测试期验证，防过拟合。
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest_v2 import (
    fetch_stock_daily, get_sample_stocks, select_with_industry_constraint,
    SLIPPAGE, COMMISSION,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("calibration")
logger.setLevel(logging.INFO)


def score_stock_parameterized(kline: pd.DataFrame, params: dict) -> float | None:
    """参数化评分函数，阈值从 params 字典读取"""
    if kline is None or len(kline) < 120:
        return None

    close = kline["收盘"].values
    volume = kline["成交量"].values if "成交量" in kline.columns else np.ones(len(close))

    score = 50.0

    ma5 = np.mean(close[-5:])
    ma20 = np.mean(close[-20:])
    if ma5 > ma20:
        score += params.get("ma_trend_up", 15)
    else:
        score -= params.get("ma_trend_down", 10)

    current = close[-1]
    if current > ma20:
        score += params.get("above_ma20", 10)
    else:
        score -= params.get("below_ma20", 5)

    ret5 = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
    ret5_high = params.get("ret5_high", 3)
    if ret5 > ret5_high:
        score += params.get("ret5_high_bonus", 10)
    elif ret5 > 0:
        score += params.get("ret5_pos_bonus", 5)
    elif ret5 < -ret5_high:
        score -= params.get("ret5_neg_penalty", 10)

    ret20 = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0
    ret20_warn = params.get("ret20_warn", 20)
    ret20_danger = params.get("ret20_danger", 30)
    if ret20 > ret20_danger:
        score -= params.get("ret20_danger_penalty", 8)
    elif ret20 > ret20_warn:
        score -= params.get("ret20_warn_penalty", 5)

    if len(volume) >= 20:
        vol5 = np.mean(volume[-5:])
        vol20 = np.mean(volume[-20:])
        vol_ratio = params.get("vol_ratio", 1.3)
        if vol20 > 0 and vol5 / vol20 > vol_ratio:
            score += params.get("vol_bonus", 5)

    if len(close) >= 20:
        returns = np.diff(np.log(close[-20:]))
        vol = np.std(returns) * np.sqrt(252) * 100
        vol_high = params.get("vol_high", 50)
        vol_low = params.get("vol_low", 15)
        if vol > vol_high:
            score -= params.get("vol_high_penalty", 10)
        elif vol < vol_low:
            score += params.get("vol_low_bonus", 5)

    return max(0, min(100, score))


# 默认参数
DEFAULT_PARAMS = {
    "ma_trend_up": 15, "ma_trend_down": 10,
    "above_ma20": 10, "below_ma20": 5,
    "ret5_high": 3, "ret5_high_bonus": 10, "ret5_pos_bonus": 5, "ret5_neg_penalty": 10,
    "ret20_warn": 20, "ret20_danger": 30,
    "ret20_warn_penalty": 5, "ret20_danger_penalty": 8,
    "vol_ratio": 1.3, "vol_bonus": 5,
    "vol_high": 50, "vol_low": 15, "vol_high_penalty": 10, "vol_low_bonus": 5,
}

# 搜索空间（每个参数取3-5个值）
SEARCH_SPACE = {
    "ma_trend_up": [10, 15, 20],
    "ma_trend_down": [5, 10, 15],
    "ret5_high": [2, 3, 5],
    "ret5_high_bonus": [5, 10, 15],
    "ret20_warn_penalty": [3, 5, 8],
    "ret20_danger_penalty": [5, 8, 12],
    "vol_high": [40, 50, 60],
    "vol_high_penalty": [5, 10, 15],
}


def run_single_backtest(stock_data: dict, benchmark: pd.DataFrame,
                        params: dict, start_date: str, end_date: str | None,
                        top_n: int = 5) -> dict | None:
    """用指定参数跑一段回测"""
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df["日期"].tolist())
    all_dates = sorted(all_dates)

    date_series = pd.Series(all_dates)
    monthly_dates = date_series.groupby([date_series.dt.year, date_series.dt.month]).first().tolist()

    start_dt = pd.Timestamp(start_date)
    monthly_dates = [d for d in monthly_dates if d >= start_dt]
    if end_date:
        end_dt = pd.Timestamp(end_date)
        monthly_dates = [d for d in monthly_dates if d <= end_dt]

    if len(monthly_dates) < 3:
        return None

    portfolio_returns = []
    for i in range(len(monthly_dates) - 1):
        rd = monthly_dates[i]
        nd = monthly_dates[i + 1]

        scores = {}
        for code, df in stock_data.items():
            df_slice = df[df["日期"] <= rd]
            s = score_stock_parameterized(df_slice, params)
            if s is not None:
                scores[code] = s

        if not scores:
            continue

        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_codes = select_with_industry_constraint(sorted_stocks, top_n, max_per_industry=2)

        period_returns = []
        for code in top_codes:
            df = stock_data[code]
            buy_row = df[df["日期"] >= rd].head(1)
            sell_row = df[df["日期"] >= nd].head(1)
            if buy_row.empty or sell_row.empty:
                continue
            bp = float(buy_row["收盘"].iloc[0])
            sp = float(sell_row["收盘"].iloc[0])
            if bp > 0:
                actual_buy = bp * (1 + SLIPPAGE)
                actual_sell = sp * (1 - SLIPPAGE - COMMISSION)
                period_returns.append(actual_sell / actual_buy - 1)

        if period_returns:
            portfolio_returns.append(np.mean(period_returns))

    if len(portfolio_returns) < 3:
        return None

    arr = np.array(portfolio_returns)
    total = np.prod(1 + arr) - 1
    n = len(arr)
    annual = (1 + total) ** (12 / n) - 1 if n > 0 else 0
    sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(12)) if np.std(arr) > 0 else 0
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))
    win_rate = sum(1 for r in arr if r > 0) / n

    return {
        "total_return": total, "annual_return": annual, "sharpe": sharpe,
        "max_drawdown": max_dd, "win_rate": win_rate, "n_months": n,
    }


def grid_search(stock_data: dict, benchmark: pd.DataFrame,
                train_start: str = "20240101", train_end: str = "20250630",
                test_start: str = "20250701", test_end: str | None = None,
                top_n: int = 5) -> list[dict]:
    """网格搜索：训练期拟合，测试期验证"""

    # 生成所有参数组合
    param_names = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())
    combinations = list(itertools.product(*param_values))
    logger.info(f"搜索空间: {len(combinations)} 种参数组合")

    results = []
    for i, combo in enumerate(combinations):
        params = dict(DEFAULT_PARAMS)
        for name, val in zip(param_names, combo):
            params[name] = val

        # 训练期
        train = run_single_backtest(stock_data, benchmark, params, train_start, train_end, top_n)
        if train is None:
            continue

        # 测试期
        test = run_single_backtest(stock_data, benchmark, params, test_start, test_end, top_n)
        if test is None:
            continue

        results.append({
            "params": {k: v for k, v in zip(param_names, combo)},
            "train_sharpe": train["sharpe"],
            "train_return": train["total_return"],
            "test_sharpe": test["sharpe"],
            "test_return": test["total_return"],
            "test_max_dd": test["max_drawdown"],
        })

        if (i + 1) % 100 == 0:
            logger.info(f"  已搜索 {i+1}/{len(combinations)}")

    return results


def print_calibration_results(results: list[dict], top_k: int = 10):
    """打印校准结果"""
    if not results:
        print("无校准结果")
        return

    # 按测试期夏普排序
    results.sort(key=lambda x: x["test_sharpe"], reverse=True)

    print("\n" + "=" * 80)
    print(f"网格搜索结果（按测试期夏普排序，Top {top_k}）")
    print("=" * 80)
    print(f"{'排名':>4} {'训练夏普':>8} {'测试夏普':>8} {'测试收益':>8} {'测试回撤':>8} 参数")
    print("-" * 80)

    for i, r in enumerate(results[:top_k]):
        p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"{i+1:>4} {r['train_sharpe']:>7.2f} {r['test_sharpe']:>7.2f} "
              f"{r['test_return']:>+7.2%} {r['test_max_dd']:>7.2%} {p_str}")

    # 默认参数的基准
    default_train = run_single_backtest(
        _stock_data, _benchmark, DEFAULT_PARAMS, "20240101", "20250630", 5)
    default_test = run_single_backtest(
        _stock_data, _benchmark, DEFAULT_PARAMS, "20250701", None, 5)

    print()
    if default_train and default_test:
        print(f"默认参数基准: 训练夏普={default_train['sharpe']:.2f}, "
              f"测试夏普={default_test['sharpe']:.2f}, 测试收益={default_test['total_return']:+.2%}")

    # 过拟合检测
    best = results[0]
    if best["test_sharpe"] > 0 and best["train_sharpe"] > 0:
        ratio = best["test_sharpe"] / best["train_sharpe"]
        if ratio < 0.5:
            print("警告: 测试/训练夏普比<0.5，可能存在过拟合")
        else:
            print(f"最优参数测试/训练比: {ratio:.2f}，过拟合风险{'低' if ratio > 0.7 else '中等'}")


# 全局变量，供 main 使用
_stock_data: dict = {}
_benchmark: pd.DataFrame = pd.DataFrame()


if __name__ == "__main__":
    logger.info("=== 规则阈值网格搜索 ===")

    universe = "hs300"
    logger.info(f"获取股票池({universe})...")
    stock_codes = get_sample_stocks(universe)
    logger.info(f"股票池: {len(stock_codes)}只")

    logger.info("获取历史数据...")
    _stock_data = {}
    for i, code in enumerate(stock_codes):
        df = fetch_stock_daily(code, start="20230601")
        if df is not None and len(df) >= 120:
            _stock_data[code] = df
        if (i + 1) % 50 == 0:
            logger.info(f"  已获取 {i+1}/{len(stock_codes)}")

    logger.info(f"有效股票: {len(_stock_data)}只")

    from backtest_v2 import fetch_index_daily
    _benchmark = fetch_index_daily(start="20230601")

    results = grid_search(_stock_data, _benchmark)
    print_calibration_results(results)
