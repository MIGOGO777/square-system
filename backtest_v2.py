"""
正方形系统 v2.2 回测脚本 — 优化版

4项核心优化（默认开启）：
1. 月度止损-8%（半月检查，作为保险机制）
2. 近20日涨幅>20%扣5分，>30%扣8分（防追高）
3. 120天预热期（确保技术指标可靠）
4. 行业分散约束（同行业最多2只）

可选优化：
5. Kelly公式（默认关闭，实测压制收益）
6. 波动率倒数加权（默认关闭，降低组合波动率）

可信度改进：
7. 交易成本模型（slippage + commission）
8. 扩大股票池（沪深300全部 / 中证500）
9. 样本外验证（train/test 时间分割）
10. 申万行业分类（替代代码前缀）
11. ATR 动态止损
12. 换仓频率可调
13. HMM 市场状态反馈到仓位
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)

# ── 交易成本参数 ──────────────────────────────────────────
SLIPPAGE = 0.001       # 滑点 0.1%
COMMISSION = 0.0005    # 佣金+印花税 0.05%

# ── HMM 市场状态仓位系数 ──────────────────────────────────
REGIME_POSITION_FACTOR = {
    "BULL": 1.0,
    "SIDEWAYS": 0.8,
    "BEAR": 0.5,
}

# ── 行业映射 ─────────────────────────────────────────────
# 改进2.1: 优先用申万行业缓存，fallback 到代码前缀
INDUSTRY_PREFIX = {
    "6000": "银行", "6010": "银行", "6011": "银行", "6012": "银行", "6013": "银行",
    "6001": "基建", "6002": "基建", "6003": "消费", "6004": "制造", "6005": "科技",
    "6006": "医药", "6007": "地产", "6008": "能源", "6009": "公用",
    "0000": "金融", "0001": "金融", "0002": "制造", "0003": "消费", "0004": "地产",
    "0005": "医药", "0006": "科技", "0007": "能源", "0008": "基建", "0009": "消费",
    "0020": "中小板", "0021": "中小板", "0022": "中小板", "0023": "中小板",
    "0024": "中小板", "0025": "中小板", "0026": "中小板", "0027": "中小板",
    "3000": "创业板", "3001": "创业板", "3002": "创业板", "3003": "创业板",
    "3004": "创业板", "3005": "创业板", "3006": "创业板", "3007": "创业板",
    "3008": "创业板", "3009": "创业板", "3010": "创业板", "3011": "创业板",
    "3012": "创业板", "6880": "科创板", "6881": "科创板", "6882": "科创板",
    "6883": "科创板", "6884": "科创板", "6885": "科创板",
}

_shenwan_cache: dict[str, str] | None = None


def fetch_shenwan_industry_map() -> dict[str, str]:
    """获取申万一级行业分类，缓存到本地 JSON"""
    global _shenwan_cache
    if _shenwan_cache is not None:
        return _shenwan_cache

    cache_file = CACHE_DIR / "industry_map.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                _shenwan_cache = json.load(f)
            logger.info(f"从缓存加载行业映射: {len(_shenwan_cache)}只")
            return _shenwan_cache
        except Exception:
            pass

    import akshare as ak
    industry_map: dict[str, str] = {}
    try:
        # 获取申万一级行业列表
        boards = ak.stock_board_industry_name_em()
        for _, row in boards.iterrows():
            board_name = row["板块名称"]
            try:
                cons = ak.stock_board_industry_cons_em(symbol=board_name)
                for _, stock_row in cons.iterrows():
                    code = str(stock_row["代码"]).zfill(6)
                    industry_map[code] = board_name
            except Exception:
                continue
        # 保存缓存
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(industry_map, f, ensure_ascii=False)
        _shenwan_cache = industry_map
        logger.info(f"申万行业映射已缓存: {len(industry_map)}只")
    except Exception as e:
        logger.warning(f"申万行业获取失败，使用前缀映射: {e}")
        _shenwan_cache = {}
    return _shenwan_cache


def get_industry(code: str) -> str:
    """获取行业分类：优先申万缓存，fallback 前缀映射"""
    global _shenwan_cache
    if _shenwan_cache is None:
        fetch_shenwan_industry_map()
    if _shenwan_cache and code in _shenwan_cache:
        return _shenwan_cache[code]
    prefix = code[:4]
    return INDUSTRY_PREFIX.get(prefix, "其他")


def fetch_index_daily(symbol: str = "sh000300",
                      start: str = "20230601") -> pd.DataFrame:
    """获取指数日线（基准）"""
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol=symbol)
    df = df.rename(columns={"date": "日期", "close": "收盘"})
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] >= start].copy()
    df = df.sort_values("日期").reset_index(drop=True)
    return df


def fetch_stock_daily(symbol: str, start: str = "20230601") -> pd.DataFrame | None:
    """获取个股日线（用新浪数据源）"""
    import akshare as ak
    try:
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


def get_sample_stocks(universe: str = "hs300") -> list[str]:
    """获取回测用的股票池

    Args:
        universe: "hs300"=沪深300全部 | "zz500"=中证500 | "hs300+zz500"=两者合并
    """
    import akshare as ak
    all_codes = []

    for idx_symbol in (["000300"] if universe == "hs300"
                       else ["000905"] if universe == "zz500"
                       else ["000300", "000905"]):
        try:
            df = ak.index_stock_cons(symbol=idx_symbol)
            codes = df["品种代码"].tolist()
            all_codes.extend(codes)
        except Exception as e:
            logger.warning(f"获取{idx_symbol}成分股失败: {e}")

    if all_codes:
        return list(dict.fromkeys(all_codes))  # 去重保序

    return ["600519", "000858", "601318", "600036", "000333",
            "600276", "601166", "000001", "600000", "601398",
            "600900", "000651", "601888", "300750", "002475",
            "600887", "601012", "600309", "000568", "002304"]


def score_stock_simple(kline: pd.DataFrame) -> float | None:
    """
    优化版评分

    改动：
    - 预热期从60天改为120天（优化4）
    - 新增：近20日涨幅>15%扣10分（优化2：防追高）
    """
    if kline is None or len(kline) < 120:  # 优化4: 120天预热
        return None

    close = kline["收盘"].values
    volume = kline["成交量"].values if "成交量" in kline.columns else np.ones(len(close))

    score = 50.0

    # 趋势：MA5 vs MA20
    ma5 = np.mean(close[-5:])
    ma20 = np.mean(close[-20:])
    if ma5 > ma20:
        score += 15
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

    # 优化2: 防追高——近20日涨幅>20%扣分（放宽阈值，避免砍掉动量股）
    ret20 = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0
    if ret20 > 30:
        score -= 8   # 涨太多，大概率回调
    elif ret20 > 20:
        score -= 5   # 偏高，有回调风险

    # 量价配合
    if len(volume) >= 20:
        vol5 = np.mean(volume[-5:])
        vol20 = np.mean(volume[-20:])
        if vol20 > 0 and vol5 / vol20 > 1.3:
            score += 5

    # 波动率
    if len(close) >= 20:
        returns = np.diff(np.log(close[-20:]))
        vol = np.std(returns) * np.sqrt(252) * 100
        if vol > 50:
            score -= 10
        elif vol < 15:
            score += 5

    return max(0, min(100, score))


def select_with_industry_constraint(scored: list[tuple[str, float]],
                                    top_n: int = 5,
                                    max_per_industry: int = 2) -> list[str]:
    """
    优化5: 行业分散约束选股

    同行业最多选max_per_industry只
    """
    selected = []
    industry_count: dict[str, int] = {}

    for code, score in scored:
        industry = get_industry(code)
        if industry_count.get(industry, 0) >= max_per_industry:
            continue
        selected.append(code)
        industry_count[industry] = industry_count.get(industry, 0) + 1
        if len(selected) >= top_n:
            break

    return selected


def calc_kelly_weight(scores: list[float], recent_returns: list[float]) -> float:
    """
    优化3: Kelly公式计算仓位比例

    基于历史胜率和盈亏比，返回仓位比例（0.2~1.0）
    """
    if len(recent_returns) < 3:
        return 0.5  # 数据不足，用半仓

    wins = [r for r in recent_returns if r > 0]
    losses = [r for r in recent_returns if r < 0]

    if not wins or not losses:
        return 0.5

    p = len(wins) / len(recent_returns)  # 胜率
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))

    if avg_loss < 1e-10:
        return 0.5

    b = avg_win / avg_loss  # 盈亏比
    q = 1 - p

    kelly_full = (p * b - q) / b if b > 0 else 0
    kelly_half = kelly_full / 2  # 半Kelly更保守

    return max(0.2, min(1.0, kelly_half))


def calc_atr(kline: pd.DataFrame, period: int = 14) -> float:
    """计算 ATR（平均真实波幅）"""
    if kline is None or len(kline) < period + 1:
        return 0.0
    high = kline["最高"].values[-period:]
    low = kline["最低"].values[-period:]
    close = kline["收盘"].values[-period - 1:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
    return float(np.mean(tr))


def check_monthly_stoploss(stock_data: dict, top_codes: list[str],
                           rebalance_date, mid_date,
                           stop_loss_pct: float = -8.0,
                           use_atr: bool = False) -> tuple[float, bool]:
    """
    月度止损检查

    use_atr=True 时用 ATR 动态止损：阈值 = max(stop_loss_pct, -2*ATR/price)
    """
    mid_returns = []
    for code in top_codes:
        df = stock_data[code]
        buy_row = df[df["日期"] >= rebalance_date].head(1)
        mid_row = df[df["日期"] >= mid_date].head(1)

        if buy_row.empty or mid_row.empty:
            continue

        buy_p = float(buy_row["收盘"].iloc[0])
        mid_p = float(mid_row["收盘"].iloc[0])

        if buy_p > 0:
            mid_returns.append(mid_p / buy_p - 1)

    if not mid_returns:
        return 0, False

    avg_mid_ret = np.mean(mid_returns) * 100

    # ATR 动态止损
    if use_atr:
        atr_stops = []
        for code in top_codes:
            df = stock_data[code]
            slice_df = df[df["日期"] <= rebalance_date].tail(20)
            atr = calc_atr(slice_df, 14)
            buy_row = df[df["日期"] >= rebalance_date].head(1)
            if not buy_row.empty:
                buy_p = float(buy_row["收盘"].iloc[0])
                if buy_p > 0:
                    atr_pct = -2 * atr / buy_p * 100
                    atr_stops.append(max(stop_loss_pct, atr_pct))
        if atr_stops:
            dynamic_threshold = np.mean(atr_stops)
            return avg_mid_ret, avg_mid_ret < dynamic_threshold

    return avg_mid_ret, avg_mid_ret < stop_loss_pct


def calc_vol_inverse_weight(stock_data: dict, codes: list[str],
                            rebalance_date) -> list[float]:
    """波动率倒数加权：低波动股获得更高权重"""
    vols = []
    for code in codes:
        df = stock_data[code]
        slice_df = df[df["日期"] <= rebalance_date].tail(20)
        if len(slice_df) < 10:
            vols.append(1.0)
            continue
        close = slice_df["收盘"].values
        returns = np.diff(np.log(close))
        vol = np.std(returns)
        vols.append(vol if vol > 1e-8 else 1e-8)

    inv_vols = [1.0 / v for v in vols]
    total = sum(inv_vols)
    return [iv / total for iv in inv_vols] if total > 0 else [1.0 / len(codes)] * len(codes)


def run_backtest(start_date: str = "20240101",
                 end_date: str | None = None,
                 top_n: int = 5,
                 universe: str = "hs300",
                 use_kelly: bool = False,
                 use_vol_weight: bool = False,
                 use_atr_stop: bool = False,
                 use_regime_factor: bool = False,
                 rebalance_freq: str = "monthly",
                 label: str = "") -> dict:
    """执行优化版回测

    Args:
        start_date: 回测起始日
        end_date: 回测结束日（None=数据末尾）
        top_n: 每期选股数
        universe: "hs300" | "zz500" | "hs300+zz500"
        use_kelly: Kelly公式调仓
        use_vol_weight: 波动率倒数加权
        use_atr_stop: ATR动态止损
        use_regime_factor: HMM市场状态反馈仓位
        rebalance_freq: "monthly" | "biweekly" | "quarterly"
        label: 标签（用于区分不同回测配置）
    """
    tag = label or f"{universe}_{rebalance_freq}"
    logger.info(f"=== 正方形系统 v2.2 回测 [{tag}] ===")
    logger.info(f"回测: {start_date} ~ {end_date or '末尾'}, {top_n}只, {universe}")

    # 数据预热期：提前半年获取数据
    data_start = "20230601"

    logger.info(f"获取股票池({universe})...")
    stock_codes = get_sample_stocks(universe)
    logger.info(f"股票池: {len(stock_codes)}只")

    logger.info("获取历史数据...")
    stock_data: dict[str, pd.DataFrame] = {}
    for i, code in enumerate(stock_codes):
        df = fetch_stock_daily(code, start=data_start)
        if df is not None and len(df) >= 120:
            stock_data[code] = df
        if (i + 1) % 20 == 0:
            logger.info(f"  已获取 {i+1}/{len(stock_codes)}")

    logger.info(f"有效股票: {len(stock_data)}只")

    if len(stock_data) < 10:
        logger.error("数据不足")
        return {}

    # 预加载申万行业分类
    fetch_shenwan_industry_map()

    logger.info("获取沪深300基准...")
    benchmark = fetch_index_daily(start=data_start)

    # 生成所有交易日
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df["日期"].tolist())
    all_dates = sorted(all_dates)

    # 按频率生成换仓日
    date_series = pd.Series(all_dates)

    if rebalance_freq == "biweekly":
        # 每两周第一个交易日
        biweekly_dates = []
        for i in range(0, len(all_dates), 10):
            biweekly_dates.append(all_dates[i])
        rebalance_dates = biweekly_dates
    elif rebalance_freq == "quarterly":
        rebalance_dates = date_series.groupby(
            [date_series.dt.year, date_series.dt.quarter]
        ).first().tolist()
    else:  # monthly
        rebalance_dates = date_series.groupby(
            [date_series.dt.year, date_series.dt.month]
        ).first().tolist()

    start_dt = pd.Timestamp(start_date)
    rebalance_dates = [d for d in rebalance_dates if d >= start_dt]
    if end_date:
        end_dt = pd.Timestamp(end_date)
        rebalance_dates = [d for d in rebalance_dates if d <= end_dt]

    if len(rebalance_dates) < 3:
        logger.error("回测期间太短")
        return {}

    logger.info(f"回测期间: {rebalance_dates[0].strftime('%Y-%m-%d')} ~ {rebalance_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"换仓次数: {len(rebalance_dates) - 1}")

    # 逐期回测
    portfolio_returns = []
    monthly_picks = []
    win_months = 0
    total_months = 0
    recent_raw_returns = []
    total_cost = 0.0
    prev_holdings: set[str] = set()

    for i in range(len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]

        # 半月止损检查点
        all_trade_dates = [d for d in all_dates if rebalance_date <= d < next_date]
        mid_idx = len(all_trade_dates) // 2
        mid_date = all_trade_dates[mid_idx] if mid_idx < len(all_trade_dates) else rebalance_date

        # 评分
        scores = {}
        for code, df in stock_data.items():
            df_slice = df[df["日期"] <= rebalance_date]
            s = score_stock_simple(df_slice)
            if s is not None:
                scores[code] = s

        if not scores:
            continue

        # 行业分散约束选股
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_codes = select_with_industry_constraint(sorted_stocks, top_n, max_per_industry=2)
        top_scores = {code: scores[code] for code in top_codes}

        # 仓位权重
        if use_vol_weight:
            weights = calc_vol_inverse_weight(stock_data, top_codes, rebalance_date)
        elif use_kelly:
            kelly_w = calc_kelly_weight(
                [scores[c] for c in top_codes],
                recent_raw_returns[-6:] if recent_raw_returns else []
            )
            weights = [kelly_w / len(top_codes)] * len(top_codes)
        else:
            weights = [1.0 / len(top_codes)] * len(top_codes)

        # HMM 市场状态仓位系数
        regime_factor = 1.0
        if use_regime_factor:
            # 简化：用沪深300近20日涨跌判断牛熊
            bm_slice = benchmark[benchmark["日期"] <= rebalance_date].tail(20)
            if len(bm_slice) >= 20:
                bm_ret = float(bm_slice["收盘"].iloc[-1]) / float(bm_slice["收盘"].iloc[0]) - 1
                if bm_ret > 0.05:
                    regime_factor = REGIME_POSITION_FACTOR["BULL"]
                elif bm_ret < -0.05:
                    regime_factor = REGIME_POSITION_FACTOR["BEAR"]
                else:
                    regime_factor = REGIME_POSITION_FACTOR["SIDEWAYS"]

        # 止损检查
        mid_ret, stopped = check_monthly_stoploss(
            stock_data, top_codes, rebalance_date, mid_date,
            stop_loss_pct=-8.0, use_atr=use_atr_stop
        )

        # 计算持有期收益（含交易成本）
        period_returns = []
        cost_this_month = 0.0
        new_holdings = set(top_codes)

        for j, code in enumerate(top_codes):
            df = stock_data[code]
            if stopped:
                buy_row = df[df["日期"] >= rebalance_date].head(1)
                sell_row = df[df["日期"] >= mid_date].head(1)
            else:
                buy_row = df[df["日期"] >= rebalance_date].head(1)
                sell_row = df[df["日期"] >= next_date].head(1)

            if buy_row.empty or sell_row.empty:
                continue

            buy_price = float(buy_row["收盘"].iloc[0])
            sell_price = float(sell_row["收盘"].iloc[0])

            if buy_price > 0:
                # 交易成本：买入加滑点，卖出减滑点+佣金
                actual_buy = buy_price * (1 + SLIPPAGE)
                actual_sell = sell_price * (1 - SLIPPAGE - COMMISSION)
                ret = actual_sell / actual_buy - 1

                # 持仓不变的股票不需要重新交易
                if code in prev_holdings and not stopped:
                    ret = sell_price / buy_price - 1  # 无成本
                else:
                    cost_rate = SLIPPAGE * 2 + COMMISSION
                    cost_this_month += cost_rate * weights[j]

                period_returns.append(ret)

        total_cost += cost_this_month

        if period_returns:
            # 加权平均收益
            if len(weights) == len(period_returns):
                raw_ret = sum(r * w for r, w in zip(period_returns, weights))
            else:
                raw_ret = np.mean(period_returns)

            raw_ret *= regime_factor  # HMM 仓位系数
            recent_raw_returns.append(raw_ret)
            portfolio_returns.append(raw_ret)

            stop_note = " [止损]" if stopped else ""
            regime_note = f" R={regime_factor:.0%}" if use_regime_factor and regime_factor < 1 else ""
            monthly_picks.append({
                "date": rebalance_date.strftime("%Y-%m"),
                "stocks": top_codes,
                "scores": [f"{c}({top_scores[c]:.0f})" for c in top_codes],
                "return": f"{raw_ret:.2%}{stop_note}{regime_note}",
                "stopped": stopped,
            })
            total_months += 1
            if raw_ret > 0:
                win_months += 1

        prev_holdings = new_holdings if not stopped else set()

    if not portfolio_returns:
        logger.error("无有效回测数据")
        return {}

    # 基准收益
    benchmark_returns = []
    for i in range(len(rebalance_dates) - 1):
        rd = rebalance_dates[i]
        nd = rebalance_dates[i + 1]
        bm_buy = benchmark[benchmark["日期"] >= rd].head(1)
        bm_sell = benchmark[benchmark["日期"] >= nd].head(1)
        if not bm_buy.empty and not bm_sell.empty:
            bp = float(bm_buy["收盘"].iloc[0])
            sp = float(bm_sell["收盘"].iloc[0])
            if bp > 0:
                benchmark_returns.append(sp / bp - 1)

    # 统计
    portfolio_arr = np.array(portfolio_returns)
    benchmark_arr = np.array(benchmark_returns[:len(portfolio_returns)])

    cum_port = np.cumprod(1 + portfolio_arr)
    cum_bench = np.cumprod(1 + benchmark_arr) if len(benchmark_arr) > 0 else np.ones_like(cum_port)

    total_return = cum_port[-1] - 1
    bench_total = cum_bench[-1] - 1 if len(cum_bench) > 0 else 0
    excess = total_return - bench_total

    n_months = len(portfolio_returns)
    annual_return = (1 + total_return) ** (12 / n_months) - 1 if n_months > 0 else 0
    bench_annual = (1 + bench_total) ** (12 / n_months) - 1 if n_months > 0 else 0

    peak = np.maximum.accumulate(cum_port)
    drawdown = (cum_port - peak) / peak
    max_drawdown = float(np.min(drawdown))

    win_rate = win_months / total_months if total_months > 0 else 0

    if np.std(portfolio_returns) > 0:
        sharpe = float(np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(12))
    else:
        sharpe = 0

    stop_count = sum(1 for p in monthly_picks if p.get("stopped", False))

    return {
        "label": tag,
        "total_return": total_return,
        "benchmark_return": bench_total,
        "excess_return": excess,
        "annual_return": annual_return,
        "benchmark_annual": bench_annual,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "n_months": n_months,
        "stop_count": stop_count,
        "total_cost": total_cost,
        "use_kelly": use_kelly,
        "use_vol_weight": use_vol_weight,
        "use_atr_stop": use_atr_stop,
        "use_regime_factor": use_regime_factor,
        "monthly_picks": monthly_picks,
    }


def print_result(result: dict):
    """打印回测结果"""
    label = result.get("label", "")
    print("\n" + "=" * 50)
    print(f"正方形系统 v2.2 回测结果 [{label}]")
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
    print(f"止损次数: {result['stop_count']}")
    print(f"交易成本: {result.get('total_cost', 0):.2%}")
    print(f"配置:     Kelly={'开' if result.get('use_kelly') else '关'}, "
          f"VolW={'开' if result.get('use_vol_weight') else '关'}, "
          f"ATR={'开' if result.get('use_atr_stop') else '关'}, "
          f"Regime={'开' if result.get('use_regime_factor') else '关'}")
    print()

    print("每月选股:")
    print("-" * 60)
    for pick in result.get("monthly_picks", []):
        stocks_str = ", ".join(pick["scores"])
        ret = pick["return"]
        print(f"  {pick['date']}: {stocks_str} → {ret}")

    print()
    if result['excess_return'] > 0:
        print(f"结论: 跑赢基准 {result['excess_return']:+.2%}")
    else:
        print(f"结论: 跑输基准 {result['excess_return']:+.2%}")


def run_walk_forward(top_n: int = 5, universe: str = "hs300") -> dict:
    """样本外验证：train/test 时间分割

    训练期：2024-01 ~ 2025-06
    测试期：2025-07 ~ 2026-04
    """
    logger.info("=== 样本外验证 ===")

    train_result = run_backtest(
        start_date="20240101", end_date="20250630",
        top_n=top_n, universe=universe, label="训练期"
    )
    test_result = run_backtest(
        start_date="20250701", end_date=None,
        top_n=top_n, universe=universe, label="测试期"
    )

    return {"train": train_result, "test": test_result}


def print_walk_forward(wf: dict):
    """打印样本外验证结果"""
    train = wf.get("train", {})
    test = wf.get("test", {})

    if not train or not test:
        print("样本外验证数据不足")
        return

    print("\n" + "=" * 60)
    print("样本外验证结果")
    print("=" * 60)
    print(f"{'指标':<12} {'训练期':>10} {'测试期':>10} {'偏差':>10}")
    print("-" * 42)

    for key, label in [("total_return", "总收益"), ("excess_return", "超额收益"),
                       ("annual_return", "年化收益"), ("sharpe", "夏普比率"),
                       ("win_rate", "月胜率"), ("max_drawdown", "最大回撤")]:
        t = train.get(key, 0)
        v = test.get(key, 0)
        if key == "win_rate":
            print(f"{label:<12} {t:>9.0%} {v:>9.0%} {abs(t-v):>9.0%}")
        elif key == "sharpe":
            print(f"{label:<12} {t:>9.2f} {v:>9.2f} {abs(t-v):>9.2f}")
        else:
            print(f"{label:<12} {t:>+9.2%} {v:>+9.2%} {abs(t-v):>9.2%}")

    # 判断是否过拟合
    train_excess = train.get("excess_return", 0)
    test_excess = test.get("excess_return", 0)
    print()
    if test_excess > 0 and train_excess > 0:
        print(f"结论: 训练期和测试期均跑赢基准，策略有正期望值")
        if abs(train_excess - test_excess) / max(abs(train_excess), 0.01) < 0.3:
            print(f"  偏差<30%，过拟合风险低")
        else:
            print(f"  偏差>30%，可能存在一定过拟合")
    elif test_excess > 0:
        print(f"结论: 测试期跑赢基准，但训练期未跑赢，需进一步验证")
    else:
        print(f"结论: 测试期跑输基准，策略可能过拟合或已失效")


def run_comparison(top_n: int = 5, universe: str = "hs300") -> list[dict]:
    """多配置对比"""
    configs = [
        {"label": "基准(等权)", "use_kelly": False, "use_vol_weight": False,
         "use_atr_stop": False, "use_regime_factor": False, "rebalance_freq": "monthly"},
        {"label": "波动率加权", "use_kelly": False, "use_vol_weight": True,
         "use_atr_stop": False, "use_regime_factor": False, "rebalance_freq": "monthly"},
        {"label": "ATR止损", "use_kelly": False, "use_vol_weight": False,
         "use_atr_stop": True, "use_regime_factor": False, "rebalance_freq": "monthly"},
        {"label": "HMM仓位", "use_kelly": False, "use_vol_weight": False,
         "use_atr_stop": False, "use_regime_factor": True, "rebalance_freq": "monthly"},
        {"label": "季度换仓", "use_kelly": False, "use_vol_weight": False,
         "use_atr_stop": False, "use_regime_factor": False, "rebalance_freq": "quarterly"},
    ]

    results = []
    for cfg in configs:
        r = run_backtest(top_n=top_n, universe=universe, **cfg)
        if r:
            results.append(r)
    return results


def print_comparison(results: list[dict]):
    """打印多配置对比表"""
    if not results:
        print("无对比数据")
        return

    print("\n" + "=" * 75)
    print("多配置对比")
    print("=" * 75)
    header = f"{'配置':<14} {'总收益':>8} {'超额':>8} {'年化':>8} {'夏普':>6} {'回撤':>8} {'胜率':>6} {'成本':>6}"
    print(header)
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<14} {r['total_return']:>+7.2%} {r['excess_return']:>+7.2%} "
              f"{r['annual_return']:>+7.2%} {r['sharpe']:>5.2f} {r['max_drawdown']:>7.2%} "
              f"{r['win_rate']:>5.0%} {r.get('total_cost', 0):>5.2%}")
    print()


if __name__ == "__main__":
    import sys as _sys

    mode = _sys.argv[1] if len(_sys.argv) > 1 else "single"

    if mode == "walkforward":
        wf = run_walk_forward(top_n=5, universe="hs300")
        print_walk_forward(wf)
    elif mode == "compare":
        results = run_comparison(top_n=5, universe="hs300")
        print_comparison(results)
    elif mode == "full":
        # 全量沪深300 + 交易成本 + 样本外
        wf = run_walk_forward(top_n=5, universe="hs300")
        print_walk_forward(wf)
        r = run_backtest(start_date="20240101", top_n=5, universe="hs300", label="全量")
        print_result(r)
    else:
        # 默认：单次回测（与之前版本兼容）
        result = run_backtest(start_date="20240101", top_n=5, universe="hs300")
        if result:
            print_result(result)
