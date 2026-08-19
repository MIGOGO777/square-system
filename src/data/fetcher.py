"""
数据层统一接口 — 对接 a-stock-data SKILL.md (v3.1)

真实数据源架构：
  行情层: mootdx (K线/盘口/财务/F10) + 腾讯财经 (PE/PB/市值)
  信号层: 同花顺热点 + 同花顺北向(自缓存) + 龙虎榜 + 行业对比
  资金面: 东财push2his(主力资金流120日) + 东财datacenter(融资融券/大宗交易/股东户数/分红)
  基础数据: mootdx finance (37字段季报) + F10 (9大类文本) + 新浪财报三表
  宏观层: 无内置宏源，用估算/默认值

依赖: mootdx, requests, pandas
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────── 配置 ───────────────────────────

@dataclass
class FetcherConfig:
    retry_times: int = 3
    retry_delay: float = 1.0
    timeout: int = 10


# ─────────────────────────── 懒加载 ───────────────────────────

_mootdx_quotes: Any = None
_requests: Any = None


def _get_mootdx():
    global _mootdx_quotes
    if _mootdx_quotes is None:
        from mootdx.quotes import Quotes
        _mootdx_quotes = Quotes
    return _mootdx_quotes


def _get_requests():
    global _requests
    if _requests is None:
        import requests as _r
        _requests = _r
    return _requests


# ─────────────────────────── 工具函数 ───────────────────────────

def _retry(func: Callable, times: int = 3, delay: float = 1.0) -> Any:
    last_err = None
    for i in range(times):
        try:
            return func()
        except Exception as e:
            last_err = e
            if i < times - 1:
                time.sleep(delay * (2 ** i))
    raise last_err


def _get_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith("8"):
        return "bj"
    else:
        return "sz"


# ─────────────────────────── DataFetcher ───────────────────────────

@dataclass
class DataFetcher:
    """A股数据获取器 — 包装 a-stock-data SKILL.md 中的真实函数"""
    config: FetcherConfig = field(default_factory=FetcherConfig)

    # ═══════════════════ 行情层 ═══════════════════

    def get_daily_kline(self, symbol: str, count: int = 60) -> pd.DataFrame:
        """
        日K线 (mootdx client.bars)
        返回: open, close, high, low, vol, amount, datetime
        """
        def _fetch():
            Quotes = _get_mootdx()
            client = Quotes.factory(market='std')
            df = client.bars(symbol=symbol, category=4, count=count)
            if df is not None and not df.empty:
                df['symbol'] = symbol
                if 'datetime' in df.columns:
                    df['date'] = pd.to_datetime(df['datetime']).dt.date
            return df if df is not None else pd.DataFrame()

        try:
            return _retry(_fetch, self.config.retry_times, self.config.retry_delay)
        except Exception as e:
            logger.warning(f"mootdx K线获取失败({symbol}): {e}")
            return pd.DataFrame()

    def get_valuation(self, codes: str | list[str]) -> pd.DataFrame:
        """
        估值数据 (腾讯财经 tencent_quote)
        返回: symbol, name, price, pe_ttm, pb, mcap_yi, float_mcap_yi,
              turnover_pct, limit_up, limit_down, change_pct 等
        """
        if isinstance(codes, str):
            code_list = [codes]
        else:
            code_list = list(codes)

        prefixed = [_get_prefix(c) + c for c in code_list]
        url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)

        def _fetch():
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                raw = resp.read().decode("gbk")

            rows = []
            for line in raw.strip().split(";"):
                if not line.strip() or "=" not in line or '"' not in line:
                    continue
                vals = line.split('"')[1].split("~")
                if len(vals) < 53:
                    continue
                code_clean = line.split("=")[0].split("_")[-1][2:]
                rows.append({
                    "symbol": code_clean,
                    "name": vals[1],
                    "price": float(vals[3]) if vals[3] else 0.0,
                    "last_close": float(vals[4]) if vals[4] else 0.0,
                    "open": float(vals[5]) if vals[5] else 0.0,
                    "change_pct": float(vals[32]) if vals[32] else 0.0,
                    "amount_wan": float(vals[37]) if vals[37] else 0.0,
                    "turnover_pct": float(vals[38]) if vals[38] else 0.0,
                    "pe_ttm": float(vals[39]) if vals[39] else 0.0,
                    "mcap_yi": float(vals[44]) if vals[44] else 0.0,
                    "float_mcap_yi": float(vals[45]) if vals[45] else 0.0,
                    "pb": float(vals[46]) if vals[46] else 0.0,
                    "limit_up": float(vals[47]) if vals[47] else 0.0,
                    "limit_down": float(vals[48]) if vals[48] else 0.0,
                    "pe_static": float(vals[52]) if vals[52] else 0.0,
                })
            return pd.DataFrame(rows)

        return _retry(_fetch, self.config.retry_times, self.config.retry_delay)

    # ═══════════════════ 信号层 ═══════════════════

    def get_north_flow(self) -> pd.DataFrame:
        """
        北向资金实时分钟流向 (同花顺 hsgtApi)
        返回: time, hgt_yi(沪股通), sgt_yi(深股通)，单位: 亿元
        """
        r = _get_requests()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0",
            "Host": "data.hexin.cn",
            "Referer": "https://data.hexin.cn/",
        }

        def _fetch():
            resp = r.get(
                "https://data.hexin.cn/market/hsgtApi/method/dayChart/",
                headers=headers, timeout=self.config.timeout
            )
            d = resp.json()
            times = d.get("time", [])
            hgt = d.get("hgt", [])
            sgt = d.get("sgt", [])
            n = len(times)
            return pd.DataFrame({
                "time": times,
                "hgt_yi": hgt[:n] + [None] * max(0, n - len(hgt)),
                "sgt_yi": sgt[:n] + [None] * max(0, n - len(sgt)),
            })

        try:
            return _retry(_fetch, 2, self.config.retry_delay)
        except Exception as e:
            logger.warning(f"北向资金实时获取失败: {e}")
            return pd.DataFrame()

    def get_north_flow_history(self, n: int = 20) -> pd.DataFrame:
        """
        北向资金历史 (本地CSV自缓存)
        返回: date, hgt, sgt
        """
        path = Path.home() / ".tradingagents" / "cache" / "northbound_daily.csv"
        if not path.exists():
            logger.info("北向历史缓存不存在，返回空")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            return df.tail(n)
        except Exception:
            return pd.DataFrame()

    def get_hot_stocks(self, trade_date: str | None = None) -> pd.DataFrame:
        """
        同花顺当日强势股+题材归因 (ths_hot_reason)
        返回: 代码, 名称, 题材归因, 收盘价, 涨幅%, 换手率%, 成交额, 大单净量
        """
        r = _get_requests()
        if trade_date is None:
            d = date.today().strftime("%Y-%m-%d")
        else:
            d = trade_date

        def _fetch():
            url = (
                f"http://zx.10jqka.com.cn/event/api/getharden/"
                f"date/{d}/orderby/date/orderway/desc/charset/GBK/"
            )
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0"}
            resp = r.get(url, headers=headers, timeout=10)
            data = resp.json()
            if data.get("errocode", 0) != 0:
                raise RuntimeError(f"同花顺热点错误: {data.get('errormsg','')}")
            rows = data.get("data") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return df
            df = df.rename(columns={
                "name": "名称", "code": "代码", "reason": "题材归因",
                "close": "收盘价", "zhangdie": "涨跌额", "zhangfu": "涨幅%",
                "huanshou": "换手率%", "chengjiaoe": "成交额",
                "chengjiaoliang": "成交量", "ddejingliang": "大单净量",
                "market": "市场",
            })
            return df

        return _retry(_fetch, 2, self.config.retry_delay)

    def get_industry_comparison(self, top_n: int = 20) -> dict:
        """
        行业横向对比 (同花顺90行业 via akshare)
        返回: {top: [{rank, name, change_pct, up_count, down_count, leader}], bottom: [...], total: int}

        注意: 此函数仍用 akshare，因为 a-stock-data 的 industry_comparison 内部也调 akshare。
        """
        try:
            import akshare as ak
            df = ak.stock_board_industry_summary_ths()
            if df.empty:
                return {"top": [], "bottom": [], "total": 0}

            rows = []
            for i, row in df.iterrows():
                rows.append({
                    "rank": i + 1,
                    "name": row.get("板块", ""),
                    "change_pct": row.get("涨跌幅", 0),
                    "turnover_yi": row.get("总成交额", 0),
                    "net_inflow_yi": row.get("净流入", 0) if "净流入" in df.columns else None,
                    "up_count": row.get("上涨家数", 0),
                    "down_count": row.get("下跌家数", 0),
                    "leader": row.get("领涨股", ""),
                })

            return {
                "top": rows[:top_n],
                "bottom": rows[-top_n:],
                "total": len(rows),
            }
        except Exception as e:
            logger.warning(f"行业对比获取失败: {e}")
            return {"top": [], "bottom": [], "total": 0}

    def get_dragon_tiger_all(self, trade_date: str | None = None) -> dict:
        """
        全市场龙虎榜 (daily_dragon_tiger, 东财 datacenter API)
        返回: {date, total_records, stocks: [{code, name, reason, close, change_pct,
               net_buy_wan, buy_wan, sell_wan, turnover_pct}]}
        """
        r = _get_requests()
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        def _fetch():
            url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
                "columns": "ALL",
                "filter": f"(TRADE_DATE>='{trade_date}')(TRADE_DATE<='{trade_date}')",
                "pageNumber": "1",
                "pageSize": "500",
                "sortTypes": "-1",
                "sortColumns": "BILLBOARD_NET_AMT",
                "source": "WEB",
                "client": "WEB",
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": "https://data.eastmoney.com/",
            }
            resp = r.get(url, params=params, headers=headers, timeout=15)
            d = resp.json()
            if not d.get("success") or not d.get("result") or not d["result"].get("data"):
                return {"date": trade_date, "total_records": 0, "stocks": []}
            data = d["result"]["data"]
            actual_date = data[0].get("TRADE_DATE", "")[:10] if data else trade_date
            stocks = []
            for row in data:
                net_buy = (row.get("BILLBOARD_NET_AMT") or 0) / 10000
                stocks.append({
                    "code": row.get("SECURITY_CODE", ""),
                    "name": row.get("SECURITY_NAME_ABBR", ""),
                    "reason": row.get("EXPLANATION", ""),
                    "close": row.get("CLOSE_PRICE") or 0,
                    "change_pct": round(float(row.get("CHANGE_RATE") or 0), 2),
                    "net_buy_wan": round(net_buy, 1),
                    "buy_wan": round((row.get("BILLBOARD_BUY_AMT") or 0) / 10000, 1),
                    "sell_wan": round((row.get("BILLBOARD_SELL_AMT") or 0) / 10000, 1),
                    "turnover_pct": round(float(row.get("TURNOVERRATE") or 0), 2),
                })
            return {"date": actual_date, "total_records": len(stocks), "stocks": stocks}

        try:
            return _retry(_fetch, 2, self.config.retry_delay)
        except Exception as e:
            logger.warning(f"龙虎榜获取失败: {e}")
            return {"date": trade_date, "total_records": 0, "stocks": []}

    # ═══════════════════ 资金面 / 筹码层 (a-stock-data v3.1) ═══════════════════

    _UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    _DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    def _eastmoney_datacenter(self, report_name: str, columns: str = "ALL",
                              filter_str: str = "", page_size: int = 50,
                              sort_columns: str = "", sort_types: str = "-1") -> list[dict]:
        """东财数据中心统一查询 — 龙虎榜/解禁/融资融券/大宗交易/股东户数/分红 共用"""
        r = _get_requests()
        params = {
            "reportName": report_name, "columns": columns,
            "filter": filter_str, "pageNumber": "1", "pageSize": str(page_size),
            "sortColumns": sort_columns, "sortTypes": sort_types,
            "source": "WEB", "client": "WEB",
        }
        resp = r.get(self._DATACENTER_URL, params=params,
                     headers={"User-Agent": self._UA}, timeout=15)
        d = resp.json()
        if d.get("result") and d["result"].get("data"):
            return d["result"]["data"]
        return []

    def get_fund_flow_120d(self, code: str) -> list[dict]:
        """
        个股资金流（日级，最近120个交易日）。
        返回: [{date, main_net(主力净流入), small_net, mid_net, large_net, super_net}]
        单位: 元
        """
        r = _get_requests()
        market_code = 1 if code.startswith("6") else 0

        def _fetch():
            url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
            params = {
                "secid": f"{market_code}.{code}",
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                "lmt": "120",
            }
            resp = r.get(url, params=params, headers={"User-Agent": self._UA},
                         timeout=3)
            d = resp.json()
            klines = d.get("data", {}).get("klines", [])
            rows = []
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 7:
                    rows.append({
                        "date": parts[0],
                        "main_net": float(parts[1]) if parts[1] != "-" else 0,
                        "small_net": float(parts[2]) if parts[2] != "-" else 0,
                        "mid_net": float(parts[3]) if parts[3] != "-" else 0,
                        "large_net": float(parts[4]) if parts[4] != "-" else 0,
                        "super_net": float(parts[5]) if parts[5] != "-" else 0,
                    })
            return rows

        try:
            return _fetch()
        except Exception as e:
            logger.warning(f"资金流获取失败({code}): {e}")
            return []

    def get_margin_data(self, code: str) -> dict:
        """
        融资融券明细（日级）。
        返回: {rzye(融资余额), rzmre(融资买入), rqye(融券余额), date}
        """
        try:
            data = self._eastmoney_datacenter(
                "RPTA_WEB_RZRQ_GGMX",
                filter_str=f'(SCODE="{code}")',
                page_size=5,
                sort_columns="DATE", sort_types="-1",
            )
            if not data:
                return {}
            row = data[0]
            return {
                "date": str(row.get("DATE", ""))[:10],
                "rzye": row.get("RZYE", 0),
                "rzmre": row.get("RZMRE", 0),
                "rqye": row.get("RQYE", 0),
                "rzrqye": row.get("RZRQYE", 0),
            }
        except Exception as e:
            logger.debug(f"融资融券获取失败({code}): {e}")
            return {}

    def get_holder_change(self, code: str) -> list[dict]:
        """
        股东户数变化（季度级）。
        返回: [{date, holder_num, change_num, change_ratio, avg_shares}]
        """
        try:
            data = self._eastmoney_datacenter(
                "RPT_HOLDERNUMLATEST",
                filter_str=f'(SECURITY_CODE="{code}")',
                page_size=10,
                sort_columns="END_DATE", sort_types="-1",
            )
            rows = []
            for row in data:
                rows.append({
                    "date": str(row.get("END_DATE", ""))[:10],
                    "holder_num": row.get("HOLDER_NUM", 0),
                    "change_num": row.get("HOLDER_NUM_CHANGE", 0),
                    "change_ratio": row.get("HOLDER_NUM_RATIO", 0),
                    "avg_shares": row.get("AVG_FREE_SHARES", 0),
                })
            return rows
        except Exception as e:
            logger.debug(f"股东户数获取失败({code}): {e}")
            return []

    def get_dividend_history(self, code: str) -> list[dict]:
        """
        分红送转历史。
        返回: [{date, bonus_rmb(每股派息), transfer_ratio, bonus_ratio}]
        """
        try:
            data = self._eastmoney_datacenter(
                "RPT_SHAREBONUS_DET",
                filter_str=f'(SECURITY_CODE="{code}")',
                page_size=20,
                sort_columns="EX_DIVIDEND_DATE", sort_types="-1",
            )
            rows = []
            for row in data:
                rows.append({
                    "date": str(row.get("EX_DIVIDEND_DATE", ""))[:10],
                    "bonus_rmb": row.get("PRETAX_BONUS_RMB", 0),
                    "transfer_ratio": row.get("TRANSFER_RATIO", 0),
                    "bonus_ratio": row.get("BONUS_RATIO", 0),
                    "plan": row.get("ASSIGN_PROGRESS", ""),
                })
            return rows
        except Exception as e:
            logger.debug(f"分红历史获取失败({code}): {e}")
            return []

    def get_block_trade(self, code: str) -> list[dict]:
        """
        大宗交易记录。
        返回: [{date, price, vol, amount, buyer, seller, premium_pct}]
        """
        try:
            data = self._eastmoney_datacenter(
                "RPT_DATA_BLOCKTRADE",
                filter_str=f'(SECURITY_CODE="{code}")',
                page_size=20,
                sort_columns="TRADE_DATE", sort_types="-1",
            )
            rows = []
            for row in data:
                close = row.get("CLOSE_PRICE") or 0
                deal_price = row.get("DEAL_PRICE") or 0
                premium = ((deal_price / close - 1) * 100) if close else 0
                rows.append({
                    "date": str(row.get("TRADE_DATE", ""))[:10],
                    "price": deal_price,
                    "close": close,
                    "premium_pct": round(premium, 2),
                    "vol": row.get("DEAL_VOLUME", 0),
                    "amount": row.get("DEAL_AMT", 0),
                    "buyer": row.get("BUYER_NAME", ""),
                    "seller": row.get("SELLER_NAME", ""),
                })
            return rows
        except Exception as e:
            logger.debug(f"大宗交易获取失败({code}): {e}")
            return []

    def get_industry_from_baidu(self, code: str) -> dict:
        """
        百度股市通 — 概念板块归属（行业/概念/地域三维分类）
        返回: {industry: [str], concept: [str], region: str}
        """
        r = _get_requests()

        def _fetch():
            url = "https://finance.pae.baidu.com/vapi/v1/getquotation"
            params = {"srcid": "5353", "all": "1", "pointType": "string",
                      "group": "quotation_block_ab", "query": code,
                      "code": code, "market_type": "ab"}
            headers = {"User-Agent": self._UA,
                       "Origin": "https://gushitong.baidu.com",
                       "Referer": "https://gushitong.baidu.com/"}
            resp = r.get(url, params=params, headers=headers, timeout=10)
            d = resp.json()
            result = d.get("Result", {})
            blocks = result.get("blocks", [])
            industry = []
            concept = []
            region = ""
            for b in blocks:
                bt = b.get("type", "")
                name = b.get("name", "")
                if "行业" in bt:
                    industry.append(name)
                elif "概念" in bt or "主题" in bt:
                    concept.append(name)
                elif "地域" in bt or "地区" in bt:
                    region = name
            return {"industry": industry, "concept": concept, "region": region}

        try:
            return _retry(_fetch, 2, self.config.retry_delay)
        except Exception as e:
            logger.debug(f"百度板块获取失败({code}): {e}")
            return {"industry": [], "concept": [], "region": ""}

    # ═══════════════════ 基础数据 ═══════════════════

    def get_quarterly(self, code: str) -> dict:
        """
        季报快照 (mootdx client.finance, 37字段)
        返回适配规则引擎的格式:
          net_margin, gross_margin, roe_list, fcf_list, debt_ratio,
          revenue_growth, revenue, net_assets
        """
        Quotes = _get_mootdx()
        client = Quotes.factory(market='std')
        try:
            df = client.finance(symbol=code)
            if isinstance(df, pd.DataFrame) and not df.empty:
                raw = df.iloc[0].to_dict()
                raw["_code"] = code  # 注入代码供 _adapt_quarterly 查新浪
                return self._adapt_quarterly(raw)
            return {}
        except Exception as e:
            logger.debug(f"mootdx finance 获取失败({code}): {e}")
            return {}

    def _adapt_quarterly(self, raw: dict) -> dict:
        """将 mootdx finance 37字段适配为规则引擎期望的格式，用新浪财报补全毛利率/负债率"""
        # mootdx 字段名: zhuyingshouru(主营收入), jinglirun(净利润),
        #   meigujingzichan(每股净资产), jingzichan(净资产),
        #   zongzichan(总资产), liudongfuzhai(流动负债), changqifuzhai(长期负债)
        profit = raw.get("jinglirun", 0) or raw.get("profit", 0)  # 净利润
        income = raw.get("zhuyingshouru", 0) or raw.get("income", 0)  # 主营收入
        bvps = raw.get("meigujingzichan", 0) or raw.get("bvps", 0)  # 每股净资产

        # 估算净利率
        net_margin = 0.0
        if income and income > 0 and profit:
            net_margin = (profit / income) * 100

        # 估算ROE: 净利润 / 净资产 * 100
        net_assets = raw.get("jingzichan", 0) or raw.get("meigujingzichan", 0)
        roe = 0.0
        if net_assets and net_assets > 0 and profit:
            roe = (profit / net_assets) * 100

        # 估算负债率: (流动负债+长期负债) / 总资产 * 100
        total_assets = raw.get("zongzichan", 0)
        current_liabilities = raw.get("liudongfuzhai", 0)
        long_liabilities = raw.get("changqifuzhai", 0)
        debt_ratio = 0.0
        if total_assets and total_assets > 0:
            debt_ratio = (current_liabilities + long_liabilities) / total_assets * 100

        result = {
            "net_margin": net_margin,
            "gross_margin": 0.0,  # 用新浪财报补全
            "roe_list": [round(roe, 2)] if roe else [],
            "fcf_list": [],
            "debt_ratio": round(debt_ratio, 2),
            "revenue_growth": 0.0,  # 用新浪财报补全
            "revenue": income,
            "net_assets": bvps,
            "eps": 0.0,  # mootdx不直接提供
            "bvps": bvps,
            "_raw": raw,
        }

        # 用新浪财报三表补全毛利率、营收增长
        code = raw.get("_code", "")
        if code:
            try:
                sina = self._fetch_sina_financials(code)
                if sina:
                    if sina.get("gross_margin"):
                        result["gross_margin"] = sina["gross_margin"]
                    if sina.get("revenue_growth"):
                        result["revenue_growth"] = sina["revenue_growth"]
            except Exception:
                pass

        return result

    def _fetch_sina_financials(self, code: str) -> dict:
        """
        从新浪财报三表获取毛利率、负债率、营收增长。
        返回: {gross_margin, debt_ratio, revenue_growth}
        """
        r = _get_requests()
        prefix = "sh" if code.startswith("6") else "sz"
        paper_code = f"{prefix}{code}"
        url = "https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022"

        def _parse_report_items(d: dict, source: str) -> list[dict]:
            """解析新浪财报API的新格式：report_list[date].data[item]"""
            report_list = d.get("result", {}).get("data", {}).get("report_list", {})
            results = []
            for date_key in sorted(report_list.keys(), reverse=True):
                report = report_list[date_key]
                items = report.get("data", [])
                row = {}
                for item in items:
                    if item.get("item_source") == source:
                        field = item.get("item_field", "")
                        title = item.get("item_title", "")
                        val = item.get("item_value", "0")
                        try:
                            row[field] = float(val)
                        except (ValueError, TypeError):
                            row[field] = 0
                        row[f"_title_{field}"] = title
                if row:
                    results.append(row)
            return results

        result = {}

        # 利润表：毛利率 + 营收增长（同比）
        try:
            params = {"paperCode": paper_code, "source": "lrb", "type": "0",
                      "page": "1", "num": "4"}
            resp = r.get(url, params=params, headers={"User-Agent": self._UA}, timeout=10)
            d = resp.json()
            report_list = d.get("result", {}).get("data", {}).get("report_list", {})
            if report_list:
                latest_date = sorted(report_list.keys(), reverse=True)[0]
                items = report_list[latest_date].get("data", [])
                for item in items:
                    field = item.get("item_field", "")
                    if field == "BIZINCO":
                        revenue = float(item.get("item_value", 0) or 0)
                        tongbi = item.get("item_tongbi", 0)
                        if tongbi is not None:
                            result["revenue_growth"] = round(float(tongbi) * 100, 2)
                    if field == "BIZCOST":
                        cost = float(item.get("item_value", 0) or 0)
                if revenue > 0 and cost > 0:
                    result["gross_margin"] = round((revenue - cost) / revenue * 100, 2)
        except Exception:
            pass

        # 资产负债表：负债率
        try:
            params = {"paperCode": paper_code, "source": "fzb", "type": "0",
                      "page": "1", "num": "2"}
            resp = r.get(url, params=params, headers={"User-Agent": self._UA}, timeout=10)
            d = resp.json()
            reports = _parse_report_items(d, "fzb")
            if reports:
                latest = reports[0]
                # TOTASSET=资产总计, TOTLIAB=负债合计
                total_assets = latest.get("TOTASSET", 0)
                total_liabilities = latest.get("TOTLIAB", 0)
                if total_assets > 0:
                    result["debt_ratio"] = round(total_liabilities / total_assets * 100, 2)
        except Exception:
            pass

        return result

    def get_f10(self, code: str, category: str = "公司概况") -> str:
        """
        F10 公司资料 (mootdx client.F10)
        category: 最新提示/公司概况/财务分析/股东研究/股本结构/资本运作/业内点评/行业分析/公司大事
        """
        Quotes = _get_mootdx()
        client = Quotes.factory(market='std')
        try:
            return client.F10(symbol=code, name=category) or ""
        except Exception:
            return ""

    def get_f10_all(self, code: str) -> dict[str, str]:
        """F10 全部9大类"""
        categories = [
            "最新提示", "公司概况", "财务分析",
            "股东研究", "股本结构", "资本运作",
            "业内点评", "行业分析", "公司大事",
        ]
        result = {}
        for cat in categories:
            result[cat] = self.get_f10(code, cat)
        return result

    def get_stock_info(self, code: str) -> dict:
        """
        个股基本面 (东财 push2 + 百度股市通 fallback)
        返回: code, name, industry, total_shares, float_shares, mcap, list_date
        """
        r = _get_requests()
        market_code = 1 if code.startswith("6") else 0

        # 优先：东财 push2
        def _fetch_eastmoney():
            url = "https://push2.eastmoney.com/api/qt/stock/get"
            params = {
                "fltt": "2", "invt": "2",
                "fields": "f57,f58,f84,f85,f127,f116,f117,f189,f43",
                "secid": f"{market_code}.{code}",
            }
            resp = r.get(url, params=params, headers={"User-Agent": self._UA}, timeout=10)
            d = resp.json().get("data", {})
            return {
                "code": d.get("f57", ""),
                "name": d.get("f58", ""),
                "industry": d.get("f127", ""),
                "total_shares": d.get("f84", 0),
                "float_shares": d.get("f85", 0),
                "mcap": d.get("f116", 0),
                "float_mcap": d.get("f117", 0),
                "list_date": str(d.get("f189", "")),
                "price": d.get("f43", 0),
            }

        try:
            return _retry(_fetch_eastmoney, 2, self.config.retry_delay)
        except Exception:
            pass

        # Fallback: 百度股市通行业分类
        try:
            baidu = self.get_industry_from_baidu(code)
            industry = baidu.get("industry", [None])[0] if baidu.get("industry") else ""
            return {"industry": industry or ""}
        except Exception as e:
            logger.debug(f"个股基本面获取失败({code}): {e}")
            return {}

    def get_full_valuation(self, code: str) -> dict:
        """
        完整估值 (full_valuation)
        返回: name, price, mcap_yi, pe_ttm, pb, eps_cur, eps_next, pe_fwd, cagr_pct, peg
        """
        # 1. 腾讯实时行情
        prefix = _get_prefix(code)
        url = f"https://qt.gtimg.cn/q={prefix}{code}"

        def _fetch():
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            resp = urllib.request.urlopen(req, timeout=10)
            data = resp.read().decode("gbk")
            vals = data.split('"')[1].split("~")
            price = float(vals[3])
            mcap = float(vals[44])
            pe_ttm = float(vals[39]) if vals[39] else 0
            pb = float(vals[46]) if vals[46] else 0

            return {
                "name": vals[1],
                "price": price,
                "mcap_yi": mcap,
                "pe_ttm": pe_ttm,
                "pb": pb,
            }

        try:
            result = _retry(_fetch, 2, self.config.retry_delay)
        except Exception as e:
            logger.warning(f"估值获取失败({code}): {e}")
            return {}

        # 2. 机构一致预期 (akshare)
        try:
            import akshare as ak
            import math
            df = ak.stock_profit_forecast_ths(symbol=code, indicator="预测年报每股收益")
            eps_cur = eps_next = None
            analyst_count = 0
            years_sorted = sorted(df["年度"].unique())
            for _, row in df.iterrows():
                y = str(row["年度"])
                if y == str(years_sorted[0]) if len(years_sorted) > 0 else False:
                    eps_cur = float(row["均值"])
                    analyst_count = int(row["预测机构数"])
                elif y == str(years_sorted[1]) if len(years_sorted) > 1 else False:
                    eps_next = float(row["均值"])

            pe_fwd = result["price"] / eps_cur if eps_cur else float("inf")
            cagr = (eps_next / eps_cur - 1) if (eps_cur and eps_next) else 0
            peg = pe_fwd / (cagr * 100) if cagr > 0 else float("inf")

            result.update({
                "eps_cur": eps_cur,
                "eps_next": eps_next,
                "pe_fwd": round(pe_fwd, 1) if eps_cur else None,
                "cagr_pct": round(cagr * 100, 0) if cagr else None,
                "peg": round(peg, 2) if peg != float("inf") else None,
                "analyst_count": analyst_count,
            })
        except Exception:
            pass

        return result

    # ═══════════════════ 数据聚合（供 engine 使用） ═══════════════════

    def get_emotion_temperature(self) -> float:
        """
        计算情绪温度（0-100）
        基于涨停池数据推算：
        - 涨停数 > 80 → 高温
        - 涨停数 < 20 → 低温
        """
        try:
            hot = self.get_hot_stocks()
            if hot.empty:
                return 50.0

            count = len(hot)
            # 涨停数 → 温度映射
            if count >= 100:
                return 90.0
            elif count >= 80:
                return 80.0
            elif count >= 60:
                return 70.0
            elif count >= 40:
                return 60.0
            elif count >= 25:
                return 50.0
            elif count >= 15:
                return 35.0
            else:
                return 20.0
        except Exception:
            return 50.0

    def get_limit_up_count(self) -> int:
        """涨停家数（从 ths_hot_reason 推算）"""
        try:
            hot = self.get_hot_stocks()
            return len(hot) if not hot.empty else 0
        except Exception:
            return 0

    def get_break_rate(self) -> float:
        """
        炸板率（估算）
        需要涨停+曾涨停但收盘未封住的数据，ths_hot_reason 只提供涨停。
        用默认估算值。
        """
        # 无法从现有 API 直接获取炸板率
        # 返回默认值，规则引擎会据此给中性评分
        return 20.0

    def get_leader_height(self) -> int:
        """
        连板高度（从 ths_hot_reason 推算）
        简化：取涨幅最大的标的，用涨幅/10近似连板数
        """
        try:
            hot = self.get_hot_stocks()
            if hot.empty:
                return 0
            max_change = hot["涨幅%"].max() if "涨幅%" in hot.columns else 0
            # 近似：涨幅 > 20% 可能是 2 板，> 30% 可能是 3 板
            return max(1, int(max_change / 10))
        except Exception:
            return 0

    def get_stock_list_with_basics(self) -> list[dict]:
        """
        获取股票基础列表
        简化实现：从 ths_hot_reason 获取活跃股 + 基础估值
        全量扫描需要 mootdx 全市场行情，这里只取活跃股
        """
        try:
            hot = self.get_hot_stocks()
            if hot.empty:
                return []

            result = []
            for _, row in hot.iterrows():
                code = str(row.get("代码", ""))
                if not code or len(code) != 6:
                    continue
                result.append({
                    "symbol": code,
                    "name": str(row.get("名称", "")),
                    "industry": "",  # 需要单独查
                    "market_cap": 0,  # 需要 tencent_quote
                    "quarterly": {},
                    "valuation": {},
                    "kline": None,
                    "f10": {},
                })
            return result
        except Exception as e:
            logger.warning(f"股票列表获取失败: {e}")
            return []

    def enrich_stock(self, stock: dict, kline_count: int = 250) -> dict:
        """
        丰富单只股票的数据（按需调用，避免全量扫描）
        kline_count: K线根数，默认250根（约1年交易日）
        """
        code = stock.get("symbol", "")
        if not code:
            return stock

        # K线
        try:
            stock["kline"] = self.get_daily_kline(code, count=kline_count)
        except Exception:
            pass

        # 估值（腾讯财经 PE/PB/市值）
        try:
            val = self.get_valuation(code)
            if not val.empty:
                v = val.iloc[0].to_dict()
                stock["valuation"] = {
                    "pe_ttm": v.get("pe_ttm", 0),
                    "pb": v.get("pb", 0),
                    "mcap_yi": v.get("mcap_yi", 0),
                }
                stock["market_cap"] = v.get("mcap_yi", 0)
        except Exception:
            pass

        # 季报（mootdx + 新浪财报补全）
        try:
            stock["quarterly"] = self.get_quarterly(code)
        except Exception:
            pass

        # 行业（东财 push2 直连，跳过push2被block，仅用百度fallback）
        try:
            baidu = self.get_industry_from_baidu(code)
            industry = baidu.get("industry", [None])[0] if baidu.get("industry") else ""
            stock["industry"] = industry or ""
        except Exception:
            pass

        # F10
        try:
            f10_text = self.get_f10(code, "最新提示")
            stock["f10"] = {"announcements": self._parse_announcements(f10_text)}
        except Exception:
            pass

        # 资金面：主力资金流120日（跳过，push2his被block）
        # try:
        #     stock["fund_flow"] = self.get_fund_flow_120d(code)
        # except Exception:
        #     pass

        # 资金面：融资融券
        try:
            stock["margin_data"] = self.get_margin_data(code)
        except Exception:
            pass

        # 资金面：股东户数变化
        try:
            stock["holder_change"] = self.get_holder_change(code)
        except Exception:
            pass

        return stock

    def _parse_announcements(self, text: str) -> list[dict]:
        """从 F10 最新提示文本中解析公告标题"""
        if not text:
            return []
        announcements = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 5 and any(kw in line for kw in ["公告", "报告", "决议", "通知", "披露"]):
                announcements.append({"title": line[:60]})
        return announcements[:10]

    # ═══════════════════ 宏观数据（估算/默认） ═══════════════════

    def get_m2_growth(self) -> float:
        """M2增速 — a-stock-data 无此数据源，返回估算默认值"""
        # 实际值需从央行或 akshare 宏观数据获取
        return 8.5

    def get_pmi(self) -> float:
        """PMI — a-stock-data 无此数据源，返回估算默认值"""
        return 50.5

    def get_cpi(self) -> float:
        """CPI — a-stock-data 无此数据源，返回估算默认值"""
        return 2.0

    def get_gdp_growth(self) -> float:
        """GDP增速 — 估算默认值"""
        return 5.0

    def get_pe_percentile(self) -> float:
        """PE历史分位 — 需要历史PE数据，暂用估算"""
        return 50.0


# ─────────────────────────── 全局单例 ───────────────────────────

_fetcher_instance: DataFetcher | None = None


def get_fetcher() -> DataFetcher:
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = DataFetcher()
    return _fetcher_instance
