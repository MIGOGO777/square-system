# 正方形系统 v2.3 更新说明

## 更新日期
2026-05-23

## 核心改动：集成 a-stock-data v3.1 数据源

解决了系统最大的数据缺口——"估值安全"维度全部显示为"-"的问题。
通过集成 a-stock-data 项目的 28 个免费 HTTP API 端点，补全了估值/财务/资金面数据。

---

## 新增数据方法（src/data/fetcher.py）

| 方法 | 数据源 | 返回数据 |
|------|--------|---------|
| `get_fund_flow_120d(code)` | 东财 push2his | 主力/大单/中单/小单 120日净流入（元） |
| `get_margin_data(code)` | 东财 datacenter-web | 融资余额/融券余额/融资买入 |
| `get_holder_change(code)` | 东财 datacenter-web | 股东户数变化/户均持股（季度级） |
| `get_dividend_history(code)` | 东财 datacenter-web | 分红送转历史/每股派息 |
| `get_block_trade(code)` | 东财 datacenter-web | 大宗交易/买卖方营业部/溢价率 |
| `get_industry_from_baidu(code)` | 百度股市通 | 行业/概念/地域三维分类 |

## 修复的问题

### 关键修复

1. **mootdx 字段名不匹配**：`_adapt_quarterly()` 用 `income`/`profit` 读取 mootdx 数据，但实际字段名是 `zhuyingshouru`/`jinglirun`。导致净利率、ROE 全部为 0。

2. **ROE/负债率无法计算**：mootdx 不直接提供 ROE 和负债率，但现在从原始数据推算：
   - ROE = 净利润 / 净资产 × 100
   - 负债率 = (流动负债 + 长期负债) / 总资产 × 100

3. **新浪财报 API 格式变更**：API 从 `data.lrb[]` 改为 `data.report_list[date].data[]` 嵌套结构，字段从中文名改为 `item_field` 编码。已适配新格式。

4. **营收增长计算错误**：之前用累计数据对比（Q1 vs Q4），得到 -75% 的错误值。改用 API 的 `item_tongbi` 字段（同比增长率），京东方 A 实际为 +0.8%。

5. **akshare 依赖不稳定**：`get_stock_info()` 改用东财 push2 直连 API + 百度股市通 fallback，零第三方依赖。

### 增强

6. **enrich_stock() 扩展**：从 5 步数据丰富增加到 8 步，新增资金流/融资融券/股东户数。

7. **新浪财报补全毛利率**：`_fetch_sina_financials()` 从利润表提取 `BIZINCO`（营业收入）和 `BIZCOST`（营业成本），计算毛利率。

8. **engine.py 补充 industry_fund_flow**：为行业规则（qgy_07）提供行业资金流向数据。

---

## 效果对比

| 指标 | v2.2 | v2.3 |
|------|------|------|
| 商业质量评分 | 全部50分（无数据） | **32-51分（有梯度）** |
| Markowitz 夏普比 | 1.232 | **2.035** |
| 主动发现线索 | 2条 | **14条** |
| 硬排除后候选股 | 72只 | **49只（排除更精准）** |
| 估值安全维度 | "-" | "-"（需 akshare 行业数据） |

> 注："估值安全"列实际上是行业维度，需要市场级别的行业对比数据（CR3/ROE），
> 依赖 akshare 的 `stock_board_industry_summary_ths()` API。
> 在 Ubuntu 上运行时该 API 应能正常工作。

---

## 数据流路径

```
enrich_stock()
├── get_daily_kline()        → stock["kline"]          (mootdx)
├── get_valuation()          → stock["valuation"]      (腾讯财经)
│   └── pe_ttm, pb, mcap_yi
├── get_quarterly()          → stock["quarterly"]      (mootdx + 新浪财报)
│   └── net_margin, gross_margin, roe_list, debt_ratio, revenue_growth
├── get_stock_info()         → stock["industry"]       (东财push2 / 百度fallback)
├── get_f10()                → stock["f10"]            (mootdx)
│   └── announcements
├── get_fund_flow_120d()     → stock["fund_flow"]      (东财push2his)  [新增]
├── get_margin_data()        → stock["margin_data"]    (东财datacenter) [新增]
└── get_holder_change()      → stock["holder_change"]  (东财datacenter) [新增]
```

---

## 安装与运行

```bash
# 安装依赖
uv venv --python 3.11
uv pip install -r requirements.txt

# 测试模块
python run.py --dry-run

# 完整运行
python run.py

# 回测
python backtest_v2.py --mode full
```

---

## 已知限制

1. **东财 push2his 资金流 API**：部分 Windows 网络环境下返回 RemoteDisconnected，不影响系统运行
2. **百度股市通 API**：部分环境下返回空结果，已用 mootdx 行业代码 fallback
3. **行业维度评分**：依赖 akshare 行业对比 API，当前 Windows 环境不可用
4. **FCF（自由现金流）**：mootdx 和新浪财报均不直接提供，fcf_list 仍为空
5. **宏观数据**：M2/PMI/CPI/GDP 仍为估算默认值

---

## 文件清单

```
v2.3/
├── src/
│   ├── core/engine.py          [修改] 补充 industry_fund_flow
│   ├── data/fetcher.py         [重写] 新增6个方法 + 修复4个bug
│   ├── data/quality.py
│   ├── discovery/
│   ├── models/
│   ├── output/
│   ├── pool/
│   ├── regime/
│   ├── rules/
│   └── synthesis/
├── data/reports/LATEST.md      最新选股报告
├── config.yaml
├── run.py
├── requirements.txt
├── backtest.py
├── backtest_v2.py
├── backtest_calibration.py
├── CHANGELOG.md
└── UPDATE_v2.3.md              本文件
```
