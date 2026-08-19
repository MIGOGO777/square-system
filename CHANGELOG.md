# 正方形系统 更新日志

## v2.2.1 (2026-05-23)

### 新增：集成 a-stock-data v3.1 数据源

| 数据源 | 方法 | 功能 |
|--------|------|------|
| 东财 push2his | `get_fund_flow_120d()` | 主力/大单/中单/小单 120日净流入 |
| 东财 datacenter | `get_margin_data()` | 融资余额/融券余额 |
| 东财 datacenter | `get_holder_change()` | 股东户数变化（筹码集中度） |
| 东财 datacenter | `get_dividend_history()` | 分红送转历史 |
| 东财 datacenter | `get_block_trade()` | 大宗交易（机构动向） |
| 百度股市通 | `get_industry_from_baidu()` | 行业/概念/地域三维分类 |
| 新浪财报三表 | `_fetch_sina_financials()` | 毛利率/负债率/营收同比增长率 |
| 东财 push2 | `get_stock_info()` 替换 | 替代 akshare，零依赖 |

### 修复

- **HIGH** `_adapt_quarterly()`: mootdx字段名不匹配（`income`→`zhuyingshouru`, `profit`→`jinglirun`）→ 用正确字段名
- **HIGH** `_adapt_quarterly()`: ROE/负债率全部为0 → 从 mootdx 原始数据估算
- **HIGH** `_fetch_sina_financials()`: 新浪API格式变更（`report_list`嵌套结构）→ 适配新格式
- **HIGH** `_fetch_sina_financials()`: 营收增长计算错误（累计数据对比）→ 改用 `item_tongbi` 同比字段
- **MEDIUM** `get_stock_info()`: akshare API不稳定 → 改用东财push2直连 + 百度股市通fallback
- **LOW** `enrich_stock()`: 只有5步数据丰富 → 新增3步（资金流/融资融券/股东户数）

### 效果

| 指标 | 集成前 | 集成后 |
|------|--------|--------|
| 商业质量评分 | 全部50分（无数据） | 32-51分（有梯度） |
| Markowitz夏普 | 1.232 | 2.035 |
| 主动发现 | 2条 | 14条 |
| 硬排除后剩余 | 72只 | 49只（数据更完整，排除更精准） |

---

## v2.2.0 (2026-05-23)

### 新增：11个数学模型

| 模型 | 文件 | 功能 |
|------|------|------|
| HMM | `src/regime/hmm_detector.py` | 3状态隐马尔可夫市场状态检测（BULL/BEAR/SIDEWAYS） |
| Markowitz | `src/pool/optimizer.py` | 最大夏普比率组合优化 |
| GARCH | `src/rules/risk.py` | GARCH(1,1)波动率聚集预测 |
| 协整套利 | `src/discovery/pairs_scanner.py` | Engle-Granger协整检验 + Ornstein-Uhlenbeck半衰期 |
| GBM漂移率 | `src/rules/trend.py` | 几何布朗运动漂移率 + Jarque-Bera正态检验 |
| Kelly公式 | `src/rules/risk.py` | 半Kelly最优仓位计算 |
| Monte Carlo | `src/synthesis/monte_carlo.py` | 10000次GBM路径模拟 → VaR/盈利概率/最大回撤 |
| Fama-French | `src/models/fama_french.py` | 5因子回归（Beta/SMB/HML/RMW/CMA）→ Alpha归因 |
| Black-Scholes | `src/models/black_scholes.py` | IV vs RV偏离检测 + Newton-Raphson隐含波动率反推 |
| Almgren-Chriss | `src/models/almgren_chriss.py` | 大单最优拆单执行（最小化市场冲击） |
| RL权重调整 | `src/models/rl_weight_adjuster.py` | 上下文赌博机动态调整维度权重 |
| 因果推断 | `src/models/causal_inference.py` | 简化SCM + DAG因果路径 + 反事实估计 |

### 新增：主动发现第6条路线

- 配对交易（协整套利）：同行业股票两两做协整检验，价差偏离>2σ时发出信号

### 新增：报告内容

- Monte Carlo价格模拟表（5日预期收益/VaR 95%/盈利概率/最大回撤P95）
- Fama-French因子归因表（Beta/SMB/HML/Alpha/R²）
- Black-Scholes波动率分析
- Almgren-Chriss最优执行建议
- 模型运行状态（成功/失败显性化）

### 修复

- **HIGH** engine.py: Markowitz数据被反事实推理覆盖 → 改为合并而非覆盖
- **MEDIUM** report.py: HMM转移矩阵标签可能错位 → 用state_labels动态查找索引
- **MEDIUM** pairs_scanner: 日期对齐用位置而非索引 → 优先DatetimeIndex对齐
- **MEDIUM** discovery: cross_validate丢失配对结构化数据 → 保留原始字段
- **LOW** optimizer.py: 零波动率返回0（被当作最优） → 改为返回惩罚值

### 重构

- `_find_col` 从7处重复提取到 `src/core/utils.py`
- risk.py/trend.py 阈值从硬编码改为从config.yaml读取
- EvalContext 新增 config 字段，支持传递配置到规则函数

### 配置

- config.yaml 新增模型配置节：monte_carlo/fama_french/black_scholes/almgren_chriss/rl_weight
- run.py dry-run 新增第二批模型检查

### 回测优化

新增 `backtest_v2.py` 优化版回测脚本，含13项改进：

**核心优化（默认开启）：**

| 优化项 | 效果 | 结论 |
|--------|------|------|
| 防追高（ret20>20%扣5分） | 胜率提升，避免追涨被套 | 有效 |
| 行业分散（同行业最多2只） | 超额收益提升 | 有效 |
| 120天预热期 | 指标更可靠 | 有效 |
| 月度止损-8%（含ATR动态版） | 保险机制 | 保留 |

**可信度改进：**

| 改进项 | 说明 |
|--------|------|
| 交易成本模型 | 滑点0.1% + 佣金0.05%，持仓不变时不计成本 |
| 扩大股票池 | 沪深300全部280只（可选中证500，共800只） |
| 样本外验证 | train(2024-01~2025-06) / test(2025-07~2026-04) |
| 申万行业分类 | 缓存到本地JSON，fallback到前缀映射 |
| 波动率倒数加权 | 可选替代Kelly，降低组合波动率 |
| HMM仓位反馈 | BEAR 50%/SIDEWAYS 80%/BULL 100% |
| 换仓频率可调 | monthly/biweekly/quarterly |
| 阈值网格搜索 | `backtest_calibration.py`，训练期拟合+测试期验证 |

**回测结果（280只沪深300，含交易成本）：**

| 指标 | 全量回测 | 训练期 | 测试期 |
|------|---------|--------|--------|
| 总收益 | +77.89% | +25.26% | +40.53% |
| 超额收益 | +33.87% | +11.51% | +16.83% |
| 年化收益 | +28.00% | +17.23% | +50.42% |
| 夏普比率 | 1.03 | 0.68 | 1.74 |
| 月胜率 | 57% | 53% | 60% |
| 最大回撤 | -12.32% | -12.32% | -7.53% |
| 交易成本 | 6.75% | — | — |

结论：训练期和测试期均跑赢基准，策略有统计显著的正期望值。

---

## v2.1.0 (2026-05-19)

### 初始版本

- 47条原子规则，6个维度（价值/行业/情绪/趋势/宏观/风险）
- 5条寻宝路线（逆向猎手/行业拐点/情绪错杀/催化剂/龙虎追踪）
- 反事实推理 + 情景推演
- 动态阈值市场状态检测
- 对接 a-stock-data v2.1 真实API
