---
name: square-v2
description: 正方形系统 v2.2 — 11个数学模型 × 原子化规则 × 主动发现 × 反事实推理（对接 a-stock-data v2.1）
---

# 正方形系统 v2.2

A 股分析核心引擎 v2.2。11个数学模型、54条原子规则、6条寻宝路线、反事实推理、动态阈值。对接 a-stock-data v2.1 真实 API。

## 触发词

- 正方形系统 v2 / 方阵 v2
- 主动发现 / 寻宝 / 找金子
- 候选池 / 股票池
- 情景推演 / 情景分析
- 反事实分析

## 数据路径

报告：`正方形系统/data/reports/LATEST.md`
运行：`python 正方形系统/run.py`
测试：`python 正方形系统/run.py --dry-run`

## 使用方式

### 1. 读取报告

直接读取 `正方形系统/data/reports/LATEST.md`，获取：
- 市场状态（宏观温度+情绪阶段+钟摆位置）
- 主动发现（五条路线的发现）
- 候选池（含各维度分数+置信度+反事实分析）
- 情景推演（高开/平开/低开）

### 2. 生成报告

当报告不存在或过期时：
```bash
cd 正方形系统 && python run.py
```

### 3. 深度分析

当报告不足以回答问题时，可单独调用各模块：
- 市场状态：`src/regime/detector.py`
- 主动发现：`src/discovery/` (5条路线)
- 规则引擎：`src/rules/` (6个维度47条规则)
- 反事实推理：`src/synthesis/counterfactual.py`
- 情景推演：`src/synthesis/scenario.py`

## 架构

```
数据层 (fetcher + quality)
  ↓
规则层 (47条原子规则, 6维度: value/industry/emotion/trend/macro/risk)
  ↓
发现层 (5条路线: 逆向/行业拐点/情绪错杀/催化剂/龙虎)
  ↓
候选池 (硬排除→行业筛选→多维评分→反事实淘汰)
  ↓
合成层 (反事实推理 + 情景推演)
  ↓
报告层 (Markdown报告)
```

## 大师映射

| 大师 | 维度 | 规则数 |
|------|------|--------|
| 段永平 | value | 4 |
| 邱国鹭 | value/industry | 7 |
| Klarman | value/risk | 5 |
| Livermore | trend | 5 |
| Marks | macro | 4 |
| Dalio | macro | 2 |
| 炒股养家 | emotion | 4 |
| 乌合之众 | emotion | 3 |
| 交易系统 | risk | 5 |
