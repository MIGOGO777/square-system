"""
报告生成器 — 正方形系统 v2.3

输出结构：
1. 市场状态（HMM概率+宏观温度+情绪阶段+钟摆位置）
2. 数学模型输出（GARCH波动率+Markowitz权重+GBM漂移率+协整配对）
3. 主动发现（六条路线的发现）
4. 候选池（含各维度分数+置信度+反事实分析+组合权重）
5. 情景推演（高开/平开/低开三种情景）
6. 风控状态
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def generate(self, result: dict) -> str:
        """
        生成 Markdown 报告

        Args:
            result: DecisionEngine.run() 的输出

        Returns:
            str: Markdown 格式报告
        """
        lines = []
        timestamp = result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))

        lines.append(f"# 正方形系统 v2.3 日报 — {timestamp}")
        lines.append("")

        # 1. 市场状态
        self._section_market_state(lines, result.get("market_state", {}))

        # 2. 数学模型输出
        self._section_models(lines, result)

        # 3. 主动发现
        self._section_discoveries(lines, result.get("discoveries", []))

        # 4. 候选池
        self._section_candidates(lines, result.get("candidates", []),
                                  result.get("counterfactuals", {}))

        # 5. 情景推演
        self._section_scenarios(lines, result.get("scenarios", {}))

        # 6. 风控状态
        self._section_risk_control(lines, result.get("candidates", []))

        # 7. 模型运行状态
        self._section_model_status(lines, result.get("model_status", {}))

        # 8. 规则统计
        self._section_rules_summary(lines, result.get("rules_summary", {}))

        return "\n".join(lines)

    def _section_market_state(self, lines: list[str], ms: dict):
        """市场状态部分"""
        lines.append("## 市场状态")
        lines.append("")

        regime = ms.get("regime", "SIDEWAYS")
        temp = ms.get("temperature", 50.0)
        temp_pct = ms.get("temperature_percentile", 50.0)
        pendulum = ms.get("pendulum_position", "中性")
        emotion_phase = ms.get("emotion_phase", "未知")
        confirmed_by = ms.get("confirmed_by", [])

        regime_emoji = {"BULL": "🐂", "BEAR": "🐻", "SIDEWAYS": "↔️"}.get(regime, "↔️")

        lines.append(f"- 宏观温度: {temp:.0f}/100（最近60天百分位: {temp_pct:.0f}%）→ {regime_emoji} {regime}")
        lines.append(f"- 情绪阶段: {emotion_phase}")
        lines.append(f"- 钟摆位置: {pendulum}")

        if confirmed_by:
            conf_str = " ".join(f"{k}✓" for k in confirmed_by)
            lines.append(f"- 三层确认: {conf_str} → 确认{regime}")
        else:
            lines.append("- 三层确认: 无确认 → SIDEWAYS（默认）")

        lines.append("")

    def _section_models(self, lines: list[str], result: dict):
        """数学模型输出部分"""
        lines.append("## 数学模型")
        lines.append("")

        # HMM状态概率（从market_state的judgments中提取）
        ms = result.get("market_state", {})
        hmm_found = False
        for j in ms.get("judgments", []):
            if j.get("rule_id") == "hmm_regime":
                meta = j.get("metadata", {})
                hmm_probs = meta.get("hmm_probs", {})
                hmm_transition = meta.get("hmm_transition", [])
                if hmm_probs:
                    lines.append("### HMM状态概率")
                    lines.append(f"- BULL: {hmm_probs.get('BULL', 0):.0%} | "
                                 f"BEAR: {hmm_probs.get('BEAR', 0):.0%} | "
                                 f"SIDEWAYS: {hmm_probs.get('SIDEWAYS', 0):.0%}")
                    lines.append(f"- {j.get('reason', '')}")
                    if hmm_transition and len(hmm_transition) >= 3:
                        labels = meta.get("hmm_state_labels", ["BULL", "BEAR", "SIDEWAYS"])
                        # 用实际标签找对应索引
                        bull_idx = labels.index("BULL") if "BULL" in labels else 0
                        bear_idx = labels.index("BEAR") if "BEAR" in labels else 1
                        lines.append(f"- 转移矩阵: BULL→BULL={hmm_transition[bull_idx][bull_idx]:.2f}, "
                                     f"BEAR→BEAR={hmm_transition[bear_idx][bear_idx]:.2f}")
                    lines.append("")
                    hmm_found = True
                break

        if not hmm_found:
            regime = ms.get("regime", "SIDEWAYS")
            temp_pct = ms.get("temperature_percentile", 50.0)
            lines.append("### 市场状态（传统检测器）")
            lines.append(f"- {regime}，温度百分位{temp_pct:.0f}%")
            lines.append("")

        # Markowitz组合优化
        optimization = result.get("optimization", {})
        if optimization:
            lines.append("### Markowitz组合优化")
            lines.append("| 标的 | 建议权重 | 预期年化 | 波动率 | 夏普 |")
            lines.append("|------|---------|---------|--------|------|")
            for sym, info in sorted(optimization.items(),
                                     key=lambda x: x[1].get("weight", 0), reverse=True):
                w = info.get("weight", 0)
                ret = info.get("expected_return", 0)
                vol = info.get("volatility", 0)
                sharpe = info.get("sharpe", 0)
                lines.append(f"| {sym} | {w:.1%} | {ret:.1%} | {vol:.1%} | {sharpe:.2f} |")
            lines.append("")

        # GARCH波动率（从候选标的的judgments中提取）
        candidates = result.get("candidates", [])
        garch_rows = []
        for c in candidates:
            for j in c.get("judgments", []):
                if j.get("rule_id") == "sys_06":
                    pred_vol = j.get("metadata", {}).get("predicted_vol_5d", 0)
                    garch_rows.append((c.get("symbol", ""), c.get("name", ""),
                                       pred_vol, j.get("score", 0), j.get("reason", "")))
                    break
        if garch_rows:
            lines.append("### GARCH波动率预测")
            for sym, name, vol, score, reason in garch_rows[:5]:
                lines.append(f"- {sym} {name}: 5日预测波动率={vol:.2f}%，风险分={score:.0f}")
            lines.append("")

        # Monte Carlo价格模拟
        mc = result.get("monte_carlo", {})
        if mc:
            lines.append("### Monte Carlo价格模拟")
            lines.append("| 标的 | 5日预期收益 | VaR 95% | 盈利概率 | 最大回撤P95 |")
            lines.append("|------|-----------|---------|---------|------------|")
            for sym, info in sorted(mc.items(),
                                     key=lambda x: x[1].get("profit_prob", 0), reverse=True):
                er = info.get("expected_return", 0)
                var = info.get("var_95", 0)
                pp = info.get("profit_prob", 0)
                mdd = info.get("max_drawdown_p95", 0)
                lines.append(f"| {sym} | {er:+.1f}% | {var:.1f}% | {pp:.0%} | {mdd:.1f}% |")
            lines.append("")

        # Fama-French因子归因
        ff = result.get("fama_french", {})
        if ff:
            lines.append("### 因子归因（Fama-French）")
            lines.append("| 标的 | Beta | SMB | HML | Alpha(年化) | R² |")
            lines.append("|------|------|-----|-----|-----------|-----|")
            for sym, info in ff.items():
                beta = info.get("beta", 0)
                smb = info.get("smb", 0)
                hml = info.get("hml", 0)
                alpha = info.get("alpha_annual", 0)
                r2 = info.get("r_squared", 0)
                lines.append(f"| {sym} | {beta:.2f} | {smb:.2f} | {hml:.2f} | {alpha:+.1f}% | {r2:.2f} |")
            lines.append("")

        # Black-Scholes波动率分析
        bs = result.get("black_scholes", {})
        if bs:
            lines.append("### 波动率分析（Black-Scholes）")
            for sym, info in bs.items():
                rv = info.get("rv_20d", 0)
                signal = info.get("signal", "rv_only")
                reason = info.get("reason", "")
                lines.append(f"- {sym}: 20日RV={rv:.1f}%，{reason}")
            lines.append("")

        # Almgren-Chriss最优执行
        ac = result.get("almgren_chriss", {})
        if ac:
            lines.append("### 最优执行（Almgren-Chriss）")
            for sym, info in ac.items():
                days = info.get("suggested_days", 1)
                n_trades = info.get("n_trades", 1)
                impact = info.get("expected_impact_pct", 0)
                lines.append(f"- {sym}: 建议分{n_trades}笔/{days}天执行，预期冲击{impact:.3f}%")
            lines.append("")

        # 协整配对
        discoveries = result.get("discoveries", [])
        pairs = [d for d in discoveries if "pairs_trading" in d.get("routes", [])]
        if pairs:
            lines.append("### 协整配对信号")
            for p in pairs[:3]:
                lines.append(f"- {p.get('name', '')}: {p.get('reason', '')}")
            lines.append("")

    def _section_discoveries(self, lines: list[str], discoveries: list[dict]):
        """主动发现部分"""
        lines.append("## 主动发现")
        lines.append("")

        if not discoveries:
            lines.append("今日无主动发现")
            lines.append("")
            return

        # 按路线分组
        by_route: dict[str, list[dict]] = {}
        for d in discoveries:
            routes = d.get("routes", ["unknown"])
            for r in routes:
                if r not in by_route:
                    by_route[r] = []
                by_route[r].append(d)

        route_names = {
            "contrarian": "逆向猎手",
            "industry_shift": "行业拐点",
            "sentiment_miss": "情绪错杀",
            "catalyst": "催化剂猎手",
            "dragon_tiger": "龙虎追踪",
            "pairs_trading": "配对交易",
        }

        for route, items in by_route.items():
            name = route_names.get(route, route)
            lines.append(f"### {name}发现")
            for item in items[:5]:
                symbol = item.get("symbol", "")
                stock_name = item.get("name", "")
                reason = item.get("reason", "")
                route_count = item.get("route_count", 1)
                cross = f"（{route_count}条路线交叉验证）" if route_count > 1 else ""
                lines.append(f"- {symbol} {stock_name}: {reason}{cross}")
            lines.append("")

    def _section_candidates(self, lines: list[str], candidates: list[dict],
                            counterfactuals: dict):
        """候选池部分"""
        lines.append("## 候选池")
        lines.append("")

        if not candidates:
            lines.append("今日无候选标的")
            lines.append("")
            return

        # 表头
        lines.append("| 代码 | 名称 | 商业质量 | 估值安全 | 趋势 | 情绪 | 综合分 | 置信度 |")
        lines.append("|------|------|---------|---------|------|------|--------|--------|")

        for c in candidates:
            symbol = c.get("symbol", "")
            name = c.get("name", "")[:4]
            comp_score = c.get("composite_score", 0)
            comp_conf = c.get("composite_confidence", 0)

            # 各维度分数
            dims = c.get("dimension_scores", {})
            value_s = dims.get("value", "-")
            industry_s = dims.get("industry", "-")
            emotion_s = dims.get("emotion", "-")
            trend_s = dims.get("trend", "-")

            def fmt(v):
                return f"{v:.0f}" if isinstance(v, (int, float)) else str(v)

            lines.append(f"| {symbol} | {name} | {fmt(value_s)} | {fmt(industry_s)} | "
                         f"{fmt(trend_s)} | {fmt(emotion_s)} | {comp_score:.1f} | {comp_conf:.0%} |")

        lines.append("")

        # 反事实分析
        lines.append("### 反事实分析")
        lines.append("")
        for c in candidates[:5]:
            symbol = c.get("symbol", "")
            name = c.get("name", "")
            cf = counterfactuals.get(symbol, {})
            if not cf:
                continue

            lines.append(f"**{symbol} {name}**")
            lines.append(f"- 如果错了: {cf.get('failure_mode', '未知')}")
            lines.append(f"- 共识错在: {cf.get('consensus_error', '未知')}")
            lines.append(f"- 机会成本: {cf.get('opportunity_cost', '未知')}")
            verdict = cf.get("verdict", "unknown")
            verdict_str = {"pass": "✅ 通过", "caution": "⚠️ 谨慎", "reject": "❌ 拒绝"}.get(verdict, verdict)
            lines.append(f"- 裁决: {verdict_str}")
            lines.append("")

    def _section_scenarios(self, lines: list[str], scenarios: dict):
        """情景推演部分"""
        lines.append("## 情景推演")
        lines.append("")

        for key, label in [("scenario_a", "情景A"), ("scenario_b", "情景B"), ("scenario_c", "情景C")]:
            sc = scenarios.get(key, {})
            if not sc:
                continue
            condition = sc.get("condition", "")
            action = sc.get("action", "")
            details = sc.get("details", [])

            lines.append(f"**{label}（{condition}）**")
            lines.append(f"- {action}")
            for d in details[:3]:
                lines.append(f"  - {d}")
            lines.append("")

        summary = scenarios.get("summary", "")
        if summary:
            lines.append(f"**综合建议**: {summary}")
            lines.append("")

    def _section_risk_control(self, lines: list[str], candidates: list[dict]):
        """风控状态部分"""
        lines.append("## 风控状态")
        lines.append("")

        # 汇总风险维度分数
        risk_scores = []
        for c in candidates:
            dims = c.get("dimension_scores", {})
            risk = dims.get("risk")
            if risk is not None:
                risk_scores.append(risk)

        if risk_scores:
            avg_risk = sum(risk_scores) / len(risk_scores)
            min_risk = min(risk_scores)
            lines.append(f"- 候选池平均风险分: {avg_risk:.0f}")
            lines.append(f"- 最低风险分: {min_risk:.0f}")

            if min_risk < 25:
                lines.append("- ⚠️ 存在高风险标的，建议谨慎")
            elif avg_risk < 40:
                lines.append("- ⚠️ 整体风险偏高，控制仓位")
            else:
                lines.append("- ✅ 风险可控")
        else:
            lines.append("- 无风险数据")

        lines.append("")

    def _section_model_status(self, lines: list[str], model_status: dict):
        """模型运行状态部分（规则12：失败必须显性化）"""
        if not model_status:
            return

        lines.append("## 模型运行状态")
        lines.append("")

        failed = []
        for name, info in model_status.items():
            ok = info.get("ok", False)
            if ok:
                count = info.get("count")
                note = info.get("note", "")
                detail = f"（{count}只标的）" if count else (f"（{note}）" if note else "")
                lines.append(f"- {name}: ✅ 正常{detail}")
            else:
                error = info.get("error", "未知原因")
                lines.append(f"- {name}: ❌ 失败 — {error}")
                failed.append(name)

        if failed:
            lines.append("")
            lines.append(f"⚠️ {len(failed)}个模型未能运行，报告中对应部分可能缺失")

        lines.append("")

    def _section_rules_summary(self, lines: list[str], summary: dict):
        """规则统计部分"""
        lines.append("## 系统统计")
        lines.append("")

        total = summary.get("total_rules", 0)
        dimensions = summary.get("dimensions", [])
        thinkers = summary.get("thinkers", [])

        lines.append(f"- 注册规则数: {total}")
        lines.append(f"- 覆盖维度: {', '.join(dimensions)}")
        lines.append(f"- 来源大师: {', '.join(thinkers)}")
        lines.append("")
        lines.append("---")
        lines.append("*正方形系统 v2.3 — 11个数学模型 × 原子化规则 × 主动发现 × 反事实推理*")
