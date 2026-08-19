"""
HMM市场状态检测器 — 基于隐马尔可夫模型的regime detection

替代固定阈值的温度百分位判定，用概率模型学习市场状态转移。

观测特征：
- emotion_temp (情绪温度)
- north_flow (北向资金净流入)
- market_change_pct (市场涨跌幅)
- leader_height (连板高度)

输出：
- state_probabilities: [P(BULL), P(BEAR), P(SIDEWAYS)]
- transition_matrix: 3x3 状态转移概率
- 当前最可能状态及其概率
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.signal import AtomicJudgment, MarketState
from src.regime.detector import RegimeDetector

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    HMM市场状态检测器

    当历史数据不足或hmmlearn不可用时，fallback到传统RegimeDetector。
    """

    def __init__(self, n_states: int = 3, min_history: int = 30,
                 n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.min_history = min_history
        self.n_iter = n_iter
        self.random_state = random_state
        self._fallback = RegimeDetector()
        self._hmm_available = self._check_hmm()

    def _check_hmm(self) -> bool:
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401
            return True
        except ImportError:
            logger.warning("hmmlearn不可用，HMM检测器将fallback到传统检测器")
            return False

    def detect(self, market_data: dict,
               history_temps: list[float] | None = None) -> MarketState:
        """
        检测市场状态（与RegimeDetector.detect签名一致）

        优先使用HMM，失败或数据不足时fallback到传统检测器。
        """
        if not self._hmm_available:
            return self._fallback.detect(market_data, history_temps)

        observation_history = market_data.get("observation_history", [])

        if len(observation_history) < self.min_history:
            logger.info(f"HMM历史数据不足({len(observation_history)}<{self.min_history})，使用传统检测器")
            return self._fallback.detect(market_data, history_temps)

        try:
            return self._detect_hmm(market_data, observation_history)
        except Exception as e:
            logger.warning(f"HMM检测失败: {e}，fallback到传统检测器")
            return self._fallback.detect(market_data, history_temps)

    def _detect_hmm(self, market_data: dict,
                    observation_history: list[dict]) -> MarketState:
        """用HMM检测市场状态"""
        from hmmlearn.hmm import GaussianHMM

        # 构建观测矩阵：每行 [emotion_temp, north_flow, market_change, leader_height]
        features = []
        for obs in observation_history:
            features.append([
                obs.get("emotion_temp", 50.0),
                obs.get("north_flow", 0.0),
                obs.get("market_change_pct", 0.0),
                obs.get("leader_height", 0),
            ])
        X = np.array(features, dtype=np.float64)

        # z-score标准化
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0  # 防止除零
        X_norm = (X - self._mean) / self._std

        # 拟合HMM
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(X_norm)

        # 构建当前观测向量
        current = self._extract_observation(market_data)
        current_norm = (current - self._mean) / self._std

        # 预测状态概率
        probs = model.predict_proba(current_norm.reshape(1, -1))[0]
        most_likely = int(np.argmax(probs))

        # 状态映射：用emotion_temp均值区分BULL/BEAR/SIDEWAYS
        state_labels = self._map_states(model, self._mean, self._std)
        current_state = state_labels[most_likely]

        # 转移矩阵
        transmat = model.transmat_.tolist()

        # 构建MarketState
        temperature = market_data.get("emotion_temp", 50.0)
        p_bull = float(probs[state_labels.index("BULL")]) if "BULL" in state_labels else 0.0
        p_bear = float(probs[state_labels.index("BEAR")]) if "BEAR" in state_labels else 0.0
        percentile = p_bull * 100

        pendulum = self._calc_pendulum(percentile)
        emotion_phase = market_data.get("emotion_phase", "未知")

        # HMM结果作为AtomicJudgment
        hmm_judgment = AtomicJudgment(
            rule_id="hmm_regime", rule_name="HMM状态检测", thinker="quantitative",
            dimension="risk",
            score=max(0, min(100, p_bull * 100)),
            confidence=float(max(probs)),
            data_quality=0.7,
            direction="BUY" if current_state == "BULL" else ("SELL" if current_state == "BEAR" else "HOLD"),
            reason=f"HMM状态: {current_state}(概率{max(probs):.0%})，BULL={p_bull:.0%} BEAR={p_bear:.0%}",
            metadata={
                "hmm_state": current_state,
                "hmm_probs": {"BULL": round(p_bull, 3), "BEAR": round(p_bear, 3),
                              "SIDEWAYS": round(1 - p_bull - p_bear, 3)},
                "hmm_transition": transmat,
                "hmm_state_labels": state_labels,
            },
        )

        return MarketState(
            regime=current_state,
            temperature=temperature,
            temperature_percentile=round(percentile, 1),
            pendulum_position=pendulum,
            emotion_phase=emotion_phase,
            judgments=[hmm_judgment],
            confirmed_by=["hmm"],
        )

    def _extract_observation(self, market_data: dict) -> np.ndarray:
        """从market_data提取观测向量"""
        # 北向资金取净流入
        north_flow = 0.0
        nf = market_data.get("north_flow")
        if nf is not None and hasattr(nf, 'empty') and not nf.empty:
            for col in ("net_flow", "净流入", "north_net"):
                if col in nf.columns:
                    north_flow = float(nf[col].iloc[-1])
                    break

        return np.array([
            market_data.get("emotion_temp", 50.0),
            north_flow,
            market_data.get("market_change_pct", 0.0),
            market_data.get("leader_height", 0),
        ], dtype=np.float64)

    def _map_states(self, model, mean: np.ndarray, std: np.ndarray) -> list[str]:
        """
        将HMM的隐状态映射到BULL/BEAR/SIDEWAYS

        方法：看各状态在emotion_temp特征上的均值
        均值最高→BULL，最低→BEAR，中间→SIDEWAYS
        """
        # 用模型的均值反推各状态在emotion_temp上的实际均值
        # model.means_ 是标准化后的均值，需要反标准化
        state_means = []
        for i in range(self.n_states):
            actual_mean = model.means_[i][0] * std[0] + mean[0]  # emotion_temp维度
            state_means.append((i, actual_mean))

        state_means.sort(key=lambda x: x[1])  # 升序

        labels = [""] * self.n_states
        if self.n_states == 3:
            labels[state_means[0][0]] = "BEAR"
            labels[state_means[1][0]] = "SIDEWAYS"
            labels[state_means[2][0]] = "BULL"
        else:
            # 通用映射
            for rank, (idx, _) in enumerate(state_means):
                if rank < self.n_states // 3:
                    labels[idx] = "BEAR"
                elif rank < 2 * self.n_states // 3:
                    labels[idx] = "SIDEWAYS"
                else:
                    labels[idx] = "BULL"

        return labels

    def _calc_pendulum(self, percentile: float) -> str:
        """钟摆位置描述"""
        if percentile <= 15:
            return "极度偏左（极度悲观）"
        elif percentile <= 30:
            return "偏左（悲观）"
        elif percentile <= 45:
            return "温和偏左"
        elif percentile <= 55:
            return "中性"
        elif percentile <= 70:
            return "温和偏右"
        elif percentile <= 85:
            return "偏右（乐观）"
        else:
            return "极度偏右（极度乐观）"


def save_observation(market_data: dict, cache_dir: str = "data/cache") -> None:
    """保存当日观测数据到CSV，供HMM使用"""
    try:
        from pathlib import Path
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        csv_path = cache_path / "observation_history.csv"

        # 提取观测
        north_flow = 0.0
        nf = market_data.get("north_flow")
        if nf is not None and hasattr(nf, 'empty') and not nf.empty:
            for col in ("net_flow", "净流入", "north_net"):
                if col in nf.columns:
                    north_flow = float(nf[col].iloc[-1])
                    break

        row = {
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "emotion_temp": market_data.get("emotion_temp", 50.0),
            "north_flow": north_flow,
            "market_change_pct": market_data.get("market_change_pct", 0.0),
            "leader_height": market_data.get("leader_height", 0),
        }

        df_new = pd.DataFrame([row])

        if csv_path.exists():
            df_old = pd.read_csv(csv_path)
            # 去重：同一天只保留最新
            df_old = df_old[df_old["date"] != row["date"]]
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        # 只保留最近120天
        df_all = df_all.tail(120)
        df_all.to_csv(csv_path, index=False)
        logger.debug(f"观测数据已保存到 {csv_path}")
    except Exception as e:
        logger.warning(f"保存观测数据失败: {e}")


def load_observation_history(cache_dir: str = "data/cache") -> list[dict]:
    """加载历史观测数据"""
    try:
        csv_path = Path(cache_dir) / "observation_history.csv"
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path)
        return df.to_dict("records")
    except Exception as e:
        logger.warning(f"加载观测历史失败: {e}")
        return []
