"""
Alpha Strategy 2: Volatility-Regime Switched Mean Reversion
Trades statistical price Z-score mean-reversion exclusively when the market is in a stationary consolidation regime (Hurst < 0.48).
Unified for batch backtesting and live streaming on_bar execution.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata
from src.features.microstructure import MicrostructureFeatureExtractor
from src.core.events import BarEvent, SignalEvent, SignalType


class AlphaRegimeAdaptiveMR(BaseStrategy, BaseHypothesis):
    """
    Hypothesis 2: Regime-Gated Statistical Mean Reversion.
    """

    DEFAULT_METADATA = HypothesisMetadata(
        hypothesis_id="ALPHA_02_REGIME_ADAPTIVE_MR",
        name="Regime-Gated Statistical Z-Score Mean Reversion",
        category="REGIME_SWITCHING",
        economic_rationale=(
            "Assets mean-revert reliably during low-volatility consolidation, but trend violently during regime shifts. "
            "Gating statistical Z-Score entries with the Hurst Exponent (H < 0.48) filters out false tops and bottoms."
        ),
        target_instruments=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        timeframe="5m",
    )

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or self.DEFAULT_METADATA
        BaseHypothesis.__init__(self, metadata=meta, parameters=parameters)
        BaseStrategy.__init__(self, strategy_id=meta.hypothesis_id, parameters=parameters)

        self.zscore_entry = self.parameters.get("zscore_entry", 2.0)
        self.zscore_exit = self.parameters.get("zscore_exit", 0.5)
        self.rolling_window = self.parameters.get("rolling_window", 20)
        self.hurst_threshold = self.parameters.get("hurst_threshold", 0.48)
        self.square_off_time = self.parameters.get("square_off_time", "15:15:00")

        # Live state tracking
        self._recent_closes: List[float] = []
        self._active_position = 0.0

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "zscore_entry": [1.8, 2.0, 2.3],
            "zscore_exit": [0.3, 0.5],
            "rolling_window": [15, 20, 30],
            "hurst_threshold": [0.45, 0.48],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        extractor = MicrostructureFeatureExtractor()

        roll_mean = df_out["close"].rolling(window=self.rolling_window).mean()
        roll_std = df_out["close"].rolling(window=self.rolling_window).std()
        df_out["zscore"] = (df_out["close"] - roll_mean) / roll_std.replace(0, np.nan)

        hurst_series = []
        close_vals = df_out["close"].values
        for i in range(len(df_out)):
            if i < self.rolling_window * 2:
                hurst_series.append(0.50)
            else:
                chunk = close_vals[i - self.rolling_window * 2 : i]
                hurst_series.append(extractor.calculate_hurst_exponent(pd.Series(chunk)))
        df_out["hurst"] = hurst_series

        time_strings = df_out.index.strftime("%H:%M:%S")
        is_square_off = time_strings >= self.square_off_time
        is_mean_reverting = df_out["hurst"] <= self.hurst_threshold

        signal = np.zeros(len(df_out))
        position = 0.0

        for i in range(len(df_out)):
            if is_square_off[i]:
                position = 0.0
            elif is_mean_reverting.iloc[i]:
                z = df_out["zscore"].iloc[i]
                if np.isnan(z):
                    continue
                if z <= -self.zscore_entry and position <= 0:
                    position = 1.0
                elif z >= self.zscore_entry and position >= 0:
                    position = -1.0
                elif position > 0 and z >= -self.zscore_exit:
                    position = 0.0
                elif position < 0 and z <= self.zscore_exit:
                    position = 0.0
            else:
                if position != 0.0:
                    position = 0.0
            signal[i] = position

        df_out["signal"] = signal
        return df_out

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """Incremental live bar evaluator."""
        bar_time_str = bar.timestamp.strftime("%H:%M:%S")
        if bar_time_str >= self.square_off_time:
            if self._active_position != 0.0:
                self._active_position = 0.0
                return SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    direction=0.0,
                    strength=0.0,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "SQUARE_OFF"},
                )
            return None

        self._recent_closes.append(bar.close)
        if len(self._recent_closes) > self.rolling_window * 2:
            self._recent_closes.pop(0)

        if len(self._recent_closes) < self.rolling_window:
            return None

        recent_slice = np.array(self._recent_closes[-self.rolling_window:])
        mean_val = np.mean(recent_slice)
        std_val = np.std(recent_slice)
        if std_val == 0:
            return None

        zscore = (bar.close - mean_val) / std_val

        # Hurst Exponent on window
        extractor = MicrostructureFeatureExtractor()
        hurst = extractor.calculate_hurst_exponent(pd.Series(self._recent_closes))

        if hurst > self.hurst_threshold:
            # Trending regime - exit mean reversion
            if self._active_position != 0.0:
                self._active_position = 0.0
                return SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    direction=0.0,
                    strength=0.0,
                    strategy_id=self.strategy_id,
                    metadata={"reason": "REGIME_SHIFT_TREND"},
                )
            return None

        # Mean Reversion entries & exits
        if zscore <= -self.zscore_entry and self._active_position <= 0:
            self._active_position = 1.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=1.0,
                strength=1.0,
                strategy_id=self.strategy_id,
                stop_loss=bar.close - (std_val * 1.5),
                take_profit=mean_val,
                metadata={"zscore": zscore, "hurst": hurst},
            )
        elif zscore >= self.zscore_entry and self._active_position >= 0:
            self._active_position = -1.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=-1.0,
                strength=1.0,
                strategy_id=self.strategy_id,
                stop_loss=bar.close + (std_val * 1.5),
                take_profit=mean_val,
                metadata={"zscore": zscore, "hurst": hurst},
            )
        elif self._active_position > 0 and zscore >= -self.zscore_exit:
            self._active_position = 0.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=0.0,
                strength=0.0,
                strategy_id=self.strategy_id,
                metadata={"reason": "MEAN_REVERTED"},
            )
        elif self._active_position < 0 and zscore <= self.zscore_exit:
            self._active_position = 0.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=0.0,
                strength=0.0,
                strategy_id=self.strategy_id,
                metadata={"reason": "MEAN_REVERTED"},
            )

        return None
