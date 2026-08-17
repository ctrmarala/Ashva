"""
Alpha Strategy 2: Volatility-Regime Switched Mean Reversion
Trades statistical price Z-score mean-reversion exclusively when the market is in a stationary consolidation regime (Hurst < 0.48).
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

        # 1. Rolling Mean & Standard Deviation for Z-Score
        roll_mean = df_out["close"].rolling(window=self.rolling_window).mean()
        roll_std = df_out["close"].rolling(window=self.rolling_window).std()
        df_out["zscore"] = (df_out["close"] - roll_mean) / roll_std.replace(0, np.nan)

        # 2. Rolling Hurst Exponent (approximated on rolling price window)
        hurst_series = []
        close_vals = df_out["close"].values
        for i in range(len(df_out)):
            if i < self.rolling_window * 2:
                hurst_series.append(0.50)
            else:
                chunk = close_vals[i - self.rolling_window * 2 : i]
                hurst_series.append(extractor.calculate_hurst_exponent(pd.Series(chunk)))
        df_out["hurst"] = hurst_series

        # 3. Time Filtering
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
                # Oversold in chop -> Buy
                if z <= -self.zscore_entry and position <= 0:
                    position = 1.0
                # Overbought in chop -> Sell / Short
                elif z >= self.zscore_entry and position >= 0:
                    position = -1.0
                # Exit back towards mean
                elif position > 0 and z >= -self.zscore_exit:
                    position = 0.0
                elif position < 0 and z <= self.zscore_exit:
                    position = 0.0
            else:
                # If regime shifts to trending, exit mean reversion position immediately
                if position != 0.0:
                    position = 0.0

            signal[i] = position

        df_out["signal"] = signal
        return df_out

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        return None
