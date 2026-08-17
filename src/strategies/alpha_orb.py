"""
Alpha Strategy 1: Institutional Opening Range & Anchored VWAP Momentum Breakout
Exploits institutional morning flow (09:15-09:45 AM) accompanied by volume surges and VWAP trend confirmation.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata
from src.features.microstructure import MicrostructureFeatureExtractor
from src.core.events import BarEvent, SignalEvent, SignalType


class AlphaInstitutionalORB(BaseStrategy, BaseHypothesis):
    """
    Hypothesis 1: 09:15-09:45 AM Institutional Basket Flow & VWAP Trend Breakout.
    """

    DEFAULT_METADATA = HypothesisMetadata(
        hypothesis_id="ALPHA_01_INSTITUTIONAL_ORB",
        name="Institutional Opening Range & Anchored VWAP Momentum",
        category="MICROSTRUCTURE_FLOW",
        economic_rationale=(
            "FIIs and Mutual Funds execute algorithmic basket orders between 09:15-09:45 AM. "
            "When price breaks out of this range with volume > 1.8x average while holding above Anchored VWAP, "
            "it indicates persistent institutional accumulation."
        ),
        target_instruments=["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "NIFTYBEES"],
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

        self.volume_surge_mult = self.parameters.get("volume_surge_mult", 1.8)
        self.orb_end_time = self.parameters.get("orb_end_time", "09:45:00")
        self.entry_cutoff_time = self.parameters.get("entry_cutoff_time", "13:30:00")
        self.square_off_time = self.parameters.get("square_off_time", "15:15:00")
        self.atr_multiplier = self.parameters.get("atr_multiplier", 1.5)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Parameter search space for optimization and DSR validation."""
        return {
            "volume_surge_mult": [1.5, 1.8, 2.2],
            "orb_end_time": ["09:30:00", "09:45:00"],
            "atr_multiplier": [1.0, 1.5, 2.0],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes intraday signals (+1, -1, 0) across full OHLCV series.
        """
        extractor = MicrostructureFeatureExtractor()
        
        # 1. Feature Engineering
        df_feat = extractor.calculate_anchored_vwap(df)
        df_feat = extractor.calculate_volume_delta(df_feat)
        df_feat = extractor.calculate_opening_range(df_feat, orb_start="09:15:00", orb_end=self.orb_end_time)

        # 2. Extract Time Strings
        time_strings = df_feat.index.strftime("%H:%M:%S")

        # Conditions
        is_trading_window = (time_strings > self.orb_end_time) & (time_strings <= self.entry_cutoff_time)
        is_square_off = time_strings >= self.square_off_time
        has_volume_surge = df_feat["volume_surge_ratio"] >= self.volume_surge_mult

        # Long Condition: Close > ORH and Close > VWAP with Volume Surge
        long_condition = (
            is_trading_window
            & (df_feat["close"] > df_feat["orb_high"])
            & (df_feat["close"] > df_feat["vwap"])
            & has_volume_surge
        )

        # Short Condition: Close < ORL and Close < VWAP with Volume Surge
        short_condition = (
            is_trading_window
            & (df_feat["close"] < df_feat["orb_low"])
            & (df_feat["close"] < df_feat["vwap"])
            & has_volume_surge
        )

        # Vectorized signal assignment
        signal = np.zeros(len(df_feat))
        position = 0.0

        for i in range(len(df_feat)):
            if is_square_off[i]:
                position = 0.0
            elif long_condition.iloc[i]:
                position = 1.0
            elif short_condition.iloc[i]:
                position = -1.0
            signal[i] = position

        df_feat["signal"] = signal
        return df_feat

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """Real-time streaming handler (implements BaseStrategy)."""
        # In live streaming, maintains state and produces real-time SignalEvents
        return None
