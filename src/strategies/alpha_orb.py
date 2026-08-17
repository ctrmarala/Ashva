"""
Alpha Strategy 1: Institutional Opening Range & Anchored VWAP Momentum Breakout
Exploits institutional morning flow (09:15-09:45 AM) accompanied by volume surges and VWAP trend confirmation.
Unified for both batch historical backtesting and incremental on_bar live streaming.
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

        # State tracking for live streaming
        self._current_day = None
        self._orb_high = -np.inf
        self._orb_low = np.inf
        self._cum_volume = 0.0
        self._cum_pv = 0.0
        self._recent_volumes: List[float] = []
        self._active_position = 0.0

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "volume_surge_mult": [1.5, 1.8, 2.2],
            "orb_end_time": ["09:30:00", "09:45:00"],
            "atr_multiplier": [1.0, 1.5, 2.0],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes intraday signals (+1, -1, 0) across full OHLCV series."""
        extractor = MicrostructureFeatureExtractor()
        df_feat = extractor.calculate_anchored_vwap(df)
        df_feat = extractor.calculate_volume_delta(df_feat)
        df_feat = extractor.calculate_opening_range(df_feat, orb_start="09:15:00", orb_end=self.orb_end_time)

        time_strings = df_feat.index.strftime("%H:%M:%S")
        is_trading_window = (time_strings > self.orb_end_time) & (time_strings <= self.entry_cutoff_time)
        is_square_off = time_strings >= self.square_off_time
        has_volume_surge = df_feat["volume_surge_ratio"] >= self.volume_surge_mult

        long_condition = (
            is_trading_window
            & (df_feat["close"] > df_feat["orb_high"])
            & (df_feat["close"] > df_feat["vwap"])
            & has_volume_surge
        )

        short_condition = (
            is_trading_window
            & (df_feat["close"] < df_feat["orb_low"])
            & (df_feat["close"] < df_feat["vwap"])
            & has_volume_surge
        )

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
        """Incremental live streaming bar handler."""
        bar_date = bar.timestamp.date()
        bar_time_str = bar.timestamp.strftime("%H:%M:%S")

        # 1. Reset daily state on new session
        if self._current_day != bar_date:
            self._current_day = bar_date
            self._orb_high = -np.inf
            self._orb_low = np.inf
            self._cum_volume = 0.0
            self._cum_pv = 0.0
            self._active_position = 0.0

        # 2. Update VWAP and Volume history
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self._cum_volume += bar.volume
        self._cum_pv += typical_price * bar.volume
        vwap = (self._cum_pv / self._cum_volume) if self._cum_volume > 0 else bar.close

        self._recent_volumes.append(bar.volume)
        if len(self._recent_volumes) > 20:
            self._recent_volumes.pop(0)
        avg_vol = np.mean(self._recent_volumes) if self._recent_volumes else bar.volume
        vol_ratio = (bar.volume / avg_vol) if avg_vol > 0 else 1.0

        # 3. Track Opening Range (09:15 to orb_end_time)
        if bar_time_str <= self.orb_end_time:
            self._orb_high = max(self._orb_high, bar.high)
            self._orb_low = min(self._orb_low, bar.low)
            return None

        # 4. Square-off check
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

        # 5. Breakout Evaluation
        if self.orb_end_time < bar_time_str <= self.entry_cutoff_time:
            has_surge = vol_ratio >= self.volume_surge_mult
            
            # Long Breakout
            if bar.close > self._orb_high and bar.close > vwap and has_surge and self._active_position <= 0:
                self._active_position = 1.0
                return SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    direction=1.0,
                    strength=1.0,
                    strategy_id=self.strategy_id,
                    stop_loss=vwap,
                    take_profit=bar.close + (bar.close - vwap) * 2.0,
                    metadata={"orb_high": self._orb_high, "vwap": vwap, "vol_surge": vol_ratio},
                )
            
            # Short Breakout
            elif bar.close < self._orb_low and bar.close < vwap and has_surge and self._active_position >= 0:
                self._active_position = -1.0
                return SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    direction=-1.0,
                    strength=1.0,
                    strategy_id=self.strategy_id,
                    stop_loss=vwap,
                    take_profit=bar.close - (vwap - bar.close) * 2.0,
                    metadata={"orb_low": self._orb_low, "vwap": vwap, "vol_surge": vol_ratio},
                )

        return None
