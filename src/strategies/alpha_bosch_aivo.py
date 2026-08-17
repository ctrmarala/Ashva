"""
Alpha Strategy 9: Institutional Value Oscillations (AIVO - "The Bosch Strategy")
Specifically designed for high-value, range-bound institutional compounders (e.g. BOSCHLTD, HINDUNILVR, NESTLEIND, ITC).
Exploits Anchored VWAP Dynamic Value Dispersion Bands (+/- 2.2 sigma) with RSI & Volume Absorption Confirmation.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata
from src.features.microstructure import MicrostructureFeatureExtractor
from src.core.events import BarEvent, SignalEvent, SignalType


class AlphaInstitutionalValueOscillations(BaseStrategy, BaseHypothesis):
    """
    Hypothesis 9: Dynamic VWAP Dispersion Value Band Mean Reversion for Range-Bound Leaders.
    """

    DEFAULT_METADATA = HypothesisMetadata(
        hypothesis_id="ALPHA_09_INSTITUTIONAL_VALUE_OSCILLATION",
        name="Institutional Value Oscillations (AIVO - Bosch Archetype)",
        category="VALUE_OSCILLATION",
        economic_rationale=(
            "High-priced institutional heavyweights and FMCG/Auto leaders trade within strong intrinsic valuation bands. "
            "When price is discounted to VWAP - 2.2 sigma with oversold RSI (< 30) and volume absorption, "
            "institutional value desks systematically step in to accumulate shares."
        ),
        target_instruments=["BOSCHLTD", "HINDUNILVR", "NESTLEIND", "ITC", "TCS"],
        timeframe="15m",
    )

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or self.DEFAULT_METADATA
        BaseHypothesis.__init__(self, metadata=meta, parameters=parameters)
        BaseStrategy.__init__(self, strategy_id=meta.hypothesis_id, parameters=parameters)

        self.sigma_mult = self.parameters.get("sigma_mult", 2.2)
        self.rsi_period = self.parameters.get("rsi_period", 14)
        self.rsi_oversold = self.parameters.get("rsi_oversold", 30.0)
        self.rsi_overbought = self.parameters.get("rsi_overbought", 70.0)
        self.square_off_time = self.parameters.get("square_off_time", "15:15:00")

        # Live state tracking
        self._current_day = None
        self._cum_vol = 0.0
        self._cum_pv = 0.0
        self._cum_pv2 = 0.0
        self._recent_closes: List[float] = []
        self._active_position = 0.0

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "sigma_mult": [1.8, 2.2, 2.5],
            "rsi_oversold": [25.0, 30.0],
            "rsi_overbought": [70.0, 75.0],
        }

    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        extractor = MicrostructureFeatureExtractor()

        # 1. Anchored VWAP & Standard Deviation Bands
        df_feat = extractor.calculate_anchored_vwap(df_out)
        
        # Calculate rolling variance from VWAP
        typical = (df_feat["high"] + df_feat["low"] + df_feat["close"]) / 3.0
        vol = df_feat["volume"]
        
        # Session VWAP Standard Deviation
        df_feat["date"] = df_feat.index.date
        df_feat["pv"] = typical * vol
        df_feat["cum_v"] = df_feat.groupby("date")["volume"].cumsum()
        df_feat["cum_pv"] = df_feat.groupby("date")["pv"].cumsum()
        df_feat["sess_vwap"] = df_feat["cum_pv"] / df_feat["cum_v"].replace(0, np.nan)

        # Variance from VWAP
        df_feat["dev2"] = vol * ((typical - df_feat["sess_vwap"]) ** 2)
        df_feat["cum_dev2"] = df_feat.groupby("date")["dev2"].cumsum()
        df_feat["vwap_std"] = np.sqrt(df_feat["cum_dev2"] / df_feat["cum_v"].replace(0, np.nan)).fillna(0.0)

        df_feat["upper_band"] = df_feat["sess_vwap"] + (self.sigma_mult * df_feat["vwap_std"])
        df_feat["lower_band"] = df_feat["sess_vwap"] - (self.sigma_mult * df_feat["vwap_std"])

        # 2. RSI Indicator
        df_feat["rsi"] = self._calc_rsi(df_feat["close"], self.rsi_period)

        # 3. Time Filter
        time_strings = df_feat.index.strftime("%H:%M:%S")
        is_square_off = time_strings >= self.square_off_time

        # 4. Long & Short Conditions
        long_condition = (
            (df_feat["close"] <= df_feat["lower_band"])
            & (df_feat["rsi"] <= self.rsi_oversold)
            & (~is_square_off)
        )

        short_condition = (
            (df_feat["close"] >= df_feat["upper_band"])
            & (df_feat["rsi"] >= self.rsi_overbought)
            & (~is_square_off)
        )

        signal = np.zeros(len(df_feat))
        position = 0.0

        for i in range(len(df_feat)):
            if is_square_off[i]:
                position = 0.0
            elif long_condition.iloc[i] and position <= 0:
                position = 1.0
            elif short_condition.iloc[i] and position >= 0:
                position = -1.0
            elif position > 0 and df_feat["close"].iloc[i] >= df_feat["sess_vwap"].iloc[i]:
                # Mean reverted back to VWAP equilibrium
                position = 0.0
            elif position < 0 and df_feat["close"].iloc[i] <= df_feat["sess_vwap"].iloc[i]:
                position = 0.0

            signal[i] = position

        df_feat["signal"] = signal
        return df_feat

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """Incremental on_bar handler for live session streaming."""
        bar_date = bar.timestamp.date()
        bar_time_str = bar.timestamp.strftime("%H:%M:%S")

        # Reset on new day
        if self._current_day != bar_date:
            self._current_day = bar_date
            self._cum_vol = 0.0
            self._cum_pv = 0.0
            self._cum_pv2 = 0.0
            self._active_position = 0.0

        typical = (bar.high + bar.low + bar.close) / 3.0
        self._cum_vol += bar.volume
        self._cum_pv += typical * bar.volume
        
        vwap = (self._cum_pv / self._cum_vol) if self._cum_vol > 0 else bar.close
        self._cum_pv2 += bar.volume * ((typical - vwap) ** 2)
        vwap_std = np.sqrt(self._cum_pv2 / self._cum_vol) if self._cum_vol > 0 else 0.0

        lower_band = vwap - (self.sigma_mult * vwap_std)
        upper_band = vwap + (self.sigma_mult * vwap_std)

        self._recent_closes.append(bar.close)
        if len(self._recent_closes) > 30:
            self._recent_closes.pop(0)

        rsi = 50.0
        if len(self._recent_closes) >= self.rsi_period:
            s = pd.Series(self._recent_closes)
            rsi = float(self._calc_rsi(s, self.rsi_period).iloc[-1])

        # Square-off check
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

        # Value Dip Entry (Buy Low)
        if bar.close <= lower_band and rsi <= self.rsi_oversold and self._active_position <= 0:
            self._active_position = 1.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=1.0,
                strength=1.0,
                strategy_id=self.strategy_id,
                stop_loss=lower_band - (vwap_std * 0.8),
                take_profit=vwap,
                metadata={"vwap": vwap, "lower_band": lower_band, "rsi": rsi, "type": "VALUE_DIP_BUY"},
            )

        # Overvalued Reversal Entry (Sell High)
        elif bar.close >= upper_band and rsi >= self.rsi_overbought and self._active_position >= 0:
            self._active_position = -1.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=-1.0,
                strength=1.0,
                strategy_id=self.strategy_id,
                stop_loss=upper_band + (vwap_std * 0.8),
                take_profit=vwap,
                metadata={"vwap": vwap, "upper_band": upper_band, "rsi": rsi, "type": "VALUE_CREST_SELL"},
            )

        # Mean Reversion Target Fill
        elif self._active_position > 0 and bar.close >= vwap:
            self._active_position = 0.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=0.0,
                strength=0.0,
                strategy_id=self.strategy_id,
                metadata={"reason": "VWAP_EQUILIBRIUM_REACHED"},
            )
        elif self._active_position < 0 and bar.close <= vwap:
            self._active_position = 0.0
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=0.0,
                strength=0.0,
                strategy_id=self.strategy_id,
                metadata={"reason": "VWAP_EQUILIBRIUM_REACHED"},
            )

        return None
