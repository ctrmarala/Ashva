"""
Ashva Quantitative Alpha Strategy — Alpha 2: Volatility Expansion Breakout
Alpha ID: 2_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When a liquid equity breaks out of a multi-bar consolidation with an explosive expansion
in Average True Range (ATR) accompanied by a strong volume surge (> 2.0x 20-period average),
institutional order flow drives strong directional momentum that overcomes transaction friction.

Economic Mechanism:
Volatility compression followed by abnormal volume expansion signals institutional participation.
Restricting entries to high-volatility expanding bars and enforcing a 2.4:1 Reward-to-Risk profile
(3.0% TP vs 1.25% SL) with a maximum of 1 trade per stock per day prevents transaction churn.

Contract Specification:
- Long Entry: Close > Donchian High (lookback), Volume >= 2.0x Vol SMA20, ATR14 >= 1.25x ATR SMA20, Close > EMA50.
- Short Entry: Close < Donchian Low (lookback), Volume >= 2.0x Vol SMA20, ATR14 >= 1.25x ATR SMA20, Close < EMA50.
- Stop Loss: 1.25% from entry price.
- Profit Target: 3.0% from entry price (Reward-to-Risk = 2.4 : 1).
- Time Horizon: Intraday (15:15 IST mandatory close, max 12 bars holding).
- Universe: Dynamic active universe (77 equities). No hardcoded instruments.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    HypothesisStatus,
    StrategyHorizon,
    MarketMechanism,
)
from src.strategies.base import BaseStrategy
from src.core.universe_manager import get_universe_symbols
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha2VolatilityExpansionBreakout(BaseHypothesis, BaseStrategy):
    """
    Alpha 2: Volatility Expansion Breakout with Dynamic Volume Confirmation.
    Captures explosive intraday breakouts while filtering low-volatility churn.
    """

    strategy_id = "2_alpha"
    hypothesis_id = "2_alpha"
    name = "2_alpha — Volatility Expansion Breakout"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "lookback_bars": 20,              # Donchian channel lookback
            "vol_mult": 2.0,                  # Volume surge multiplier
            "atr_mult": 1.25,                 # ATR expansion threshold vs SMA20(ATR)
            "stop_loss_pct": 0.0125,          # 1.25% stop loss
            "take_profit_pct": 0.030,         # 3.0% take profit (2.4:1 RR)
            "max_holding_bars": 12,           # 3 hours holding on 15m
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:30",      # Cease early opening noise
            "entry_end_time": "14:00",        # Cease new late entries
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="2_alpha",
            name="2_alpha — Volatility Expansion Breakout",
            category="VOLATILITY_EXPANSION",
            economic_rationale=(
                "When a liquid equity breaks out of a multi-bar consolidation with an explosive expansion "
                "in Average True Range (ATR) accompanied by a strong volume surge (> 2.0x 20-period average), "
                "institutional order flow drives strong directional momentum that overcomes transaction friction."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="2_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "lookback_bars": [15, 20, 30],
            "vol_mult": [1.8, 2.0, 2.5],
            "stop_loss_pct": [0.010, 0.0125, 0.015],
            "take_profit_pct": [0.025, 0.030, 0.040],
            "max_holding_bars": [8, 12, 16],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates leak-free directional signals across historical OHLCV data.
        Returns DataFrame with 'signal', 'stop_loss', and 'take_profit' columns.
        """
        if df.empty or len(df) < 30:
            out = df.copy()
            out["signal"] = 0.0
            out["stop_loss"] = 0.0
            out["take_profit"] = 0.0
            return out

        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        out.sort_index(inplace=True)

        lookback = int(self.parameters.get("lookback_bars", 20))
        vol_mult = float(self.parameters.get("vol_mult", 2.0))
        atr_mult = float(self.parameters.get("atr_mult", 1.25))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0125))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.030))
        max_bars = int(self.parameters.get("max_holding_bars", 12))
        entry_start = str(self.parameters.get("entry_start_time", "09:30"))
        entry_end = str(self.parameters.get("entry_end_time", "14:00"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time indicators (shift(1) to guarantee zero lookahead)
        high_prev = out["high"].shift(1)
        low_prev = out["low"].shift(1)
        close_prev = out["close"].shift(1)
        vol_prev = out["volume"].shift(1)

        # 1. Donchian channels computed strictly on historical closed bars
        donchian_high = high_prev.rolling(lookback).max()
        donchian_low = low_prev.rolling(lookback).min()

        # 2. Volume moving average
        vol_sma20 = vol_prev.rolling(20).mean()

        # 3. ATR 14
        tr1 = high_prev - low_prev
        tr2 = (high_prev - close_prev.shift(1)).abs()
        tr3 = (low_prev - close_prev.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_sma20 = atr14.rolling(20).mean()

        # 4. Trend filter EMA50
        ema50 = close_prev.ewm(span=50, adjust=False).mean()

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        dates = pd.Series(out.index.date, index=out.index)
        grouped = out.groupby(dates)

        for date_val, group in grouped:
            group_len = len(group)
            if group_len < 2:
                continue

            pos = 0.0
            entry_px = 0.0
            sl_px = 0.0
            tp_px = 0.0
            bars_held = 0
            traded_today = False

            for i in range(group_len):
                curr_bar = group.iloc[i]
                idx_pos = out.index.get_loc(curr_bar.name)
                t_str = curr_bar.name.strftime("%H:%M")
                c_px = float(curr_bar["close"])
                h_px = float(curr_bar["high"])
                l_px = float(curr_bar["low"])

                # Square-off at or past 15:15 IST
                if t_str >= square_off:
                    if pos != 0.0:
                        pos = 0.0
                        entry_px = 0.0
                        sl_px = 0.0
                        tp_px = 0.0
                        bars_held = 0
                    signals[idx_pos] = 0.0
                    continue

                if pos != 0.0:
                    bars_held += 1
                    # Long Exit conditions
                    if pos > 0:
                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0
                    # Short Exit conditions
                    elif pos < 0:
                        if h_px >= sl_px or l_px <= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0

                    signals[idx_pos] = pos
                    stop_losses[idx_pos] = sl_px
                    take_profits[idx_pos] = tp_px

                else:
                    # New Entry check within entry window (Max 1 trade per stock per day)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        d_high = donchian_high.iloc[idx_pos]
                        d_low = donchian_low.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]
                        a14 = atr14.iloc[idx_pos]
                        a_sma = atr_sma20.iloc[idx_pos]
                        e50 = ema50.iloc[idx_pos]

                        if pd.notna(d_high) and pd.notna(d_low) and pd.notna(v_sma) and pd.notna(a_sma) and v_sma > 0 and a_sma > 0:
                            # Volatility expansion & volume surge conditions
                            is_vol_surge = (v_prev >= v_sma * vol_mult)
                            is_atr_expansion = (a14 >= a_sma * atr_mult)

                            # Bullish Breakout
                            if c_px > d_high and is_vol_surge and is_atr_expansion and c_px > e50:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish Breakdown
                            elif c_px < d_low and is_vol_surge and is_atr_expansion and c_px < e50:
                                pos = -1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 + sl_pct)
                                tp_px = entry_px * (1.0 - tp_pct)
                                bars_held = 0
                                traded_today = True

                    signals[idx_pos] = pos
                    stop_losses[idx_pos] = sl_px
                    take_profits[idx_pos] = tp_px

        out["signal"] = signals
        out["stop_loss"] = stop_losses
        out["take_profit"] = take_profits
        return out

    def on_bar(self, event: BarEvent) -> List[SignalEvent]:
        """Real-time event-driven bar handler for live/paper engine."""
        sym = event.symbol
        c_px = event.close
        h_px = event.high
        l_px = event.low
        t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)

        # 15:15 Square-off
        if t_str >= "15:15":
            if pos != 0.0:
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
            return []

        # Check holding exit
        if pos != 0.0:
            sl_px = self._entry_price[sym] * (1.0 - 0.0125) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0125)
            tp_px = self._entry_price[sym] * (1.0 + 0.030) if pos > 0 else self._entry_price[sym] * (1.0 - 0.030)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 12)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 12)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []