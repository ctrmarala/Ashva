"""
Ashva Quantitative Alpha Strategy — Alpha 18: European Open Momentum Expansion (EOM-1300)
Alpha ID: 18_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Between 11:30 and 13:00 IST (Indian market lunchtime), price action settles into a tight consolidation box.
When the European/London financial markets open at 13:00–13:30 IST, fresh cross-border foreign institutional flows
enter liquid Indian large caps (ADRs/GDRs, IT, Banking). A breakout of the 11:30–13:00 consolidation box with volume
drives a swift 2.0R expansion into the afternoon.

Economic Mechanism:
London market opening creates a predictable liquidity and volatility spike at 13:00–13:30 IST.
Breaking out of the low-volatility lunchtime equilibrium establishes strong afternoon continuation.

Contract Specification:
- Midday Consolidation Box: High and Low established between 11:30 and 13:00 IST.
- European Open Breakout: 13:00 to 14:00 IST.
- Volume Confirmation: RVOL >= 1.25x 20-period Volume SMA.
- Stop Loss: 0.40% from entry.
- Profit Target: 0.85% from entry (2.12 : 1 Reward-to-Risk).
- Dynamic Profit Lock: Move SL to +0.15% once +0.45% profit reached.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 6 holding bars on 15m).
- Universe: Dynamic active universe (77 equities). No hardcoded instruments.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    StrategyHorizon,
    MarketMechanism,
)
from src.strategies.base import BaseStrategy
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha18EuropeanOpenMomentum(BaseHypothesis, BaseStrategy):
    """
    Alpha 18: European Open Momentum Expansion (EOM-1300).
    Time-of-day cross-border liquidity injection breakout strategy.
    """

    strategy_id = "18_alpha"
    hypothesis_id = "18_alpha"
    name = "18_alpha — European Open Momentum Expansion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.25,                 # 1.25x relative volume surge
            "stop_loss_pct": 0.0040,          # 0.40% tight structural stop
            "take_profit_pct": 0.0085,        # 0.85% take profit (2.12:1 RR)
            "trail_trigger_pct": 0.0045,      # Dynamic profit lock trigger at +0.45%
            "trail_lock_pct": 0.0015,         # Lock in +0.15% once trigger hit
            "max_holding_bars": 6,            # Max holding duration (1.5 hours on 15m)
            "timeframe": "15m",               # Default research timeframe
            "box_start_time": "11:30",        # Midday box start
            "box_end_time": "13:00",          # Midday box end
            "entry_start_time": "13:00",      # European open window opens
            "entry_end_time": "14:00",        # European open window cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="18_alpha",
            name="18_alpha — European Open Momentum Expansion",
            category="TIME_OF_DAY_MICROSTRUCTURE",
            economic_rationale=(
                "Captures European open (13:00 IST) liquidity injection following midday consolidation. "
                "Fast 6-bar holding duration minimizes overnight risk while capturing rapid 2.12:1 expansion."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="18_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25, 1.40],
            "stop_loss_pct": [0.0035, 0.0040, 0.0050],
            "take_profit_pct": [0.0075, 0.0085, 0.0100],
            "max_holding_bars": [5, 6, 8],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 50:
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

        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0040))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0085))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0045))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0015))
        max_bars = int(self.parameters.get("max_holding_bars", 6))
        box_start = str(self.parameters.get("box_start_time", "11:30"))
        box_end = str(self.parameters.get("box_end_time", "13:00"))
        entry_start = str(self.parameters.get("entry_start_time", "13:00"))
        entry_end = str(self.parameters.get("entry_end_time", "14:00"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        vol_prev = out["volume"].shift(1)
        vol_sma20 = vol_prev.rolling(20).mean()

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        dates_series = pd.Series(out.index.date, index=out.index)
        grouped = out.groupby(dates_series)

        for date_val, group in grouped:
            group_len = len(group)
            if group_len < 4:
                continue

            pos = 0.0
            entry_px = 0.0
            sl_px = 0.0
            tp_px = 0.0
            bars_held = 0
            traded_today = False

            box_high = -1e9
            box_low = 1e9
            box_formed = False

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

                # Build midday box
                if box_start <= t_str <= box_end:
                    if h_px > box_high:
                        box_high = h_px
                    if l_px < box_low:
                        box_low = l_px
                    box_formed = True

                if pos != 0.0:
                    bars_held += 1
                    # Dynamic Trailing Profit Lock
                    if pos > 0:
                        if h_px >= entry_px * (1.0 + trail_trig):
                            sl_px = max(sl_px, entry_px * (1.0 + trail_lock))
                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0
                    elif pos < 0:
                        if l_px <= entry_px * (1.0 - trail_trig):
                            sl_px = min(sl_px, entry_px * (1.0 - trail_lock))
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
                    # Check European Open breakout
                    if (not traded_today) and box_formed and (entry_start <= t_str <= entry_end):
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(v_sma) and v_sma > 0:
                            is_vol_ok = (v_prev >= v_sma * min_rvol)

                            # Bullish European Open Breakout
                            if (c_px > box_high) and is_vol_ok:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish European Open Breakdown
                            elif (c_px < box_low) and is_vol_ok:
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
        sym = event.symbol
        c_px = event.close
        h_px = event.high
        l_px = event.low
        t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)

        if t_str >= "15:15":
            if pos != 0.0:
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
            return []

        if pos != 0.0:
            sl_px = self._entry_price[sym] * (1.0 - 0.0040) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0040)
            tp_px = self._entry_price[sym] * (1.0 + 0.0085) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0085)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 6)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 6)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []