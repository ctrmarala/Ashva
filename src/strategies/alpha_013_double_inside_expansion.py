"""
Ashva Quantitative Alpha Strategy — Alpha 13: Multi-Session Double-Inside Asymmetric Expansion (DIE-2R)
Alpha ID: 13_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When Day T-1 is an Inside Day within Day T-2, and Day T-2 is an Inside Day within Day T-3 (Double Inside Day),
the multi-session price action represents an extreme equilibrium bottleneck.
A 15m morning breakout on Day T above Day T-1 High (Long) or below Day T-1 Low (Short) with RVOL >= 1.15x
initiates a high-probability unidirectional 2.0R expansion before 15:15 IST with minimal drawdown.

Economic Mechanism:
Consecutive inside sessions compress volatility and cluster resting stop-orders tightly above and below the T-1 range.
The initial morning impulse triggers cascading liquidity stops that propel prices toward the 2.0R target with high win-rate.

Contract Specification:
- Multi-Session Contraction: High(T-1) <= High(T-2) & Low(T-1) >= Low(T-2) AND High(T-2) <= High(T-3) & Low(T-2) >= Low(T-3).
- 15m Morning Breakout: 09:30 to 11:30 IST.
- Volume Confirmation: Relative Volume (RVOL) >= 1.15x 20-period SMA.
- Stop Loss: 0.35% from entry (tied to narrow coiled boundary).
- Profit Target: 0.75% from entry (approx 2.14 : 1 Reward-to-Risk).
- Dynamic Profit Lock: Move SL to +0.10% once +0.45% is reached.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 8 holding bars on 15m).
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


class Alpha13DoubleInsideExpansion(BaseHypothesis, BaseStrategy):
    """
    Alpha 13: Multi-Session Double-Inside Asymmetric Expansion (DIE-2R).
    High-win-rate volatility expansion following multi-day equilibrium bottlenecks.
    """

    strategy_id = "13_alpha"
    hypothesis_id = "13_alpha"
    name = "13_alpha — Multi-Session Double-Inside Asymmetric Expansion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.15,                 # 1.15x relative volume surge
            "stop_loss_pct": 0.0035,          # 0.35% tight structural stop
            "take_profit_pct": 0.0075,        # 0.75% take profit (2.14:1 RR)
            "trail_trigger_pct": 0.0045,      # Dynamic profit lock trigger at +0.45%
            "trail_lock_pct": 0.0010,         # Lock in +0.10% once trigger hit
            "max_holding_bars": 8,            # Max holding duration (2 hours on 15m)
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:30",      # Morning window opens
            "entry_end_time": "11:30",        # Morning cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="13_alpha",
            name="13_alpha — Multi-Session Double-Inside Asymmetric Expansion",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "Captures explosive 2.0R expansion following two consecutive inside days. "
                "Tight 0.35% stop tied to the narrow T-1 range provides superior post-cost net edge."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="13_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.05, 1.15, 1.30],
            "stop_loss_pct": [0.0030, 0.0035, 0.0045],
            "take_profit_pct": [0.0070, 0.0075, 0.0090],
            "max_holding_bars": [6, 8, 10],
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

        min_rvol = float(self.parameters.get("min_rvol", 1.15))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0035))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0075))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0045))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0010))
        max_bars = int(self.parameters.get("max_holding_bars", 8))
        entry_start = str(self.parameters.get("entry_start_time", "09:30"))
        entry_end = str(self.parameters.get("entry_end_time", "11:30"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time daily calculations
        daily_high = out["high"].groupby(out.index.date).max()
        daily_low = out["low"].groupby(out.index.date).min()

        # Shift by 1, 2, 3 to get T-1, T-2, T-3 (strictly completed past days)
        h_t1 = daily_high.shift(1)
        l_t1 = daily_low.shift(1)
        h_t2 = daily_high.shift(2)
        l_t2 = daily_low.shift(2)
        h_t3 = daily_high.shift(3)
        l_t3 = daily_low.shift(3)

        # Double Inside Condition
        is_id_t1 = (h_t1 <= h_t2) & (l_t1 >= l_t2)
        is_id_t2 = (h_t2 <= h_t3) & (l_t2 >= l_t3)
        is_double_inside = is_id_t1 & is_id_t2

        dates_series = pd.Series(out.index.date, index=out.index)
        double_id_map = dates_series.map(is_double_inside).fillna(False)
        h_t1_map = dates_series.map(h_t1)
        l_t1_map = dates_series.map(l_t1)

        vol_prev = out["volume"].shift(1)
        vol_sma20 = vol_prev.rolling(20).mean()

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        grouped = out.groupby(dates_series)

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
                    # Check Morning Double-Inside Breakout
                    has_double_id = bool(double_id_map.iloc[idx_pos])
                    if (not traded_today) and has_double_id and (entry_start <= t_str <= entry_end):
                        prior_h = h_t1_map.iloc[idx_pos]
                        prior_l = l_t1_map.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(prior_h) and pd.notna(prior_l) and pd.notna(v_sma) and v_sma > 0:
                            is_vol_ok = (v_prev >= v_sma * min_rvol)

                            # Bullish Expansion: Price breaks above Prior Day High with RVOL
                            if (c_px > prior_h) and is_vol_ok:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish Expansion: Price breaks below Prior Day Low with RVOL
                            elif (c_px < prior_l) and is_vol_ok:
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
            sl_px = self._entry_price[sym] * (1.0 - 0.0035) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0035)
            tp_px = self._entry_price[sym] * (1.0 + 0.0075) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0075)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 8)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 8)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []