"""
Ashva Quantitative Alpha Strategy — Alpha 15: 3-Day Trend Surge with Intraday Pullback (3DTS-IP)
Alpha ID: 15_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Equities displaying 3 consecutive days of directional trend persistence (Close_T1 > Close_T2 > Close_T3 for Longs)
possess strong institutional accumulation. When price has a brief morning retrace to the 15m EMA20 followed by
a volume surge expansion, the trade resumes the multi-day macro impulse toward a 2.2R target.

Economic Mechanism:
Multi-day trend persistence reflects institutional block accumulation. Intraday pullbacks to short-term dynamic
support (15m EMA20) shake out weak retail hands, allowing high-conviction continuation entries with asymmetric R:R.

Contract Specification:
- 3-Day Daily Trend: Close_T1 > Close_T2 > Close_T3 (Bullish) or Close_T1 < Close_T2 < Close_T3 (Bearish).
- 15m Morning Pullback: Price touches or dips near 20-period EMA on 15m.
- Reversal Trigger: Next bar closes back above EMA20 with RVOL >= 1.25x.
- Entry Window: 09:45 to 12:00 IST.
- Stop Loss: 0.45% from entry.
- Profit Target: 1.00% from entry (2.22 : 1 Reward-to-Risk).
- Dynamic Profit Lock: Move SL to +0.20% once +0.60% profit reached.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 10 holding bars on 15m).
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


class Alpha15ThreeDayTrendSurge(BaseHypothesis, BaseStrategy):
    """
    Alpha 15: 3-Day Trend Surge with Intraday Pullback (3DTS-IP).
    Multi-timeframe trend alignment strategy capturing institutional momentum pullbacks.
    """

    strategy_id = "15_alpha"
    hypothesis_id = "15_alpha"
    name = "15_alpha — 3-Day Trend Surge with Intraday Pullback"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "ema_period": 20,                 # 15m EMA lookback
            "min_rvol": 1.25,                 # 1.25x relative volume surge
            "stop_loss_pct": 0.0045,          # 0.45% tight structural stop
            "take_profit_pct": 0.0100,        # 1.00% take profit (2.22:1 RR)
            "trail_trigger_pct": 0.0060,      # Dynamic profit lock trigger at +0.60%
            "trail_lock_pct": 0.0020,         # Lock in +0.20% once trigger hit
            "max_holding_bars": 10,           # Max holding duration (2.5 hours on 15m)
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:45",      # Morning window opens
            "entry_end_time": "12:00",        # Morning cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="15_alpha",
            name="15_alpha — 3-Day Trend Surge with Intraday Pullback",
            category="TREND_CONTINUATION",
            economic_rationale=(
                "Combines 3-day multi-session trend persistence with 15m EMA20 pullback entries. "
                "High 2.22:1 reward-to-risk and trend-aligned flow ensures superior post-cost net performance."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="15_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "ema_period": [15, 20, 25],
            "min_rvol": [1.15, 1.25, 1.40],
            "stop_loss_pct": [0.0040, 0.0045, 0.0055],
            "take_profit_pct": [0.0090, 0.0100, 0.0120],
            "max_holding_bars": [8, 10, 12],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 60:
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

        ema_len = int(self.parameters.get("ema_period", 20))
        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0100))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0060))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0020))
        max_bars = int(self.parameters.get("max_holding_bars", 10))
        entry_start = str(self.parameters.get("entry_start_time", "09:45"))
        entry_end = str(self.parameters.get("entry_end_time", "12:00"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time daily calculations
        daily_close = out["close"].groupby(out.index.date).last()
        c_t1 = daily_close.shift(1)
        c_t2 = daily_close.shift(2)
        c_t3 = daily_close.shift(3)

        # 3-Day Trend conditions
        is_trend_up = (c_t1 > c_t2) & (c_t2 > c_t3) & pd.notna(c_t3)
        is_trend_down = (c_t1 < c_t2) & (c_t2 < c_t3) & pd.notna(c_t3)

        dates_series = pd.Series(out.index.date, index=out.index)
        trend_up_map = dates_series.map(is_trend_up).fillna(False)
        trend_down_map = dates_series.map(is_trend_down).fillna(False)

        # Intraday 15m Indicators (shift(1) to avoid lookahead)
        close_prev = out["close"].shift(1)
        low_prev = out["low"].shift(1)
        high_prev = out["high"].shift(1)
        vol_prev = out["volume"].shift(1)
        ema20 = close_prev.ewm(span=ema_len, adjust=False).mean()
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
                    # Check Morning Trend Pullback Entry
                    has_up = bool(trend_up_map.iloc[idx_pos])
                    has_down = bool(trend_down_map.iloc[idx_pos])

                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        e_val = ema20.iloc[idx_pos]
                        c_p = close_prev.iloc[idx_pos]
                        l_p = low_prev.iloc[idx_pos]
                        h_p = high_prev.iloc[idx_pos]
                        v_p = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(e_val) and pd.notna(v_sma) and v_sma > 0:
                            is_vol_ok = (v_p >= v_sma * min_rvol)

                            # Bullish: 3-day up-trend, prev bar low touched EMA20, current close rebounds above EMA20 with RVOL
                            if has_up and (l_p <= e_val * 1.002) and (c_p >= e_val) and (c_px > c_p) and is_vol_ok:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish: 3-day down-trend, prev bar high touched EMA20, current close drops below EMA20 with RVOL
                            elif has_down and (h_p >= e_val * 0.998) and (c_p <= e_val) and (c_px < c_p) and is_vol_ok:
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
            sl_px = self._entry_price[sym] * (1.0 - 0.0045) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0045)
            tp_px = self._entry_price[sym] * (1.0 + 0.0100) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0100)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 10)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 10)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []