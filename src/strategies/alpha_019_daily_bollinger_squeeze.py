"""
Ashva Quantitative Alpha Strategy — Alpha 19: Daily Bollinger Squeeze Expansion (DBS-15m)
Alpha ID: 19_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When a stock's Daily Bollinger Band Width (20-day) compresses to the lowest 15th percentile of its 60-day history (Daily Squeeze),
volatility is severely coiled. A 15m morning breakout on Day T with institutional volume (RVOL >= 1.30x) marks the beginning
of a multi-day volatility expansion cycle, delivering an explosive intraday 2.2R run.

Economic Mechanism:
Periods of prolonged low volatility are inevitably followed by large volatility expansions.
Entering on the initial 15m momentum thrust of a Daily Squeeze release provides unmatched asymmetric R:R.

Contract Specification:
- Daily Squeeze Condition: Daily BB Width (20 period) <= 60-day Rolling 15th percentile of BB Width.
- 15m Morning Trigger: 09:30 to 11:30 IST.
- Volume Confirmation: RVOL >= 1.30x 20-period Volume SMA.
- Stop Loss: 0.45% from entry.
- Profit Target: 1.00% from entry (2.22 : 1 Reward-to-Risk).
- Dynamic Profit Lock: Move SL to +0.20% once +0.55% profit reached.
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


class Alpha19DailyBollingerSqueeze(BaseHypothesis, BaseStrategy):
    """
    Alpha 19: Daily Bollinger Squeeze Expansion (DBS-15m).
    Captures multi-day volatility expansion cycles triggered by morning volume breakouts.
    """

    strategy_id = "19_alpha"
    hypothesis_id = "19_alpha"
    name = "19_alpha — Daily Bollinger Squeeze Expansion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.30,                 # 1.30x relative volume surge
            "stop_loss_pct": 0.0045,          # 0.45% tight structural stop
            "take_profit_pct": 0.0100,        # 1.00% take profit (2.22:1 RR)
            "trail_trigger_pct": 0.0055,      # Dynamic profit lock trigger at +0.55%
            "trail_lock_pct": 0.0020,         # Lock in +0.20% once trigger hit
            "max_holding_bars": 10,           # Max holding duration (2.5 hours on 15m)
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:30",      # Morning window opens
            "entry_end_time": "11:30",        # Morning cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="19_alpha",
            name="19_alpha — Daily Bollinger Squeeze Expansion",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "Captures initial directional thrust of a multi-day Daily Bollinger Squeeze release. "
                "High 2.22:1 reward-to-risk clears statutory friction easily."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="19_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.30, 1.45],
            "stop_loss_pct": [0.0035, 0.0045, 0.0055],
            "take_profit_pct": [0.0085, 0.0100, 0.0120],
            "max_holding_bars": [8, 10, 12],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 80:
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

        min_rvol = float(self.parameters.get("min_rvol", 1.30))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0100))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0055))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0020))
        max_bars = int(self.parameters.get("max_holding_bars", 10))
        entry_start = str(self.parameters.get("entry_start_time", "09:30"))
        entry_end = str(self.parameters.get("entry_end_time", "11:30"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time daily calculations
        daily_close = out["close"].groupby(out.index.date).last()
        daily_bb_mid = daily_close.rolling(20).mean()
        daily_bb_std = daily_close.rolling(20).std()
        daily_bb_width = (2.0 * daily_bb_std) / (daily_bb_mid + 1e-10)

        # Shift daily indicators to T-1 (completed yesterday)
        bb_w_t1 = daily_bb_width.shift(1)
        bb_w_p15 = bb_w_t1.rolling(60).quantile(0.20)
        is_daily_squeeze = (bb_w_t1 <= bb_w_p15) & pd.notna(bb_w_p15)

        daily_high = out["high"].groupby(out.index.date).max().shift(1)
        daily_low = out["low"].groupby(out.index.date).min().shift(1)

        dates_series = pd.Series(out.index.date, index=out.index)
        squeeze_map = dates_series.map(is_daily_squeeze).fillna(False)
        pdh_map = dates_series.map(daily_high)
        pdl_map = dates_series.map(daily_low)

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
                    # Check Daily Squeeze Expansion Breakout
                    has_squeeze = bool(squeeze_map.iloc[idx_pos])
                    if (not traded_today) and has_squeeze and (entry_start <= t_str <= entry_end):
                        prior_h = pdh_map.iloc[idx_pos]
                        prior_l = pdl_map.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(prior_h) and pd.notna(prior_l) and pd.notna(v_sma) and v_sma > 0:
                            is_vol_ok = (v_prev >= v_sma * min_rvol)

                            # Bullish Squeeze Release: Price breaks above Prior Day High with RVOL
                            if (c_px > prior_h) and is_vol_ok:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish Squeeze Release: Price breaks below Prior Day Low with RVOL
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
            sl_px = self._entry_price[sym] * (1.0 - 0.0045) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0045)
            tp_px = self._entry_price[sym] * (1.0 + 0.0100) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0100)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 10)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 10)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []