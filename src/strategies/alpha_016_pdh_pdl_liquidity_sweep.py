"""
Ashva Quantitative Alpha Strategy — Alpha 16: Prior Day Extreme Liquidity Sweep (PDH-PDL Sweep)
Alpha ID: 16_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When price opens and briefly pierces the Prior Day High (PDH) or Prior Day Low (PDL) by 0.1%–0.4% in the morning
(09:30–11:00 IST) to sweep resting breakout liquidity, but fails to sustain and closes back inside the prior day range
with high volume, an institutional trap is sprung. Shorting the PDH failure or buying the PDL failure delivers a
rapid, high-win-rate 2.5R mean-reversion move back toward the session midpoint.

Economic Mechanism:
Retail breakout traders place buy stops above PDH and sell stops below PDL. Smart money institutions trigger these
pools of liquidity to fill large counter-trend orders. Once the stop run exhausts, price reverses sharply.

Contract Specification:
- Liquidity Sweep Condition: High > PDH * 1.001 but Close < PDH (Bearish Sweep) OR Low < PDL * 0.999 but Close > PDL (Bullish Sweep).
- 15m Morning Window: 09:30 to 11:15 IST.
- Volume Rejection: RVOL >= 1.25x 20-period Volume SMA.
- Stop Loss: 0.40% (placed just beyond the sweep wick).
- Profit Target: 1.00% from entry (2.5 : 1 Reward-to-Risk).
- Dynamic Profit Lock: Move SL to +0.20% once +0.50% profit reached.
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


class Alpha16PDHPDLLiquiditySweep(BaseHypothesis, BaseStrategy):
    """
    Alpha 16: Prior Day Extreme Liquidity Sweep (PDH-PDL Sweep).
    Institutional stop-sweep trap and mean-reversion strategy with asymmetric 2.5:1 payoff.
    """

    strategy_id = "16_alpha"
    hypothesis_id = "16_alpha"
    name = "16_alpha — Prior Day Extreme Liquidity Sweep"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.25,                 # 1.25x relative volume surge
            "stop_loss_pct": 0.0040,          # 0.40% tight structural stop
            "take_profit_pct": 0.0100,        # 1.00% take profit (2.5:1 RR)
            "trail_trigger_pct": 0.0050,      # Dynamic profit lock trigger at +0.50%
            "trail_lock_pct": 0.0020,         # Lock in +0.20% once trigger hit
            "max_holding_bars": 8,            # Max holding duration (2 hours on 15m)
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:30",      # Morning window opens
            "entry_end_time": "11:15",        # Morning cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="16_alpha",
            name="16_alpha — Prior Day Extreme Liquidity Sweep",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "Exploits failed breakouts beyond Prior Day High/Low liquidity zones. "
                "Immediate rejection back inside the prior range provides 2.5:1 asymmetric reward-to-risk."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="16_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25, 1.40],
            "stop_loss_pct": [0.0035, 0.0040, 0.0050],
            "take_profit_pct": [0.0085, 0.0100, 0.0120],
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

        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0040))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0100))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0050))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0020))
        max_bars = int(self.parameters.get("max_holding_bars", 8))
        entry_start = str(self.parameters.get("entry_start_time", "09:30"))
        entry_end = str(self.parameters.get("entry_end_time", "11:15"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time daily calculations
        daily_high = out["high"].groupby(out.index.date).max()
        daily_low = out["low"].groupby(out.index.date).min()

        pdh = daily_high.shift(1)
        pdl = daily_low.shift(1)

        dates_series = pd.Series(out.index.date, index=out.index)
        pdh_map = dates_series.map(pdh)
        pdl_map = dates_series.map(pdl)

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
                    # Check Morning Liquidity Sweep
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        prior_h = pdh_map.iloc[idx_pos]
                        prior_l = pdl_map.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(prior_h) and pd.notna(prior_l) and pd.notna(v_sma) and v_sma > 0:
                            is_vol_ok = (v_prev >= v_sma * min_rvol)

                            # Bearish Sweep: Intrabar pierced PDH, but closed back BELOW PDH (Rejection)
                            if (h_px >= prior_h * 1.0005) and (c_px <= prior_h) and is_vol_ok:
                                pos = -1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 + sl_pct)
                                tp_px = entry_px * (1.0 - tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bullish Sweep: Intrabar pierced PDL, but closed back ABOVE PDL (Rejection)
                            elif (l_px <= prior_l * 0.9995) and (c_px >= prior_l) and is_vol_ok:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
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
            tp_px = self._entry_price[sym] * (1.0 + 0.0100) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0100)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 8)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 8)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []