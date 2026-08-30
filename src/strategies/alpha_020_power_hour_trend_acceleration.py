"""
Ashva Quantitative Alpha Strategy — Alpha 20: Power Hour Closing Trend Acceleration (PHTA-1415)
Alpha ID: 20_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Between 14:15 and 15:10 IST (Power Hour), institutional closing auction participation and index fund rebalancing
generate powerful unidirectional momentum. Stocks maintaining strong intraday trend alignment (Price > Session VWAP,
RSI > 55) that make fresh new session highs at 14:15 with heavy volume continue aggressively into the 15:15 close.

Economic Mechanism:
Mutual funds, ETFs, and institutional benchmark-tracking portfolios execute MOC (Market-On-Close) and VWAP orders in the
final 60 minutes of the session, driving strong momentum persistence with minimal pullback risk.

Contract Specification:
- Power Hour Window: 14:15 to 14:45 IST.
- Intraday Trend Alignment: Close > Session VWAP (Bullish) or Close < Session VWAP (Bearish).
- Session High/Low Breakout: Current 15m bar breaks above the highest high established prior to 14:15.
- Volume Confirmation: RVOL >= 1.35x 20-period Volume SMA.
- Stop Loss: 0.35% from entry.
- Profit Target: 0.70% from entry (2.0 : 1 Reward-to-Risk).
- Mandatory Exit: 15:15 IST.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 4 holding bars on 15m).
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


class Alpha20PowerHourTrendAcceleration(BaseHypothesis, BaseStrategy):
    """
    Alpha 20: Power Hour Closing Trend Acceleration (PHTA-1415).
    Institutional closing-auction flow acceleration strategy.
    """

    strategy_id = "20_alpha"
    hypothesis_id = "20_alpha"
    name = "20_alpha — Power Hour Closing Trend Acceleration"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.35,                 # 1.35x relative volume surge
            "stop_loss_pct": 0.0035,          # 0.35% tight structural stop
            "take_profit_pct": 0.0070,        # 0.70% take profit (2.0:1 RR)
            "trail_trigger_pct": 0.0040,      # Dynamic profit lock trigger at +0.40%
            "trail_lock_pct": 0.0010,         # Lock in +0.10% once trigger hit
            "max_holding_bars": 4,            # Max holding duration (1 hour on 15m)
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "14:15",      # Power hour entry opens
            "entry_end_time": "14:45",        # Power hour cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="20_alpha",
            name="20_alpha — Power Hour Closing Trend Acceleration",
            category="TIME_OF_DAY_MICROSTRUCTURE",
            economic_rationale=(
                "Captures institutional closing auction flows and MOC imbalances during Power Hour (14:15–15:15). "
                "Tight 0.35% stop and 4-bar holding duration minimizes holding risk."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="20_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.20, 1.35, 1.50],
            "stop_loss_pct": [0.0030, 0.0035, 0.0045],
            "take_profit_pct": [0.0065, 0.0070, 0.0085],
            "max_holding_bars": [3, 4, 5],
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

        min_rvol = float(self.parameters.get("min_rvol", 1.35))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0035))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0070))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0040))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0010))
        max_bars = int(self.parameters.get("max_holding_bars", 4))
        entry_start = str(self.parameters.get("entry_start_time", "14:15"))
        entry_end = str(self.parameters.get("entry_end_time", "14:45"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        vol_prev = out["volume"].shift(1)
        vol_sma20 = vol_prev.rolling(20).mean()

        # Point-in-time intraday VWAP calculation
        typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_price * out["volume"]

        dates_series = pd.Series(out.index.date, index=out.index)
        cum_pv = pv.groupby(dates_series).cumsum()
        cum_vol = out["volume"].groupby(dates_series).cumsum()
        vwap = cum_pv / (cum_vol + 1e-10)

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        grouped = out.groupby(dates_series)

        for date_val, group in grouped:
            group_len = len(group)
            if group_len < 6:
                continue

            pos = 0.0
            entry_px = 0.0
            sl_px = 0.0
            tp_px = 0.0
            bars_held = 0
            traded_today = False

            pre_1415_high = -1e9
            pre_1415_low = 1e9

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

                # Record session high/low before 14:15
                if t_str < entry_start:
                    if h_px > pre_1415_high:
                        pre_1415_high = h_px
                    if l_px < pre_1415_low:
                        pre_1415_low = l_px

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
                    # Check Power Hour breakout
                    if (not traded_today) and (entry_start <= t_str <= entry_end) and (pre_1415_high > -1e8):
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]
                        vwap_val = vwap.iloc[idx_pos]

                        if pd.notna(v_sma) and v_sma > 0 and pd.notna(vwap_val):
                            is_vol_ok = (v_prev >= v_sma * min_rvol)

                            # Bullish Power Hour Breakout: Above VWAP, breaks above session high
                            if (c_px > vwap_val) and (c_px > pre_1415_high) and is_vol_ok:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish Power Hour Breakdown: Below VWAP, breaks below session low
                            elif (c_px < vwap_val) and (c_px < pre_1415_low) and is_vol_ok:
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
            tp_px = self._entry_price[sym] * (1.0 + 0.0070) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0070)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 4)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 4)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []