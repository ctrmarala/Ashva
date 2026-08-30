"""
Ashva Quantitative Alpha Strategy — Alpha 17: Open-Equals-Extreme Institutional Drive (OEE-ID)
Alpha ID: 17_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When a high-volume liquid stock opens at 09:15 and forms an opening 15m candle where Open == Low (within 0.05% tolerance,
meaning zero lower selling wick) and finishes with a strong bullish body (> 70% of bar range) on heavy relative volume
(RVOL >= 1.75x), aggressive institutional market buy orders are overpowering all limit liquidity. Entering on the close
of the first bar delivers a high-conviction 2.0R continuation.

Economic Mechanism:
Open=Low indicates absolute buyer dominance from the opening auction tick. The lack of downward auction testing signals
large institutional orders working via TWAP/VWAP algorithms that continue buying throughout the morning.

Contract Specification:
- Open=Low / Open=High Condition: |Open - Low| / Open <= 0.0005 (Bullish) or |Open - High| / Open <= 0.0005 (Bearish).
- Body Ratio: |Close - Open| / (High - Low) >= 0.70.
- Volume Surge: First bar RVOL >= 1.75x 20-period Volume SMA.
- Trigger Bar: Exactly the first 15m bar of the session (09:30 IST close).
- Stop Loss: 0.45% from entry.
- Profit Target: 0.90% from entry (2.0 : 1 Reward-to-Risk).
- Dynamic Profit Lock: Move SL to +0.15% once +0.50% profit reached.
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


class Alpha17OpenEqualsExtremeDrive(BaseHypothesis, BaseStrategy):
    """
    Alpha 17: Open-Equals-Extreme Institutional Drive (OEE-ID).
    Opening auction order flow imbalance strategy with zero adverse wick and heavy volume.
    """

    strategy_id = "17_alpha"
    hypothesis_id = "17_alpha"
    name = "17_alpha — Open-Equals-Extreme Institutional Drive"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "max_wick_tol": 0.0005,           # Max 0.05% tolerance for open=extreme
            "min_body_ratio": 0.70,           # Min 70% candle body ratio
            "min_rvol": 1.75,                 # 1.75x relative volume surge
            "stop_loss_pct": 0.0045,          # 0.45% tight structural stop
            "take_profit_pct": 0.0090,        # 0.90% take profit (2.0:1 RR)
            "trail_trigger_pct": 0.0050,      # Dynamic profit lock trigger at +0.50%
            "trail_lock_pct": 0.0015,         # Lock in +0.15% once trigger hit
            "max_holding_bars": 8,            # Max holding duration (2 hours on 15m)
            "timeframe": "15m",               # Default research timeframe
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="17_alpha",
            name="17_alpha — Open-Equals-Extreme Institutional Drive",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "Exploits immediate institutional buying/selling aggression at the opening bell. "
                "Open=Low / Open=High with zero wick and heavy volume signals persistent institutional algorithms."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="17_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_body_ratio": [0.65, 0.70, 0.75],
            "min_rvol": [1.50, 1.75, 2.00],
            "stop_loss_pct": [0.0040, 0.0045, 0.0055],
            "take_profit_pct": [0.0080, 0.0090, 0.0105],
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

        wick_tol = float(self.parameters.get("max_wick_tol", 0.0005))
        min_body = float(self.parameters.get("min_body_ratio", 0.70))
        min_rvol = float(self.parameters.get("min_rvol", 1.75))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0090))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0050))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0015))
        max_bars = int(self.parameters.get("max_holding_bars", 8))
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
            if group_len < 2:
                continue

            pos = 0.0
            entry_px = 0.0
            sl_px = 0.0
            tp_px = 0.0
            bars_held = 0

            for i in range(group_len):
                curr_bar = group.iloc[i]
                idx_pos = out.index.get_loc(curr_bar.name)
                t_str = curr_bar.name.strftime("%H:%M")
                c_px = float(curr_bar["close"])
                o_px = float(curr_bar["open"])
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
                    # Check first bar of the day (09:15 - 09:30)
                    if i == 0:
                        bar_range = h_px - l_px
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if bar_range > 0 and pd.notna(v_sma) and v_sma > 0:
                            body = abs(c_px - o_px)
                            body_ratio = body / bar_range
                            is_vol_surge = (curr_bar["volume"] >= v_sma * min_rvol)

                            # Bullish Open=Low Drive
                            if (abs(o_px - l_px) / o_px <= wick_tol) and (c_px > o_px) and (body_ratio >= min_body) and is_vol_surge:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0

                            # Bearish Open=High Drive
                            elif (abs(h_px - o_px) / o_px <= wick_tol) and (c_px < o_px) and (body_ratio >= min_body) and is_vol_surge:
                                pos = -1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 + sl_pct)
                                tp_px = entry_px * (1.0 - tp_pct)
                                bars_held = 0

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
            tp_px = self._entry_price[sym] * (1.0 + 0.0090) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0090)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 8)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 8)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []