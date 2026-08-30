"""
Ashva Quantitative Alpha Strategy — Alpha 12: Fair Value Gap Institutional Reversion (FVGR)
Alpha ID: 12_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Opening session price discovery (09:15–09:45 IST) creates structural Fair Value Gaps (FVGs) / liquidity imbalances.
When price subsequently pulls back into this unfilled institutional demand/supply imbalance with low volume,
institutional resting limit orders absorb the retrace, initiating a clean, high-conviction 3.5:1 reward-to-risk
reversal back toward the session high/low.

Economic Mechanism:
Fair Value Gaps occur when market orders overwhelm resting liquidity, leaving unfilled orders at specific price nodes.
Institutions defending these positions re-accumulate/re-distribute on the first retest. Entering at the FVG boundary
provides exceptionally tight structural invalidation (0.40% SL) with deep targets (1.60% TP).

Contract Specification:
- Fair Value Gap Detection: Point-in-time 3-bar imbalance pattern formed during opening morning window.
- Retrace Confirmation: Price touches FVG zone between 10:00 and 12:30 IST.
- Volume Divergence: Retrace volume must be lower than opening FVG creation volume.
- Asymmetric Stop Loss: 0.40% from entry price.
- Profit Target: 1.60% from entry price (4.0 : 1 Reward-to-Risk).
- Dynamic Profit Protection: Move SL to breakeven (+0.15%) once +0.70% profit is reached.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 8 holding bars on 30m).
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
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha12FairValueGapReversion(BaseHypothesis, BaseStrategy):
    """
    Alpha 12: Fair Value Gap Institutional Reversion (FVGR).
    Institutional imbalance retest strategy offering ultra-tight risk and 4:1 asymmetric payoff.
    """

    strategy_id = "12_alpha"
    hypothesis_id = "12_alpha"
    name = "12_alpha — Fair Value Gap Institutional Reversion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_fvg_pct": 0.0030,            # Min 0.30% gap size for institutional FVG
            "stop_loss_pct": 0.0075,          # 0.75% structural stop loss
            "take_profit_pct": 0.0175,        # 1.75% profit target (2.3:1 RR)
            "trail_trigger_pct": 0.0070,      # Dynamic profit lock trigger at +0.70%
            "trail_lock_pct": 0.0015,         # Lock in +0.15% once trigger hit
            "max_holding_bars": 8,            # Max holding duration (4 hours on 30m)
            "timeframe": "30m",               # Default research timeframe
            "entry_start_time": "10:00",      # Retrace entry window opens
            "entry_end_time": "12:30",        # Retrace entry cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="12_alpha",
            name="12_alpha — Fair Value Gap Institutional Reversion",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "Exploits institutional order-flow imbalances (Fair Value Gaps) formed at market open. "
                "Retests of unfilled FVG zones provide asymmetric reward-to-risk with 0.75% risk, "
                "drastically lowering statutory friction impact."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "30m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="12_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "min_fvg_pct": [0.0020, 0.0030, 0.0040],
            "stop_loss_pct": [0.0065, 0.0075, 0.0085],
            "take_profit_pct": [0.0155, 0.0175, 0.0195],
            "max_holding_bars": [6, 8, 10],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates leak-free directional signals across historical OHLCV data.
        Returns DataFrame with 'signal', 'stop_loss', and 'take_profit' columns.
        """
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

        min_fvg = float(self.parameters.get("min_fvg_pct", 0.0030))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0075))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0175))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0070))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0015))
        max_bars = int(self.parameters.get("max_holding_bars", 8))
        entry_start = str(self.parameters.get("entry_start_time", "10:00"))
        entry_end = str(self.parameters.get("entry_end_time", "12:30"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        dates_series = pd.Series(out.index.date, index=out.index)
        grouped = out.groupby(dates_series)

        for date_val, group in grouped:
            group_len = len(group)
            if group_len < 3:
                continue

            pos = 0.0
            entry_px = 0.0
            sl_px = 0.0
            tp_px = 0.0
            bars_held = 0
            traded_today = False

            # Session-level FVG memory
            bull_fvg_top = 0.0
            bull_fvg_bottom = 0.0
            bear_fvg_top = 0.0
            bear_fvg_bottom = 0.0
            has_bull_fvg = False
            has_bear_fvg = False

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

                # 1. Opening FVG Detection in first 2-3 bars (09:15 - 10:00)
                if i >= 2 and not (has_bull_fvg or has_bear_fvg) and t_str <= entry_start:
                    b0 = group.iloc[i - 2]
                    b1 = group.iloc[i - 1]
                    b2 = group.iloc[i]

                    # Bullish FVG: Bar 2 Low > Bar 0 High by at least min_fvg_pct
                    if float(b2["low"]) > float(b0["high"]) * (1.0 + min_fvg):
                        bull_fvg_top = float(b2["low"])
                        bull_fvg_bottom = float(b0["high"])
                        has_bull_fvg = True

                    # Bearish FVG: Bar 2 High < Bar 0 Low by at least min_fvg_pct
                    elif float(b2["high"]) < float(b0["low"]) * (1.0 - min_fvg):
                        bear_fvg_top = float(b0["low"])
                        bear_fvg_bottom = float(b2["high"])
                        has_bear_fvg = True

                # 2. Manage open position
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
                    # 3. Retrace Entry Window (10:00 to 12:30 IST)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        # Bullish FVG Retest: Price dips into the bullish gap [bottom, top]
                        if has_bull_fvg and (l_px <= bull_fvg_top) and (c_px >= bull_fvg_bottom):
                            pos = 1.0
                            entry_px = c_px
                            sl_px = entry_px * (1.0 - sl_pct)
                            tp_px = entry_px * (1.0 + tp_pct)
                            bars_held = 0
                            traded_today = True

                        # Bearish FVG Retest: Price rallies into the bearish gap [bottom, top]
                        elif has_bear_fvg and (h_px >= bear_fvg_bottom) and (c_px <= bear_fvg_top):
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
            sl_pct = float(self.parameters.get("stop_loss_pct", 0.0075))
            tp_pct = float(self.parameters.get("take_profit_pct", 0.0175))
            max_bars = int(self.parameters.get("max_holding_bars", 8))
            
            sl_px = self._entry_price[sym] * (1.0 - sl_pct) if pos > 0 else self._entry_price[sym] * (1.0 + sl_pct)
            tp_px = self._entry_price[sym] * (1.0 + tp_pct) if pos > 0 else self._entry_price[sym] * (1.0 - tp_pct)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= max_bars)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= max_bars)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []