"""
Ashva Quantitative Alpha Strategy — Alpha 1: Opening Gap Continuation
Alpha ID: 1_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When a stock opens with a meaningful overnight gap and early-session price action
confirms acceptance of the gap direction rather than rejecting it, the stock may continue
moving in the same direction during the intraday session.

Economic Mechanism:
Overnight information creates an opening inventory and order imbalance. If the market accepts
the gap (early-session trading sustains the price level and breaks beyond the opening range),
the initial imbalance can persist into an intraday momentum continuation trend.

Contract Specification:
- Long Entry: Gap Up >= gap_threshold_pct, opening bar closes positive, price breaks above opening high.
- Short Entry: Gap Down <= -gap_threshold_pct, opening bar closes negative, price breaks below opening low.
- Stop Loss: Fixed alpha-level stop-loss percentage from entry price.
- Profit Target: Fixed alpha-level take-profit percentage from entry price.
- Exit: Target hit, Stop Loss hit, Max Holding Bars reached, or Intraday 15:15 Square-off.
- Universe: Dynamic active universe (77 equities). No hardcoded instruments.
- Horizon: Intraday (15:15 IST mandatory close).
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


class Alpha1OpeningGapContinuation(BaseHypothesis, BaseStrategy):
    """
    Alpha 1: Opening Gap Continuation
    Captures intraday directional continuation following overnight price acceptance.
    """

    strategy_id = "1_alpha"
    hypothesis_id = "1_alpha"
    name = "1_alpha — Opening Gap Continuation"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gap_threshold_pct": 0.005,       # 0.5% opening gap threshold
            "confirmation_bars": 1,           # Opening bar confirmation
            "stop_loss_pct": 0.010,           # 1.0% stop loss
            "take_profit_pct": 0.020,         # 2.0% profit target (2:1 RR)
            "max_holding_bars": 16,           # 4 hours holding on 15m
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:30",      # Confirmation completed by 09:30
            "entry_end_time": "14:00",        # Cease new entries
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="1_alpha",
            name="1_alpha — Opening Gap Continuation",
            category="OPENING_AUCTION",
            economic_rationale=(
                "When a stock opens with a meaningful overnight gap and early-session price action "
                "confirms acceptance of the gap direction rather than rejecting it, the initial inventory "
                "and informational imbalance may persist and drive intraday continuation."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="1_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "gap_threshold_pct": [0.003, 0.005, 0.008],
            "stop_loss_pct": [0.008, 0.010, 0.015],
            "take_profit_pct": [0.015, 0.020, 0.030],
            "max_holding_bars": [8, 16, 24],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates leak-free directional signals across historical OHLCV data.
        Returns DataFrame with 'signal' column (+1.0 Long, -1.0 Short, 0.0 Flat).
        """
        if df.empty or len(df) < 5:
            out = df.copy()
            out["signal"] = 0.0
            return out

        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        out.sort_index(inplace=True)

        gap_thresh = float(self.parameters.get("gap_threshold_pct", 0.005))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.010))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.020))
        max_bars = int(self.parameters.get("max_holding_bars", 16))
        entry_start = str(self.parameters.get("entry_start_time", "09:30"))
        entry_end = str(self.parameters.get("entry_end_time", "14:00"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Compute prior day close strictly across session boundaries (Strict Zero Look-Ahead)
        dates = pd.Series(out.index.date, index=out.index)
        daily_close = out.groupby(dates)["close"].last()
        prev_day_close_map = daily_close.shift(1).to_dict()

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)
        grouped = out.groupby(dates)

        for date_val, group in grouped:
            prev_close = prev_day_close_map.get(date_val)
            if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
                continue

            group_len = len(group)
            if group_len < 2:
                continue

            open_bar = group.iloc[0]
            open_price = float(open_bar["open"])
            gap_pct = (open_price - prev_close) / prev_close

            # Check gap condition
            is_gap_up = gap_pct >= gap_thresh
            is_gap_down = gap_pct <= -gap_thresh

            if not (is_gap_up or is_gap_down):
                continue

            # Early session acceptance on opening bar (bar 0)
            b0_close = float(open_bar["close"])
            b0_open = float(open_bar["open"])
            b0_high = float(open_bar["high"])
            b0_low = float(open_bar["low"])

            # Acceptance: Price maintains gap direction and doesn't immediately fill
            accepted_bull = is_gap_up and (b0_close >= b0_open) and (b0_low >= prev_close * 0.999)
            accepted_bear = is_gap_down and (b0_close <= b0_open) and (b0_high <= prev_close * 1.001)

            if not (accepted_bull or accepted_bear):
                continue

            pos = 0.0
            entry_px = 0.0
            sl_px = 0.0
            tp_px = 0.0
            bars_held = 0
            traded_today = False

            for i in range(1, group_len):
                curr_bar = group.iloc[i]
                t_str = curr_bar.name.strftime("%H:%M")
                c_px = float(curr_bar["close"])
                h_px = float(curr_bar["high"])
                l_px = float(curr_bar["low"])

                # Check active position exits
                if pos != 0.0:
                    bars_held += 1
                    exit_triggered = False

                    if t_str >= square_off:
                        exit_triggered = True
                    elif bars_held >= max_bars:
                        exit_triggered = True
                    elif pos == 1.0: # Long exits
                        if l_px <= sl_px or h_px >= tp_px:
                            exit_triggered = True
                    elif pos == -1.0: # Short exits
                        if h_px >= sl_px or l_px <= tp_px:
                            exit_triggered = True

                    if exit_triggered:
                        pos = 0.0
                        entry_px = 0.0
                        sl_px = 0.0
                        tp_px = 0.0
                        bars_held = 0
                    else:
                        row_loc = out.index.get_loc(curr_bar.name)
                        signals[row_loc] = pos
                        stop_losses[row_loc] = sl_px
                        take_profits[row_loc] = tp_px
                        continue

                # Check new entry if eligible
                if not traded_today and (entry_start <= t_str <= entry_end):
                    if accepted_bull and c_px > b0_high:
                        pos = 1.0
                        entry_px = c_px
                        sl_px = entry_px * (1.0 - sl_pct)
                        tp_px = entry_px * (1.0 + tp_pct)
                        bars_held = 0
                        traded_today = True
                        row_loc = out.index.get_loc(curr_bar.name)
                        signals[row_loc] = 1.0
                        stop_losses[row_loc] = sl_px
                        take_profits[row_loc] = tp_px
                    elif accepted_bear and c_px < b0_low:
                        pos = -1.0
                        entry_px = c_px
                        sl_px = entry_px * (1.0 + sl_pct)
                        tp_px = entry_px * (1.0 - tp_pct)
                        bars_held = 0
                        traded_today = True
                        row_loc = out.index.get_loc(curr_bar.name)
                        signals[row_loc] = -1.0
                        stop_losses[row_loc] = sl_px
                        take_profits[row_loc] = tp_px

        out["signal"] = signals
        out["stop_loss"] = stop_losses
        out["take_profit"] = take_profits
        return out

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """Real-time streaming event handler."""
        sym = bar.symbol
        t_str = bar.timestamp.strftime("%H:%M") if hasattr(bar.timestamp, 'strftime') else "09:30"
        
        curr_pos = self._current_pos.get(sym, 0.0)
        if curr_pos != 0.0:
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1
            if t_str >= self.parameters.get("square_off_time", "15:15") or self._bars_held[sym] >= self.parameters.get("max_holding_bars", 16):
                self._current_pos[sym] = 0.0
                return SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=sym,
                    timestamp=bar.timestamp,
                    signal_type=SignalType.EXIT,
                    strength=1.0,
                    timeframe=self.metadata.timeframe,
                    metadata={"reason": "TIME_SQUARE_OFF"}
                )
        return None
