"""
Ashva Quantitative Alpha Strategy — Alpha 22: NR4 Volatility Expansion High-RR (NR4-VE)
Alpha ID: 22_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When Day T-1 has the narrowest daily range of the past 4 sessions (NR4), the stock is in acute
volatility compression. When Day T's opening 15m bar closes above the NR4 high with a strong bullish
body (body >= 60% of bar) and RVOL >= 1.25x, price expands in a powerful unidirectional burst.
With a 1.60% take-profit target and only 0.45% stop loss, the asymmetric 3.56:1 payoff structure
produces positive expected value at win rates as low as 36%.

Economic Mechanism:
NR4 represents a 4-session equilibrium cycle. Retail order flow exhausts itself in 4 days of
contracting range. The first directional morning bar with institutional volume (RVOL) signals
smart money building new directional positions, driving rapid expansion.

Cost-Validated Geometry:
- Capital per trade: Rs 1.25L (25% of Rs 5L)
- At 40% win rate: 0.40 * Rs2000 - 0.60 * Rs562 = Rs800 - Rs337 = +Rs463 expected gross
- Average statutory cost per trade: ~Rs300
- Net expectancy: +Rs163 per trade
- Break-even win rate: 36% (very comfortable margin)

Contract Specification:
- NR4: Daily Range(T-1) < min(Range(T-2), Range(T-3), Range(T-4)).
- Opening 15m Bar: Close above NR4 High (Long) or below NR4 Low (Short).
- Body ratio >= 0.60 AND RVOL >= 1.25x.
- Entry: Close of first 15m bar (09:15-09:30 IST).
- Stop Loss: 0.45% from entry.
- Profit Target: 1.60% from entry (3.56:1 RR).
- Dynamic Profit Lock: Move SL to +0.50% once +1.00% profit reached.
- Time Horizon: Intraday 15m (15:15 IST square-off, max 10 bars).
- Universe: Dynamic. No hardcoded instruments.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis, HypothesisMetadata, StrategyHorizon, MarketMechanism,
)
from src.strategies.base import BaseStrategy
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha22NR4VoLatilityExpansion(BaseHypothesis, BaseStrategy):
    """
    Alpha 22: NR4 Volatility Expansion High-RR (NR4-VE).
    4-session range compression breakout with cost-validated 3.56:1 RR geometry.
    """

    strategy_id = "22_alpha"
    hypothesis_id = "22_alpha"
    name = "22_alpha — NR4 Volatility Expansion High-RR"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_body_ratio": 0.60,
            "min_rvol": 1.25,
            "stop_loss_pct": 0.0045,
            "take_profit_pct": 0.0160,        # 1.60% TP (3.56:1 RR)
            "trail_trigger_pct": 0.0100,
            "trail_lock_pct": 0.0050,
            "max_holding_bars": 10,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="22_alpha",
            name="22_alpha — NR4 Volatility Expansion High-RR",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "4-session NR4 range compression followed by institutional morning breakout. "
                "Cost-validated 3.56:1 RR breaks even at 36% win rate after Indian statutory costs."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="22_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25, 1.40],
            "take_profit_pct": [0.0140, 0.0160, 0.0180],
            "stop_loss_pct": [0.0040, 0.0045, 0.0055],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 60:
            out = df.copy()
            out["signal"] = 0.0; out["stop_loss"] = 0.0; out["take_profit"] = 0.0
            return out

        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")
        out.sort_index(inplace=True)

        min_body = float(self.parameters.get("min_body_ratio", 0.60))
        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0160))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0100))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0050))
        max_bars = int(self.parameters.get("max_holding_bars", 10))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        dates_series = pd.Series(out.index.date, index=out.index)

        # Point-in-time daily range — strictly shifted
        daily_high = out["high"].groupby(dates_series).max()
        daily_low = out["low"].groupby(dates_series).min()
        daily_range = daily_high - daily_low

        # NR4: range(T-1) < min(range(T-2), range(T-3), range(T-4))
        r_t1 = daily_range.shift(1)
        r_prev3_min = pd.concat([daily_range.shift(2), daily_range.shift(3), daily_range.shift(4)], axis=1).min(axis=1)
        is_nr4 = (r_t1 < r_prev3_min) & pd.notna(r_prev3_min)

        nr4_high = daily_high.shift(1)
        nr4_low = daily_low.shift(1)

        nr4_map = dates_series.map(is_nr4).fillna(False)
        nr4_h_map = dates_series.map(nr4_high)
        nr4_l_map = dates_series.map(nr4_low)

        vol_sma20 = out["volume"].shift(1).rolling(20).mean()

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        grouped = out.groupby(dates_series)

        for date_val, group in grouped:
            if len(group) < 2:
                continue

            pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0
            bars_held = 0; traded_today = False

            for i, (ts, row) in enumerate(group.iterrows()):
                idx_pos = out.index.get_loc(ts)
                t_str = ts.strftime("%H:%M")
                c_px = float(row["close"]); h_px = float(row["high"])
                l_px = float(row["low"]); o_px = float(row["open"])

                if t_str >= square_off:
                    if pos != 0.0:
                        pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0; bars_held = 0
                    signals[idx_pos] = 0.0; continue

                if pos != 0.0:
                    bars_held += 1
                    if pos > 0:
                        if h_px >= entry_px * (1.0 + trail_trig):
                            sl_px = max(sl_px, entry_px * (1.0 + trail_lock))
                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0; bars_held = 0
                    else:
                        if l_px <= entry_px * (1.0 - trail_trig):
                            sl_px = min(sl_px, entry_px * (1.0 - trail_lock))
                        if h_px >= sl_px or l_px <= tp_px or bars_held >= max_bars:
                            pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0; bars_held = 0

                    signals[idx_pos] = pos; stop_losses[idx_pos] = sl_px; take_profits[idx_pos] = tp_px

                else:
                    # Only fire on first bar of the session
                    if i == 0 and not traded_today:
                        has_nr4 = bool(nr4_map.iloc[idx_pos])
                        nr4_h = nr4_h_map.iloc[idx_pos]
                        nr4_l = nr4_l_map.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if has_nr4 and pd.notna(nr4_h) and pd.notna(nr4_l) and pd.notna(v_sma) and v_sma > 0:
                            bar_range = h_px - l_px
                            body = abs(c_px - o_px)
                            body_ratio = body / bar_range if bar_range > 0 else 0.0
                            is_vol_ok = row["volume"] >= v_sma * min_rvol

                            if (c_px > nr4_h) and (c_px > o_px) and body_ratio >= min_body and is_vol_ok:
                                pos = 1.0; entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0; traded_today = True

                            elif (c_px < nr4_l) and (c_px < o_px) and body_ratio >= min_body and is_vol_ok:
                                pos = -1.0; entry_px = c_px
                                sl_px = entry_px * (1.0 + sl_pct)
                                tp_px = entry_px * (1.0 - tp_pct)
                                bars_held = 0; traded_today = True

                    signals[idx_pos] = pos; stop_losses[idx_pos] = sl_px; take_profits[idx_pos] = tp_px

        out["signal"] = signals; out["stop_loss"] = stop_losses; out["take_profit"] = take_profits
        return out

    def on_bar(self, event: BarEvent) -> List[SignalEvent]:
        sym = event.symbol; t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)
        if t_str >= "15:15":
            if pos != 0.0:
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
            return []
        if pos != 0.0:
            ep = self._entry_price.get(sym, event.close)
            sl = ep * (1.0 - 0.0045) if pos > 0 else ep * (1.0 + 0.0045)
            tp = ep * (1.0 + 0.0160) if pos > 0 else ep * (1.0 - 0.0160)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1
            if (pos > 0 and (event.low <= sl or event.high >= tp or self._bars_held[sym] >= 10)) or \
               (pos < 0 and (event.high >= sl or event.low <= tp or self._bars_held[sym] >= 10)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
        return []
