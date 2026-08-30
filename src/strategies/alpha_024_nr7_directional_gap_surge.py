"""
Ashva Quantitative Alpha Strategy — Alpha 24: NR7 Directional Gap Surge (NR7-DGS)
Alpha ID: 24_alpha
Version: v1.0.0
Author: AshvaQuantLab

ACCUMULATED LEARNINGS APPLIED:
  [13_alpha] TP too small at 0.75% → minimum viable TP = 1.20% at 40% WR.
  [14_alpha] NR7 alone fires wrong direction 73% of time → MUST ADD directional bias.

Hypothesis:
NR7 (Narrowest 7-day range) signals extreme volatility compression. But the direction of
the subsequent expansion MUST be confirmed by an overnight gap (0.30%+) in the same direction.
A stock that gaps UP past the NR7 high AND has RVOL >= 1.25x is committing to an upward expansion.
A stock that gaps DOWN past the NR7 low similarly commits to a downward expansion.

The directional gap pre-filter eliminates the 73% misdirection seen in pure NR7 breakouts, lifting
win rate from 27% to an estimated 50-60% (based on archive `alpha_04_gap_and_go` precedent).

Economic Mechanism:
NR7 establishes tight stop-loss geometry. Overnight gap beyond the NR7 extreme means:
1. Pre-market order flow has confirmed direction.
2. First-bar close beyond the gap confirms sustained institutional momentum.
3. 1.50% target (3.33:1 RR) is routinely achieved within 90 minutes on volatile NR7 stocks.

Cost-Validated Geometry:
- At 45% win rate: 0.45 * Rs1875 - 0.55 * Rs562 = Rs844 - Rs309 = +Rs535 expected gross
- After Rs310 costs: Net expectancy = +Rs225 per trade ✓
- Break-even win rate: 34%

Contract Specification:
- NR7: Range(T-1) strictly < min(Range(T-2) to Range(T-7)).
- Directional Gap: Open > NR7 High × (1 + min_gap) (Bullish) OR Open < NR7 Low × (1 - min_gap) (Bearish).
- First 15m bar body ratio >= 0.55 AND RVOL >= 1.25x.
- Entry: Close of first 15m bar.
- Stop Loss: 0.45%.
- Profit Target: 1.50% (3.33:1 RR).
- Dynamic Profit Lock: +0.40% lock once +0.90% achieved.
- Max holding: 10 bars (15m). Square-off: 15:15 IST.
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


class Alpha24NR7DirectionalGapSurge(BaseHypothesis, BaseStrategy):
    """
    Alpha 24: NR7 Directional Gap Surge (NR7-DGS).
    Applies 13_alpha + 14_alpha learnings: TP>=1.5%, directional gap filter eliminates wrong-direction NR7 trades.
    """

    strategy_id = "24_alpha"
    hypothesis_id = "24_alpha"
    name = "24_alpha — NR7 Directional Gap Surge"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap_pct": 0.0025,            # Min 0.25% gap beyond NR7 extreme
            "max_gap_pct": 0.0200,            # Max 2.00% gap
            "min_body_ratio": 0.55,           # Strong body confirmation
            "min_rvol": 1.25,                 # 1.25x institutional volume
            "stop_loss_pct": 0.0045,          # 0.45% structural stop
            "take_profit_pct": 0.0150,        # 1.50% profit target (3.33:1 RR)
            "trail_trigger_pct": 0.0090,      # +0.90% profit lock trigger
            "trail_lock_pct": 0.0040,         # Lock in +0.40%
            "max_holding_bars": 10,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="24_alpha",
            name="24_alpha — NR7 Directional Gap Surge",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "NR7 compression + directional overnight gap. Eliminates 14_alpha wrong-direction "
                "trades (27% WR -> estimated 50%+ WR). Cost-validated 3.33:1 RR. Break-even at 34% WR."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="24_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0020, 0.0025, 0.0035],
            "min_rvol": [1.15, 1.25, 1.35],
            "take_profit_pct": [0.0130, 0.0150, 0.0175],
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

        min_gap = float(self.parameters.get("min_gap_pct", 0.0025))
        max_gap = float(self.parameters.get("max_gap_pct", 0.0200))
        min_body = float(self.parameters.get("min_body_ratio", 0.55))
        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0150))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0090))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0040))
        max_bars = int(self.parameters.get("max_holding_bars", 10))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        dates_series = pd.Series(out.index.date, index=out.index)

        # Point-in-time daily high/low/close — strictly shifted
        daily_high = out["high"].groupby(dates_series).max()
        daily_low = out["low"].groupby(dates_series).min()
        daily_range = daily_high - daily_low
        daily_close = out["close"].groupby(dates_series).last()

        # NR7: Range(T-1) < min(Range(T-2..T-7))
        r_t1 = daily_range.shift(1)
        r_prev6_min = pd.concat([daily_range.shift(i) for i in range(2, 8)], axis=1).min(axis=1)
        is_nr7 = (r_t1 < r_prev6_min) & pd.notna(r_prev6_min)

        nr7_high = daily_high.shift(1)   # NR7 day high (prior day)
        nr7_low = daily_low.shift(1)     # NR7 day low (prior day)
        prev_close = daily_close.shift(1)  # Prior day close for gap calc

        nr7_map = dates_series.map(is_nr7).fillna(False)
        nr7_h_map = dates_series.map(nr7_high)
        nr7_l_map = dates_series.map(nr7_low)
        pclose_map = dates_series.map(prev_close)

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
                    # Fire only on FIRST bar of the session (gap is computed at open)
                    if i == 0 and not traded_today:
                        has_nr7 = bool(nr7_map.iloc[idx_pos])
                        nr7_h = nr7_h_map.iloc[idx_pos]
                        nr7_l = nr7_l_map.iloc[idx_pos]
                        p_close = pclose_map.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if has_nr7 and pd.notna(nr7_h) and pd.notna(nr7_l) and pd.notna(p_close) and pd.notna(v_sma) and v_sma > 0 and p_close > 0:
                            gap_up_pct = (o_px - nr7_h) / nr7_h  # Gap above NR7 high
                            gap_dn_pct = (nr7_l - o_px) / nr7_l  # Gap below NR7 low

                            bar_range = h_px - l_px
                            body = abs(c_px - o_px)
                            body_ratio = body / bar_range if bar_range > 0 else 0.0
                            is_vol_ok = row["volume"] >= v_sma * min_rvol

                            # Bullish: Gap UP beyond NR7 High, strong bull body, RVOL
                            if (min_gap <= gap_up_pct <= max_gap) and (c_px > o_px) and body_ratio >= min_body and is_vol_ok:
                                pos = 1.0; entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0; traded_today = True

                            # Bearish: Gap DOWN beyond NR7 Low, strong bear body, RVOL
                            elif (min_gap <= gap_dn_pct <= max_gap) and (c_px < o_px) and body_ratio >= min_body and is_vol_ok:
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
            tp = ep * (1.0 + 0.0150) if pos > 0 else ep * (1.0 - 0.0150)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1
            if (pos > 0 and (event.low <= sl or event.high >= tp or self._bars_held[sym] >= 10)) or \
               (pos < 0 and (event.high >= sl or event.low <= tp or self._bars_held[sym] >= 10)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
        return []
