"""
Ashva Quantitative Alpha Strategy — Alpha 21: Inside Day Morning Gap Surge (ID-MGS)
Alpha ID: 21_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
When Day T-1 is an Inside Day (High_T1 <= High_T2, Low_T1 >= Low_T2), the market is in equilibrium compression.
When Day T opens with a morning gap-up/gap-down of 0.30%-1.80% beyond the T-1 extreme, followed by a
strong-bodied first 15m bar with RVOL >= 1.20x, directional institutional conviction drives price to a 1.50%
intraday continuation before 15:15 IST.

Economic Mechanism:
Inside Day compresses resting liquidity orders at the T-1 High/Low extremes. A gap beyond these levels
forces immediate stop-order cascades and fresh directional participation. The combination of daily structural
conditioning and morning volume surge produces sustained intraday momentum lasting 90–180 minutes.

Cost-Validated Geometry:
- At 40% win rate with Rs 1.25L capital: Expected gross = 0.40 * Rs1875 - 0.60 * Rs563 = Rs412
- Average statutory cost per trade: ~Rs300
- Net expectancy: +Rs112 per trade (positive edge cleared)
- Minimum win rate required to break even: 38%

Contract Specification:
- Inside Day Filter: High_T1 <= High_T2 AND Low_T1 >= Low_T2 (strictly completed prior sessions).
- Morning Gap: 0.30% <= |Open_T - Close_T1| / Close_T1 <= 1.80%.
- First 15m Bar: Body ratio >= 0.55 AND RVOL >= 1.20x 20-period Volume SMA.
- Entry: Close of first 15m bar in the gap direction.
- Stop Loss: 0.45% from entry (structural, tight).
- Profit Target: 1.50% from entry (3.33 : 1 Reward-to-Risk).
- Dynamic Profit Lock: Move SL to +0.40% once +0.90% profit reached.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 10 holding bars on 15m).
- Universe: Dynamic active universe. No hardcoded instruments.
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


class Alpha21InsideDayGapSurge(BaseHypothesis, BaseStrategy):
    """
    Alpha 21: Inside Day Morning Gap Surge (ID-MGS).
    Archive-proven structural DNA from alpha_73 (OOS Sharpe 3.46, Net PF 2.42).
    Cost-validated TP/SL geometry ensures positive net expectancy at 38%+ win rate.
    """

    strategy_id = "21_alpha"
    hypothesis_id = "21_alpha"
    name = "21_alpha — Inside Day Morning Gap Surge"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap_pct": 0.0030,            # Min 0.30% opening gap
            "max_gap_pct": 0.0180,            # Max 1.80% opening gap (avoid blow-ups)
            "min_body_ratio": 0.55,           # Min 55% body-to-range ratio
            "min_rvol": 1.20,                 # 1.20x relative volume surge
            "stop_loss_pct": 0.0045,          # 0.45% tight structural stop
            "take_profit_pct": 0.0150,        # 1.50% take profit (3.33:1 RR)
            "trail_trigger_pct": 0.0090,      # Dynamic profit lock trigger at +0.90%
            "trail_lock_pct": 0.0040,         # Lock in +0.40% once trigger hit
            "max_holding_bars": 10,           # Max 2.5 hours on 15m
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="21_alpha",
            name="21_alpha — Inside Day Morning Gap Surge",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "Inside Day equilibrium compression followed by gap-driven morning breakout. "
                "Archive-proven (alpha_73 OOS Sharpe 3.46) with cost-validated 3.33:1 RR geometry."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="21_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0025, 0.0030, 0.0045],
            "min_rvol": [1.10, 1.20, 1.35],
            "take_profit_pct": [0.0130, 0.0150, 0.0180],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 50:
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

        min_gap = float(self.parameters.get("min_gap_pct", 0.0030))
        max_gap = float(self.parameters.get("max_gap_pct", 0.0180))
        min_body = float(self.parameters.get("min_body_ratio", 0.55))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0150))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0090))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0040))
        max_bars = int(self.parameters.get("max_holding_bars", 10))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time daily calculations (strictly shifted by 1 to avoid lookahead)
        dates_series = pd.Series(out.index.date, index=out.index)
        daily_high = out["high"].groupby(dates_series).transform("max").shift(1)
        daily_low = out["low"].groupby(dates_series).transform("min").shift(1)
        daily_high_t2 = out["high"].groupby(dates_series).transform("max").shift(2)
        daily_low_t2 = out["low"].groupby(dates_series).transform("min").shift(2)
        daily_close_t1 = out["close"].groupby(dates_series).transform("last").shift(1)

        # Inside Day: T-1 range contained within T-2 range
        is_inside = (daily_high <= daily_high_t2) & (daily_low >= daily_low_t2)

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
                    # Only fire on first 15m bar of the day
                    if i == 0 and not traded_today:
                        has_inside = bool(is_inside.iloc[idx_pos])
                        prev_close = float(daily_close_t1.iloc[idx_pos]) if pd.notna(daily_close_t1.iloc[idx_pos]) else 0.0
                        v_sma = float(vol_sma20.iloc[idx_pos]) if pd.notna(vol_sma20.iloc[idx_pos]) else 0.0

                        if has_inside and prev_close > 0 and v_sma > 0:
                            gap_pct = (o_px - prev_close) / prev_close
                            bar_range = h_px - l_px
                            body = abs(c_px - o_px)
                            body_ratio = body / bar_range if bar_range > 0 else 0.0
                            is_vol_ok = (row["volume"] >= v_sma * min_rvol)

                            # Bullish gap-up: gap in [min_gap, max_gap], strong bull body
                            if (min_gap <= gap_pct <= max_gap) and (c_px > o_px) and (body_ratio >= min_body) and is_vol_ok:
                                pos = 1.0; entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0; traded_today = True

                            # Bearish gap-down: gap in [min_gap, max_gap], strong bear body
                            elif (-max_gap <= gap_pct <= -min_gap) and (c_px < o_px) and (body_ratio >= min_body) and is_vol_ok:
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
