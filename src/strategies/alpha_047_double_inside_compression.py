"""
Ashva Quantitative Strategy: Double Inside Day Compound Compression (Alpha 47)
Category: DOUBLE_INSIDE_COMPOUND_COMPRESSION
Market Mechanism: BREAKOUT

Hypothesis:
When an equity forms a Double Inside Day (Day T-1 is inside Day T-2, and Day T-2 is inside Day T-3),
market entropy contracts to a multi-day extreme equilibrium. An opening impulse on Day T with RVOL >= 1.15x
and directional close beyond Day T-1's boundary produces violent directional trend-expansion with high post-tax net profit factor.
"""

from typing import Dict, List, Any, Optional
from datetime import time
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


class Alpha47DoubleInsideCompression(BaseHypothesis, BaseStrategy):
    strategy_id = "47_alpha"
    hypothesis_id = "47_alpha"
    name = "47_alpha — Double Inside Day Compound Compression"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "max_gap": 0.0250,
            "min_rvol": 1.15,
            "min_body": 0.55,
            "target_rr": 1.60,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="47_alpha",
            name="47_alpha — Double Inside Day Compound Compression",
            category="DOUBLE_INSIDE_COMPOUND_COMPRESSION",
            economic_rationale=(
                "Two consecutive Inside Days create a rare compound equilibrium. "
                "The opening volume catalyst triggers rapid directional follow-through as coiled inventory unloads."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="47_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.20],
            "target_rr": [1.50, 1.70],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)

        if n < 80:
            out["signal"] = signals
            out["stop_loss"] = stop_loss
            out["take_profit"] = take_profit
            return out

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        c = daily_summary["day_close"]

        # Double Inside Day: Day T-1 inside Day T-2, AND Day T-2 inside Day T-3
        is_id_1 = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
        is_id_2 = (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))
        double_inside = is_id_1 & is_id_2

        prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["double_inside"] = pd.Series(dates, index=out.index).map(double_inside).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()
        out["prev_day_high"] = pd.Series(dates, index=out.index).map(prev_high).ffill()
        out["prev_day_low"] = pd.Series(dates, index=out.index).map(prev_low).ffill()

        tod_vol = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_vol

        t_0915 = time(9, 15)

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        prev_closes = out["prev_day_close"].values
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values
        double_inside_flags = out["double_inside"].values

        min_gap = float(self.parameters.get("min_gap", 0.0035))
        max_gap = float(self.parameters.get("max_gap", 0.0250))
        min_rvol = float(self.parameters.get("min_rvol", 1.15))
        min_body = float(self.parameters.get("min_body", 0.55))
        target_rr = float(self.parameters.get("target_rr", 1.60))

        current_day = None
        traded_today = False

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap_pct = (opens[i] - prev_closes[i]) / prev_closes[i]
                abs_gap = abs(gap_pct)
                rvol = volumes[i] / max(1.0, tod_vols[i])
                bar_range = highs[i] - lows[i]
                body_ratio = (abs(closes[i] - opens[i]) / bar_range) if bar_range > 0 else 0.0

                if double_inside_flags[i] and min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= min_body:
                    if gap_pct > 0 and closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
                        traded_today = True
                    elif gap_pct < 0 and closes[i] < prev_lows[i] and closes[i] < opens[i]:
                        sl = highs[i]
                        risk = max(closes[i] * 0.0025, sl - closes[i])
                        signals[i] = -1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] - (target_rr * risk)
                        traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        return out

    def on_bar(self, event: BarEvent) -> List[SignalEvent]:
        sym = event.symbol
        t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)
        if t_str >= "15:15" and pos != 0.0:
            self._current_pos[sym] = 0.0
            return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
        return []
