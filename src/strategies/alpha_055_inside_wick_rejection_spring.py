"""
Ashva Quantitative Strategy: Inside Day Wick Rejection Spring (Alpha 55)
Category: VOLATILITY_ABSORPTION_SPRING
Market Mechanism: BREAKOUT

Hypothesis:
When an equity experiences an Inside Day on Day T-1 with pronounced directional wick absorption
(lower shadow >= 30% of range for bullish setups, indicating aggressive institutional dip-buying at the lows;
upper shadow >= 30% of range for bearish setups, indicating strong institutional supply/rejection at the highs),
the coiled inventory is structurally primed for directional trend continuation.
An opening 15m breakout impulse on Day T with RVOL >= 1.20x and decisive directional candle body (>= 0.60)
triggers explosive trend expansion with superior win rate and institutional profit factor.
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


class Alpha55InsideWickRejectionSpring(BaseHypothesis, BaseStrategy):
    strategy_id = "55_alpha"
    hypothesis_id = "55_alpha"
    name = "55_alpha — Inside Day Wick Rejection Spring"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "max_gap": 0.0220,
            "min_rvol": 1.20,
            "min_body": 0.60,
            "min_wick_ratio": 0.30,
            "target_rr": 1.60,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="55_alpha",
            name="55_alpha — Inside Day Wick Rejection Spring",
            category="VOLATILITY_ABSORPTION_SPRING",
            economic_rationale=(
                "Inside Day with directional wick absorption reveals asymmetrical liquidity absorption "
                "prior to range breakout. The morning auction impulse catalyzes rapid expansion."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="55_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25],
            "min_wick_ratio": [0.25, 0.35],
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
            day_open=("open", "first"),
        )
        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        c = daily_summary["day_close"]
        o = daily_summary["day_open"]
        rng = h - l

        # Inside Day on Day T-1
        is_id = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))

        # Directional wick absorption on Day T-1
        min_wick = float(self.parameters.get("min_wick_ratio", 0.30))
        lower_wick = np.minimum(o.shift(1), c.shift(1)) - l.shift(1)
        upper_wick = h.shift(1) - np.maximum(o.shift(1), c.shift(1))

        cond_long = is_id & (lower_wick >= min_wick * rng.shift(1))
        cond_short = is_id & (upper_wick >= min_wick * rng.shift(1))

        prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["cond_long"] = pd.Series(dates, index=out.index).map(cond_long).ffill().fillna(False)
        out["cond_short"] = pd.Series(dates, index=out.index).map(cond_short).ffill().fillna(False)
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
        cond_long_flags = out["cond_long"].values
        cond_short_flags = out["cond_short"].values

        min_gap = float(self.parameters.get("min_gap", 0.0035))
        max_gap = float(self.parameters.get("max_gap", 0.0220))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        min_body = float(self.parameters.get("min_body", 0.60))
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

                if min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= min_body:
                    if cond_long_flags[i] and gap_pct > 0 and closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
                        traded_today = True
                    elif cond_short_flags[i] and gap_pct < 0 and closes[i] < prev_lows[i] and closes[i] < opens[i]:
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