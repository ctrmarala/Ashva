"""
Ashva Quantitative Strategy: High-Tight Inside Compression Breakout (Alpha 44)
Category: HIGH_TIGHT_VOLATILITY_COMPRESSION
Market Mechanism: BREAKOUT

Hypothesis:
When an equity trading near its 20-day extreme (within 3.5% of 20-day High for Longs or 20-day Low for Shorts)
forms an Inside Day (Day T-1 within Day T-2), institutional inventory is coiled at the threshold of multi-week expansion.
An opening impulse on Day T with RVOL >= 1.20x and a decisive 15m bar close beyond Day T-1's boundary produces
high-velocity breakout continuation with superior risk-reward and low tail drawdown.
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


class Alpha44HighTightInsideCompression(BaseHypothesis, BaseStrategy):
    strategy_id = "44_alpha"
    hypothesis_id = "44_alpha"
    name = "44_alpha — High-Tight Inside Compression"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "max_gap": 0.0220,
            "min_rvol": 1.20,
            "min_body": 0.60,
            "target_rr": 1.50,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="44_alpha",
            name="44_alpha — High-Tight Inside Compression",
            category="HIGH_TIGHT_VOLATILITY_COMPRESSION",
            economic_rationale=(
                "Inside Day compression positioned at multi-week extremes captures pre-expansion coil. "
                "The opening volume drive triggers immediate institutional trend continuation beyond previous resistance."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="44_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25],
            "target_rr": [1.50, 1.75],
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

        # 20-Day Rolling High & Low
        hh20 = h.rolling(20, min_periods=15).max()
        ll20 = l.rolling(20, min_periods=15).min()

        # Inside Day on Day T-1
        is_id = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))

        # Proximity to 20-day high/low (within 3.5%)
        is_high_tight = h.shift(1) >= (0.965 * hh20.shift(2))
        is_low_tight = l.shift(1) <= (1.035 * ll20.shift(2))

        prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["is_id"] = pd.Series(dates, index=out.index).map(is_id).ffill().fillna(False)
        out["is_high_tight"] = pd.Series(dates, index=out.index).map(is_high_tight).ffill().fillna(False)
        out["is_low_tight"] = pd.Series(dates, index=out.index).map(is_low_tight).ffill().fillna(False)
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
        id_flags = out["is_id"].values
        high_tight_flags = out["is_high_tight"].values
        low_tight_flags = out["is_low_tight"].values

        min_gap = float(self.parameters.get("min_gap", 0.0035))
        max_gap = float(self.parameters.get("max_gap", 0.0220))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        min_body = float(self.parameters.get("min_body", 0.60))
        target_rr = float(self.parameters.get("target_rr", 1.50))

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

                if id_flags[i] and min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= min_body:
                    # High-tight Long Breakout
                    if high_tight_flags[i] and gap_pct > 0 and closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
                        traded_today = True
                    # Low-tight Short Breakdown
                    elif low_tight_flags[i] and gap_pct < 0 and closes[i] < prev_lows[i] and closes[i] < opens[i]:
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
