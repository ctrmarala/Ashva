"""
Ashva Quantitative Strategy: Trend Directional Close Spring (Alpha 63)
Category: TREND_DIRECTIONAL_EXPANSION
Market Mechanism: BREAKOUT

Hypothesis:
When an equity is in confirmed 50 SMA trend and Day T-1 closes in the extreme top 30% (for Long)
or extreme bottom 30% (for Short) of its compressed daily range (<= 0.75x ATR), institutional
dominance is established. Day T morning 15m directional breakout leads to sustained continuation.
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


class Alpha63DirectionalCloseTrendSpring(BaseHypothesis, BaseStrategy):
    strategy_id = "63_alpha"
    hypothesis_id = "63_alpha"
    name = "63_alpha — Trend Directional Close Spring"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "max_gap": 0.0250,
            "min_rvol": 1.20,
            "min_body": 0.60,
            "target_rr": 1.60,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="63_alpha",
            name="63_alpha — Trend Directional Close Spring",
            category="TREND_DIRECTIONAL_EXPANSION",
            economic_rationale=(
                "50 SMA trend alignment with Day T-1 close in top/bottom 30% of compressed range "
                "creates high-probability morning directional thrust."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="63_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25],
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
        rng = h - l

        atr14 = rng.shift(2).rolling(14, min_periods=10).mean()
        sma50 = c.shift(2).rolling(50, min_periods=30).mean()

        r = rng.shift(1).replace(0, np.nan)
        close_pos = (c.shift(1) - l.shift(1)) / r
        is_compressed = rng.shift(1) <= (0.75 * atr14)

        bull_trend_spring = (c.shift(1) > sma50) & (close_pos >= 0.70) & is_compressed
        bear_trend_spring = (c.shift(1) < sma50) & (close_pos <= 0.30) & is_compressed

        prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["bull_trend_spring"] = pd.Series(dates, index=out.index).map(bull_trend_spring).ffill().fillna(False)
        out["bear_trend_spring"] = pd.Series(dates, index=out.index).map(bear_trend_spring).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()
        out["prev_day_high"] = pd.Series(dates, index=out.index).map(prev_high).ffill()
        out["prev_day_low"] = pd.Series(dates, index=out.index).map(prev_low).ffill()

        tod_vol = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_vol

        min_gap = self.parameters.get("min_gap", 0.0035)
        max_gap = self.parameters.get("max_gap", 0.0250)
        min_rvol = self.parameters.get("min_rvol", 1.20)
        min_body = self.parameters.get("min_body", 0.60)
        target_rr = self.parameters.get("target_rr", 1.60)
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
        bull_flags = out["bull_trend_spring"].values
        bear_flags = out["bear_trend_spring"].values

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
                    if bull_flags[i] and gap_pct > 0 and closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
                        traded_today = True
                    elif bear_flags[i] and gap_pct < 0 and closes[i] < prev_lows[i] and closes[i] < opens[i]:
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
        return []
