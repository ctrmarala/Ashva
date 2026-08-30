"""
Ashva Quantitative Alpha Strategy — Alpha 27: Three-Day Trend Surge (3DS-15M)
Alpha ID: 27_alpha
Version: v1.0.0
Author: AshvaQuantLab

Economic Hypothesis:
When a stock exhibits a persistent 3-day directional trend (Close_T1 > Close_T2 > Close_T3 and Low_T1 > Low_T2 > Low_T3 for Long),
an aligned morning opening continuation on Day T with RVOL >= 1.20x confirms institutional momentum persistence.
Using the 09:15 candle extreme as a tight structural stop loss delivers an asymmetric 1.5R target before 15:15 IST.

Contract Specification:
- Mechanism: MOMENTUM
- Timeframe: 15m
- Horizon: INTRADAY (15:15 IST hard square-off)
- Pure 1-bar impulse signal.
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


class Alpha27ThreeDayTrendSurge(BaseHypothesis, BaseStrategy):
    """
    Alpha 27: Three-Day Trend Surge (3DS-15M).
    Multi-session trend persistence with morning volume continuation.
    """

    strategy_id = "27_alpha"
    hypothesis_id = "27_alpha"
    name = "27_alpha — Three-Day Trend Surge"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0025,
            "max_gap": 0.0150,
            "min_rvol": 1.20,
            "min_body": 0.50,
            "target_rr": 1.50,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="27_alpha",
            name="27_alpha — Three-Day Trend Surge",
            category="TREND_MOMENTUM",
            economic_rationale=(
                "Multi-session directional trend continuation confirmed by morning volume surge. "
                "Pure impulse signal with tight 15m candle stop loss."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="27_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap": [0.0025, 0.0035],
            "target_rr": [1.50, 1.75],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)

        if n < 60:
            out["signal"] = signals; out["stop_loss"] = stop_loss; out["take_profit"] = take_profit
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

        # 3-Day Bullish Trend: c(T-1) > c(T-2) > c(T-3) and l(T-1) > l(T-2) > l(T-3)
        is_bull_3d = (c.shift(1) > c.shift(2)) & (c.shift(2) > c.shift(3)) & (l.shift(1) > l.shift(2)) & (l.shift(2) > l.shift(3))
        # 3-Day Bearish Trend: c(T-1) < c(T-2) < c(T-3) and h(T-1) < h(T-2) < h(T-3)
        is_bear_3d = (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3))

        prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["is_bull_3d"] = pd.Series(dates, index=out.index).map(is_bull_3d).ffill().fillna(False)
        out["is_bear_3d"] = pd.Series(dates, index=out.index).map(is_bear_3d).ffill().fillna(False)
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
        bull_3d = out["is_bull_3d"].values
        bear_3d = out["is_bear_3d"].values

        min_gap = float(self.parameters.get("min_gap", 0.0025))
        max_gap = float(self.parameters.get("max_gap", 0.0150))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        min_body = float(self.parameters.get("min_body", 0.50))
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
                body_size = abs(closes[i] - opens[i])
                body_ratio = (body_size / bar_range) if bar_range > 0 else 0.0

                if min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= min_body:
                    if bull_3d[i] and gap_pct > 0 and closes[i] > opens[i] and closes[i] > prev_closes[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
                        traded_today = True

                    elif bear_3d[i] and gap_pct < 0 and closes[i] < opens[i] and closes[i] < prev_closes[i]:
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
        sym = event.symbol; t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)
        if t_str >= "15:15" and pos != 0.0:
            self._current_pos[sym] = 0.0
            return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
        return []
