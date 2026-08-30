"""
Ashva Quantitative Alpha Strategy — Alpha 28: NR4 Volatility Breakout (NR4-15M)
Alpha ID: 28_alpha
Version: v1.0.0
Author: AshvaQuantLab

Economic Hypothesis:
NR4 (Narrowest 4-day range) represents acute volatility compression. When Day T opens with an aligned breakout
of the NR4 extreme on the 09:15 bar with RVOL >= 1.20x and strong body ratio, the multi-session equilibrium
breaks violently toward an asymmetric 1.5R target.

Contract Specification:
- Mechanism: BREAKOUT
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


class Alpha28NR4VolatilityBreakout(BaseHypothesis, BaseStrategy):
    """
    Alpha 28: NR4 Volatility Breakout (NR4-15M).
    Pure 1-bar impulse signal with 15m candle extreme stop loss and 1.5R target.
    """

    strategy_id = "28_alpha"
    hypothesis_id = "28_alpha"
    name = "28_alpha — NR4 Volatility Breakout"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.20,
            "min_body": 0.55,
            "target_rr": 1.50,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="28_alpha",
            name="28_alpha — NR4 Volatility Breakout",
            category="VOLATILITY_EXPANSION",
            economic_rationale=(
                "NR4 4-session range compression breakout with morning volume surge. "
                "Pure impulse signal with tight 15m candle stop loss."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="28_alpha", parameters=merged)
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
        rng = h - l

        # NR4: Range(T-1) < min(Range(T-2), Range(T-3), Range(T-4))
        is_nr4 = (rng.shift(1) < rng.shift(2)) & (rng.shift(1) < rng.shift(3)) & (rng.shift(1) < rng.shift(4))

        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["is_nr4"] = pd.Series(dates, index=out.index).map(is_nr4).ffill().fillna(False)
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
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values
        nr4_flags = out["is_nr4"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        min_body = float(self.parameters.get("min_body", 0.55))
        target_rr = float(self.parameters.get("target_rr", 1.50))

        current_day = None
        traded_today = False

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today or pd.isna(prev_highs[i]) or prev_highs[i] <= 0:
                continue

            if bar_time == t_0915:
                rvol = volumes[i] / max(1.0, tod_vols[i])
                bar_range = highs[i] - lows[i]
                body_size = abs(closes[i] - opens[i])
                body_ratio = (body_size / bar_range) if bar_range > 0 else 0.0

                if nr4_flags[i] and rvol >= min_rvol and body_ratio >= min_body:
                    if closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
                        traded_today = True

                    elif closes[i] < prev_lows[i] and closes[i] < opens[i]:
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
