"""
Ashva Quantitative Strategy: Multi-Day Exhaustion Gap Fade Reversion (Alpha 38)
Category: STATISTICAL_MEAN_REVERSION
Market Mechanism: REVERSAL

Hypothesis:
When an equity rallies aggressively for 3 consecutive days and opens with an exhaustion gap (Gap >= +0.75%),
the opening auction represents retail FOMO and institutional profit-taking.
Fading the opening drive when the first 15m candle forms a rejection wick (Shooting Star / Upper Shadow >= 50%)
captures mean reversion back to the previous day close with high statistical expectancy.
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


class Alpha38ExhaustionGapFadeReversion(BaseHypothesis, BaseStrategy):
    strategy_id = "38_alpha"
    hypothesis_id = "38_alpha"
    name = "38_alpha — Multi-Day Exhaustion Gap Fade Reversion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0075,
            "max_gap": 0.0350,
            "min_shadow_ratio": 0.40,
            "target_rr": 1.50,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="38_alpha",
            name="38_alpha — Multi-Day Exhaustion Gap Fade Reversion",
            category="STATISTICAL_MEAN_REVERSION",
            economic_rationale=(
                "3-day extended trends opening with large gaps experience immediate institutional profit-taking. "
                "Fading the exhaustion candle wick captures high-probability mean-reversion."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="38_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap": [0.0075, 0.0100],
            "min_shadow_ratio": [0.35, 0.50],
            "target_rr": [1.50, 1.75],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)

        if n < 60:
            out["signal"] = signals
            out["stop_loss"] = stop_loss
            out["take_profit"] = take_profit
            return out

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time

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

        # 3 consecutive up-days
        three_up_days = (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) & (c.shift(3) > o.shift(3))
        # 3 consecutive down-days
        three_down_days = (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) & (c.shift(3) < o.shift(3))

        prev_close = c.shift(1)

        out["three_up_days"] = pd.Series(dates, index=out.index).map(three_up_days).ffill().fillna(False)
        out["three_down_days"] = pd.Series(dates, index=out.index).map(three_down_days).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()

        t_0915 = time(9, 15)

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        prev_closes = out["prev_day_close"].values
        up_flags = out["three_up_days"].values
        down_flags = out["three_down_days"].values

        min_gap = float(self.parameters.get("min_gap", 0.0075))
        max_gap = float(self.parameters.get("max_gap", 0.0350))
        min_shadow_ratio = float(self.parameters.get("min_shadow_ratio", 0.40))
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
                candle_range = highs[i] - lows[i]

                if min_gap <= abs_gap <= max_gap and candle_range > 0:
                    upper_shadow = highs[i] - max(opens[i], closes[i])
                    lower_shadow = min(opens[i], closes[i]) - lows[i]
                    upper_ratio = upper_shadow / candle_range
                    lower_ratio = lower_shadow / candle_range

                    # Exhaustion Gap-Up Fade (Short): 3 Up Days + Gap Up + Rejection Upper Shadow
                    if up_flags[i] and gap_pct > 0 and upper_ratio >= min_shadow_ratio and closes[i] < opens[i]:
                        sl = highs[i]
                        risk = max(closes[i] * 0.0025, sl - closes[i])
                        signals[i] = -1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] - (target_rr * risk)
                        traded_today = True
                    # Exhaustion Gap-Down Fade (Long): 3 Down Days + Gap Down + Rejection Lower Shadow
                    elif down_flags[i] and gap_pct < 0 and lower_ratio >= min_shadow_ratio and closes[i] > opens[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
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
