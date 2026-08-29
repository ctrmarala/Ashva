"""
Ashva Quantitative Strategy: Inside Day Opening Gap Momentum Acceleration (Alpha 45)
Captures powerful directional momentum when an Inside Day equilibrium is resolved by an aligned opening gap.

Hypothesis:
When an equity forms an Inside Day (Day T-1 High < Day T-2 High and Day T-1 Low > Day T-2 Low),
two-session market equilibrium reaches maximum tightness. On Day T, an overnight gap (>= 0.30%) that opens
and closes beyond the Day T-1 extreme with volume shock (RVOL >= 1.50x) triggers rapid institutional delta-hedging
and sustained directional expansion toward a 1.50R target, squared off by 15:15 IST.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    HypothesisStatus,
    StrategyHorizon,
    MarketMechanism,
)


class Alpha45InsideDayGapMomentum(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_45_INSIDE_DAY_GAP_MOMENTUM",
            name="Alpha_45_Inside_Day_Gap_Momentum",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "An Inside Day compression resolved by an aligned overnight gap forces immediate capitulation of "
                "prior range-bound positions, driving powerful one-way expansion with minimal adverse excursion."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        default_params = {
            "min_gap_pct": 0.0030,
            "min_rvol": 1.50,
            "target_rr": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0025, 0.0030, 0.0040],
            "min_rvol": [1.35, 1.50, 1.75],
            "target_rr": [1.25, 1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )

        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        is_inside = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)

        out["is_inside_day"] = pd.Series(dates, index=out.index).map(is_inside).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()
        out["prev_day_high"] = pd.Series(dates, index=out.index).map(prev_high).ffill()
        out["prev_day_low"] = pd.Series(dates, index=out.index).map(prev_low).ffill()

        tod_rolling = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        inside_flags = out["is_inside_day"].values
        prev_closes = out["prev_day_close"].values
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0030))
        min_rvol = float(self.parameters.get("min_rvol", 1.50))
        target_rr = float(self.parameters.get("target_rr", 1.50))

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_0915 = pd.to_datetime("09:15:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""

            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 45 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or not inside_flags[i] or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap = (opens[i] - prev_closes[i]) / prev_closes[i]
                rvol = volumes[i] / max(1.0, tod_vols[i])

                # Bullish Inside Day Gap Breakout (LONG)
                if gap >= min_gap and closes[i] > prev_highs[i] and closes[i] > opens[i] and rvol >= min_rvol:
                    curr_state = 1.0
                    sl = lows[i]
                    risk = closes[i] - sl
                    tp = closes[i] + (target_rr * risk)
                    curr_sl = sl
                    curr_tp = tp
                    curr_rationale = f"Alpha 45 LONG: Inside Day Gap Breakout Close={closes[i]:.1f} > PDH={prev_highs[i]:.1f} | RVOL={rvol:.2f}x"
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Bearish Inside Day Gap Breakout (SHORT)
                elif gap <= -min_gap and closes[i] < prev_lows[i] and closes[i] < opens[i] and rvol >= min_rvol:
                    curr_state = -1.0
                    sl = highs[i]
                    risk = sl - closes[i]
                    tp = closes[i] - (target_rr * risk)
                    curr_sl = sl
                    curr_tp = tp
                    curr_rationale = f"Alpha 45 SHORT: Inside Day Gap Breakout Close={closes[i]:.1f} < PDL={prev_lows[i]:.1f} | RVOL={rvol:.2f}x"
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
