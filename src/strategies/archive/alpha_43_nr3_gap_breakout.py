"""
Ashva Quantitative Strategy: NR3 Volatility Compression Overnight Gap Breakout (Alpha 43)
Captures directional intraday trend expansion when a 3-day narrow range (NR3) is resolved by an opening gap breakout.

Hypothesis:
When Day T-1 range is narrower than both Day T-2 and Day T-3 (NR3 condition), multi-day volatility compression
stores deep potential energy. On Day T, an overnight gap (>= 0.30%) that opens and closes beyond Day T-1 high/low
with volume (RVOL >= 1.35x) triggers powerful directional expansion toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha43NR3GapBreakout(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_43_NR3_GAP_BREAKOUT",
            name="Alpha_43_NR3_Gap_Breakout",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "NR3 3-day range compression represents prolonged market equilibrium. An opening gap breaking "
                "prior day range on heavy volume triggers powerful multi-session order-flow expansion."
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
            "min_rvol": 1.35,
            "target_rr": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0025, 0.0030, 0.0040],
            "min_rvol": [1.20, 1.35, 1.50],
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

        daily_range = daily_summary["day_high"] - daily_summary["day_low"]
        is_nr3 = (daily_range.shift(1) < daily_range.shift(2)) & (daily_range.shift(1) < daily_range.shift(3))
        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)

        out["is_nr3_day"] = pd.Series(dates, index=out.index).map(is_nr3).ffill().fillna(False)
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
        nr3_flags = out["is_nr3_day"].values
        prev_closes = out["prev_day_close"].values
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0030))
        min_rvol = float(self.parameters.get("min_rvol", 1.35))
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
                    rationales[i] = "Alpha 43 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or not nr3_flags[i] or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap = (opens[i] - prev_closes[i]) / prev_closes[i]
                rvol = volumes[i] / max(1.0, tod_vols[i])

                # Bullish NR3 Breakout
                if gap >= min_gap and closes[i] > prev_highs[i] and closes[i] > opens[i] and rvol >= min_rvol:
                    curr_state = 1.0
                    sl = lows[i]
                    risk = closes[i] - sl
                    tp = closes[i] + (target_rr * risk)
                    curr_sl = sl
                    curr_tp = tp
                    curr_rationale = f"Alpha 43 LONG: NR3 Gap Breakout Close={closes[i]:.1f} > PDH={prev_highs[i]:.1f} | RVOL={rvol:.2f}x"
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Bearish NR3 Breakout
                elif gap <= -min_gap and closes[i] < prev_lows[i] and closes[i] < opens[i] and rvol >= min_rvol:
                    curr_state = -1.0
                    sl = highs[i]
                    risk = sl - closes[i]
                    tp = closes[i] - (target_rr * risk)
                    curr_sl = sl
                    curr_tp = tp
                    curr_rationale = f"Alpha 43 SHORT: NR3 Gap Breakout Close={closes[i]:.1f} < PDL={prev_lows[i]:.1f} | RVOL={rvol:.2f}x"
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
