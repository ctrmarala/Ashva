"""
Ashva Quantitative Strategy: Volatility Contraction Open-Equals-Extreme Drive (Alpha 44)
Captures powerful institutional momentum when an equity opens at the exact bar extreme on a post-compression session.

Hypothesis:
When an equity experiences previous-session range contraction (NR2 or Inside Day) and opens with zero adverse
excursion on Bar 1 (Open=Low for Long, Open=High for Short) with volume shock (RVOL >= 1.40x), institutional
programmatic accumulation drives persistent one-way expansion toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha44ContractionOpenDrive(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_44_CONTRACTION_OPEN_DRIVE",
            name="Alpha_44_Contraction_Open_Drive",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "Combining multi-day range compression with a zero-wick institutional opening drive ensures high-conviction "
                "directional participation with minimum adverse excursion."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
        )
        default_params = {
            "max_wick_ratio": 0.05,
            "min_rvol": 1.40,
            "target_rr": 1.50,
            "min_bar_range_pct": 0.0035,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "max_wick_ratio": [0.03, 0.05, 0.08],
            "min_rvol": [1.25, 1.40, 1.60],
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
        is_nr2 = daily_range.shift(1) < daily_range.shift(2)
        is_inside = (daily_summary["day_high"].shift(1) < daily_summary["day_high"].shift(2)) & (daily_summary["day_low"].shift(1) > daily_summary["day_low"].shift(2))
        is_contraction = is_nr2 | is_inside

        out["is_contraction_day"] = pd.Series(dates, index=out.index).map(is_contraction).ffill().fillna(False)

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
        cont_flags = out["is_contraction_day"].values

        max_wick = float(self.parameters.get("max_wick_ratio", 0.05))
        min_rvol = float(self.parameters.get("min_rvol", 1.40))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        min_range_pct = float(self.parameters.get("min_bar_range_pct", 0.0035))

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
                    rationales[i] = "Alpha 44 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or not cont_flags[i]:
                continue

            if bar_time == t_0915:
                bar_range = highs[i] - lows[i]
                if bar_range <= 0 or opens[i] <= 0:
                    continue

                range_pct = bar_range / opens[i]
                if range_pct < min_range_pct:
                    continue

                rvol = volumes[i] / max(1.0, tod_vols[i])
                if rvol < min_rvol:
                    continue

                lower_wick = opens[i] - lows[i]
                upper_wick = highs[i] - opens[i]

                # Bullish Drive on Contraction Day (LONG)
                if (lower_wick / bar_range) <= max_wick and closes[i] > opens[i]:
                    curr_state = 1.0
                    sl = lows[i]
                    risk = closes[i] - sl
                    tp = closes[i] + (target_rr * risk)
                    curr_sl = sl
                    curr_tp = tp
                    curr_rationale = f"Alpha 44 LONG: Contraction Open=Low Drive (Wick={lower_wick/bar_range*100:.1f}%) | RVOL={rvol:.2f}x"
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Bearish Drive on Contraction Day (SHORT)
                elif (upper_wick / bar_range) <= max_wick and closes[i] < opens[i]:
                    curr_state = -1.0
                    sl = highs[i]
                    risk = sl - closes[i]
                    tp = closes[i] - (target_rr * risk)
                    curr_sl = sl
                    curr_tp = tp
                    curr_rationale = f"Alpha 44 SHORT: Contraction Open=High Drive (Wick={upper_wick/bar_range*100:.1f}%) | RVOL={rvol:.2f}x"
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
