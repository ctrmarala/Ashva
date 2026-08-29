"""
Ashva Quantitative Strategy: NR7 Extreme Volume Gap Expansion (Alpha 53)
Captures powerful trend continuation when 7-day range compression (NR7) meets extreme opening volume shock (RVOL >= 2.0x).

Hypothesis:
When Day T-1 range is the narrowest of the prior 7 sessions (NR7 condition), multi-day equilibrium reaches
absolute maximum tightness. On Day T, an overnight gap (>= 0.30%) breaking Day T-1 high/low validated by extreme
volume shock (RVOL >= 2.00x) and solid candle body (>= 60%) drives persistent institutional momentum toward a 1.75R target,
squared off by 15:15 IST.
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


class Alpha53NR7ExtremeGapExpansion(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_53_NR7_EXTREME_GAP_EXPANSION",
            name="Alpha_53_NR7_Extreme_Gap_Expansion",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "NR7 daily equilibrium combined with 2.0x extreme volume shock on an aligned opening gap triggers "
                "explosive multi-session expansion that easily overcomes Indian statutory cost hurdles."
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
            "min_rvol": 2.00,
            "min_body_ratio": 0.60,
            "target_rr": 1.75,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0025, 0.0030, 0.0035],
            "min_rvol": [1.75, 2.00, 2.25],
            "target_rr": [1.50, 1.75, 2.00],
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

        # NR7 calculation from completed prior sessions (Shifted 1 session)
        is_nr7 = pd.Series(False, index=daily_summary.index)
        for i in range(7, len(daily_summary)):
            t_minus_1_range = daily_range.iloc[i - 1]
            prior_6_ranges = daily_range.iloc[i - 7:i - 1]
            if t_minus_1_range < prior_6_ranges.min():
                is_nr7.iloc[i] = True

        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)

        out["is_nr7_day"] = pd.Series(dates, index=out.index).map(is_nr7).ffill().fillna(False)
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
        nr7_flags = out["is_nr7_day"].values
        prev_closes = out["prev_day_close"].values
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0030))
        min_rvol = float(self.parameters.get("min_rvol", 2.00))
        min_body = float(self.parameters.get("min_body_ratio", 0.60))
        target_rr = float(self.parameters.get("target_rr", 1.75))

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
                    rationales[i] = "Alpha 53 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or not nr7_flags[i] or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap = (opens[i] - prev_closes[i]) / prev_closes[i]
                rvol = volumes[i] / max(1.0, tod_vols[i])
                bar_range = highs[i] - lows[i]
                body_size = abs(closes[i] - opens[i])

                if bar_range > 0 and (body_size / bar_range) >= min_body and rvol >= min_rvol:
                    # Bullish NR7 Extreme Shock (LONG)
                    if gap >= min_gap and closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        curr_state = 1.0
                        sl = lows[i]
                        risk = closes[i] - sl
                        tp = closes[i] + (target_rr * risk)
                        curr_sl = sl
                        curr_tp = tp
                        curr_rationale = f"Alpha 53 LONG: NR7 Extreme Gap Close={closes[i]:.1f} > PDH={prev_highs[i]:.1f} | RVOL={rvol:.2f}x"
                        signals[i] = 1.0
                        stop_loss[i] = curr_sl
                        take_profit[i] = curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

                    # Bearish NR7 Extreme Shock (SHORT)
                    elif gap <= -min_gap and closes[i] < prev_lows[i] and closes[i] < opens[i]:
                        curr_state = -1.0
                        sl = highs[i]
                        risk = sl - closes[i]
                        tp = closes[i] - (target_rr * risk)
                        curr_sl = sl
                        curr_tp = tp
                        curr_rationale = f"Alpha 53 SHORT: NR7 Extreme Gap Close={closes[i]:.1f} < PDL={prev_lows[i]:.1f} | RVOL={rvol:.2f}x"
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
