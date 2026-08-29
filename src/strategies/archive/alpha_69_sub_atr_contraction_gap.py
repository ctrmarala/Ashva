"""
Ashva Quantitative Strategy: Sub-ATR Daily Range Contraction Gap Drift (Alpha 69)
Captures directional intraday trend expansion when Day T-1 range is compressed below 80% of daily ATR on a moderate opening gap.

Hypothesis:
When Day T-1 range is below 80% of its 14-day average true range (Sub-ATR compression) and Day T opens with a moderate gap
(0.35% to 1.20%) on 2.0x volume shock with strong directional body (>= 60%), suppressed daily volatility releases
with strong institutional momentum toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha69SubATRContractionGap(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_69_SUB_ATR_CONTRACTION_GAP",
            name="Alpha_69_Sub_ATR_Contraction_Gap",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "Sub-ATR daily range contraction identifies energy coiled in the prior session. A moderate opening gap "
                "with 2.0x volume triggers fast expansion."
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
            "max_prior_range_atr": 0.80,
            "min_gap_pct": 0.0035,
            "max_gap_pct": 0.0120,
            "min_rvol": 2.00,
            "min_body_ratio": 0.60,
            "target_rr": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0030, 0.0035, 0.0040],
            "min_rvol": [1.75, 2.00, 2.25],
            "target_rr": [1.25, 1.50, 1.75],
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
        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_range
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        prev_range = daily_range.shift(1)
        is_contracted = (prev_range / daily_atr14) <= self.parameters.get("max_prior_range_atr", 0.80)

        out["is_contracted_day"] = pd.Series(dates, index=out.index).map(is_contracted).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()

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
        contracted_flags = out["is_contracted_day"].values
        prev_closes = out["prev_day_close"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0035))
        max_gap = float(self.parameters.get("max_gap_pct", 0.0120))
        min_rvol = float(self.parameters.get("min_rvol", 2.00))
        min_body = float(self.parameters.get("min_body_ratio", 0.60))
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
                    rationales[i] = "Alpha 69 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or not contracted_flags[i] or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap_pct = (opens[i] - prev_closes[i]) / prev_closes[i]
                abs_gap = abs(gap_pct)

                if min_gap <= abs_gap <= max_gap:
                    bar_range = highs[i] - lows[i]
                    body_size = abs(closes[i] - opens[i])
                    rvol = volumes[i] / max(1.0, tod_vols[i])

                    if bar_range > 0 and (body_size / bar_range) >= min_body and rvol >= min_rvol:
                        # Bullish Sub-ATR Gap Drift (LONG)
                        if gap_pct > 0 and closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 69 LONG: Sub-ATR Gap=+{gap_pct*100:.2f}% | RVOL={rvol:.2f}x"
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            rationales[i] = curr_rationale
                            traded_today = True

                        # Bearish Sub-ATR Gap Drift (SHORT)
                        elif gap_pct < 0 and closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 69 SHORT: Sub-ATR Gap={gap_pct*100:.2f}% | RVOL={rvol:.2f}x"
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
