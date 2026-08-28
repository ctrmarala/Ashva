"""
Ashva Quantitative Strategy: 2-Day Momentum Continuation Surge 2.0R (Alpha 87)
Captures directional momentum continuation when a 2-day established trend opens with an aligned surge.

Hypothesis:
When an equity exhibits 2 consecutive sessions of higher highs and higher lows (established short-term trend),
an aligned opening gap (>= 0.20%) and strong opening expansion bar (RVOL >= 1.35x) triggers high-probability
directional follow-through towards a 2.0R target with superior risk-adjusted net returns.
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


class Alpha87TwoDayTrendSurge2R(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_87_TWO_DAY_TREND_SURGE_2R",
            name="Alpha_87_Two_Day_Trend_Surge_2R",
            category="MOMENTUM",
            economic_rationale=(
                "Two consecutive sessions of directional accumulation followed by an aligned opening drive "
                "creates high trend follow-through velocity with an asymmetric 2.0R payoff."
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
            "min_gap_pct": 0.0020,
            "min_rvol": 1.35,
            "min_body_pct": 0.60,
            "target_rr": 2.00,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0015, 0.0020, 0.0025],
            "min_rvol": [1.20, 1.35, 1.50],
            "target_rr": [1.75, 2.00, 2.25],
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
        c = daily_summary["day_close"]

        # 2-Day Trend: T-1 > T-2
        is_2d_up = (c.shift(1) > c.shift(2)) & (h.shift(1) > h.shift(2)) & (l.shift(1) > l.shift(2))
        is_2d_dn = (c.shift(1) < c.shift(2)) & (h.shift(1) < h.shift(2)) & (l.shift(1) < l.shift(2))

        prev_close = c.shift(1)

        out["is_2d_up"] = pd.Series(dates, index=out.index).map(is_2d_up).ffill().fillna(False)
        out["is_2d_dn"] = pd.Series(dates, index=out.index).map(is_2d_dn).ffill().fillna(False)
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
        is_up_flags = out["is_2d_up"].values
        is_dn_flags = out["is_2d_dn"].values
        prev_closes = out["prev_day_close"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0020))
        min_rvol = float(self.parameters.get("min_rvol", 1.35))
        min_body = float(self.parameters.get("min_body_pct", 0.60))
        target_rr = float(self.parameters.get("target_rr", 2.00))

        current_day = None
        traded_today = False
        t_0915 = pd.to_datetime("09:15:00").time()

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap = (opens[i] - prev_closes[i]) / prev_closes[i]
                rvol = volumes[i] / max(1.0, tod_vols[i])
                body = abs(closes[i] - opens[i])
                bar_range = highs[i] - lows[i]

                # Bullish 2-Day Trend Surge (LONG)
                if is_up_flags[i] and gap >= min_gap and closes[i] > opens[i] and (body / max(1e-5, bar_range)) >= min_body and rvol >= min_rvol:
                    sl = lows[i]
                    risk = closes[i] - sl
                    if 0.0025 * closes[i] <= risk <= 0.0120 * closes[i]:
                        tp = closes[i] + (target_rr * risk)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = tp
                        rationales[i] = f"Alpha 87 LONG: 2D Trend Surge Gap={gap*100:.2f}% RVOL={rvol:.2f}x"
                        traded_today = True

                # Bearish 2-Day Trend Surge (SHORT)
                elif is_dn_flags[i] and gap <= -min_gap and closes[i] < opens[i] and (body / max(1e-5, bar_range)) >= min_body and rvol >= min_rvol:
                    sl = highs[i]
                    risk = sl - closes[i]
                    if 0.0025 * closes[i] <= risk <= 0.0120 * closes[i]:
                        tp = closes[i] - (target_rr * risk)
                        signals[i] = -1.0
                        stop_loss[i] = sl
                        take_profit[i] = tp
                        rationales[i] = f"Alpha 87 SHORT: 2D Trend Surge Gap={gap*100:.2f}% RVOL={rvol:.2f}x"
                        traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
