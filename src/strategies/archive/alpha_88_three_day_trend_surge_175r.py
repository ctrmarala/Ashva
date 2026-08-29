"""
Ashva Quantitative Strategy: 3-Day Momentum Continuation Surge 1.75R (Alpha 88)
Captures broad multi-day momentum continuation with an accessible 1.75R target and moderate volume filter.

Hypothesis:
When an equity exhibits 3 consecutive days of higher highs and higher lows, an aligned opening gap (>= 0.15%)
with moderate opening volume (RVOL >= 1.20x) provides dependable morning trend drift toward a 1.75R target.
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


class Alpha88ThreeDayTrendSurge175R(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_88_THREE_DAY_TREND_SURGE_175R",
            name="Alpha_88_Three_Day_Trend_Surge_175R",
            category="MOMENTUM",
            economic_rationale=(
                "Established 3-day directional accumulation with moderate volume hurdle creates consistent "
                "morning continuation flow toward a 1.75R target."
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
            "min_gap_pct": 0.0015,
            "min_rvol": 1.20,
            "min_body_pct": 0.55,
            "target_rr": 1.75,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0010, 0.0015, 0.0020],
            "min_rvol": [1.10, 1.20, 1.30],
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

        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        c = daily_summary["day_close"]

        # 3-Day Trend: T-1 > T-2 and T-2 > T-3
        is_3d_up = (c.shift(1) > c.shift(2)) & (c.shift(2) > c.shift(3)) & (h.shift(1) > h.shift(2)) & (l.shift(1) > l.shift(2))
        is_3d_dn = (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3)) & (h.shift(1) < h.shift(2)) & (l.shift(1) < l.shift(2))

        prev_close = c.shift(1)

        out["is_3d_up"] = pd.Series(dates, index=out.index).map(is_3d_up).ffill().fillna(False)
        out["is_3d_dn"] = pd.Series(dates, index=out.index).map(is_3d_dn).ffill().fillna(False)
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
        is_up_flags = out["is_3d_up"].values
        is_dn_flags = out["is_3d_dn"].values
        prev_closes = out["prev_day_close"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0015))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        min_body = float(self.parameters.get("min_body_pct", 0.55))
        target_rr = float(self.parameters.get("target_rr", 1.75))

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

                # Bullish 3-Day Trend Surge (LONG)
                if is_up_flags[i] and gap >= min_gap and closes[i] > opens[i] and (body / max(1e-5, bar_range)) >= min_body and rvol >= min_rvol:
                    sl = lows[i]
                    risk = closes[i] - sl
                    if 0.0025 * closes[i] <= risk <= 0.0120 * closes[i]:
                        tp = closes[i] + (target_rr * risk)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = tp
                        rationales[i] = f"Alpha 88 LONG: 3D Trend Surge 1.75R Gap={gap*100:.2f}% RVOL={rvol:.2f}x"
                        traded_today = True

                # Bearish 3-Day Trend Surge (SHORT)
                elif is_dn_flags[i] and gap <= -min_gap and closes[i] < opens[i] and (body / max(1e-5, bar_range)) >= min_body and rvol >= min_rvol:
                    sl = highs[i]
                    risk = sl - closes[i]
                    if 0.0025 * closes[i] <= risk <= 0.0120 * closes[i]:
                        tp = closes[i] - (target_rr * risk)
                        signals[i] = -1.0
                        stop_loss[i] = sl
                        take_profit[i] = tp
                        rationales[i] = f"Alpha 88 SHORT: 3D Trend Surge 1.75R Gap={gap*100:.2f}% RVOL={rvol:.2f}x"
                        traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
