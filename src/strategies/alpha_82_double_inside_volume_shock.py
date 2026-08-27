"""
Ashva Quantitative Strategy: Double Inside Day Volume Shock Expansion (Alpha 82)
Captures directional momentum when a double inside day pattern is broken with >= 1.50x volume shock.

Hypothesis:
When a double inside day pattern is resolved with heavy institutional volume shock (RVOL >= 1.50x),
order flow certainty increases significantly, delivering clean follow-through toward a 1.75R target.
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


class Alpha82DoubleInsideVolumeShock(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_82_DOUBLE_INSIDE_VOLUME_SHOCK",
            name="Alpha_82_Double_Inside_Volume_Shock",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "Combining double inside day compression with 1.50x volume shock filters out low-volume false breakouts "
                "and captures strong institutional trending momentum."
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
            "target_rr": 1.75,
            "min_rvol": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "target_rr": [1.50, 1.75, 2.00],
            "min_rvol": [1.35, 1.50, 1.75],
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

        # Double Inside Day: Day T-1 inside Day T-2, and Day T-2 inside Day T-3
        is_d_inside = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2)) & (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))

        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["is_double_inside"] = pd.Series(dates, index=out.index).map(is_d_inside).ffill().fillna(False)
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
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        double_inside_flags = out["is_double_inside"].values
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values

        target_rr = float(self.parameters.get("target_rr", 1.75))
        min_rvol = float(self.parameters.get("min_rvol", 1.50))

        current_day = None
        traded_today = False

        t_0915 = pd.to_datetime("09:15:00").time()
        t_1030 = pd.to_datetime("10:30:00").time()

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today or not double_inside_flags[i] or pd.isna(prev_highs[i]) or prev_highs[i] <= 0:
                continue

            # Breakout window 09:15 to 10:30 (Impulse Trigger)
            if t_0915 <= bar_time <= t_1030:
                rvol = volumes[i] / max(1.0, tod_vols[i])

                # Bullish Double Inside Breakout (LONG)
                if closes[i] > prev_highs[i] and closes[i] > opens[i] and rvol >= min_rvol:
                    sl = prev_lows[i]
                    risk = closes[i] - sl
                    tp = closes[i] + (target_rr * risk)
                    signals[i] = 1.0
                    stop_loss[i] = sl
                    take_profit[i] = tp
                    rationales[i] = f"Alpha 82 LONG: Double Inside Vol Shock Close={closes[i]:.1f} > PDH={prev_highs[i]:.1f} | RVOL={rvol:.2f}x"
                    traded_today = True

                # Bearish Double Inside Breakout (SHORT)
                elif closes[i] < prev_lows[i] and closes[i] < opens[i] and rvol >= min_rvol:
                    sl = prev_highs[i]
                    risk = sl - closes[i]
                    tp = closes[i] - (target_rr * risk)
                    signals[i] = -1.0
                    stop_loss[i] = sl
                    take_profit[i] = tp
                    rationales[i] = f"Alpha 82 SHORT: Double Inside Vol Shock Close={closes[i]:.1f} < PDL={prev_lows[i]:.1f} | RVOL={rvol:.2f}x"
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
