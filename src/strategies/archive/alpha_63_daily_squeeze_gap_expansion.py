"""
Ashva Quantitative Strategy: Daily Bollinger-Keltner Squeeze Gap Breakout (Alpha 63)
Captures explosive directional momentum when a multi-day Bollinger inside Keltner squeeze is released by an aligned opening gap.

Hypothesis:
When a daily Bollinger Band is compressed inside the Keltner Channel (Volatility Squeeze condition) and Day T opens
with an aligned opening gap (>= 0.30%) breaking Day T-1 high/low on 1.75x volume, institutional breakout energy
unleashes with high directional momentum toward a 1.75R target, squared off by 15:15 IST.
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


class Alpha63DailySqueezeGapExpansion(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_63_DAILY_SQUEEZE_GAP_EXPANSION",
            name="Alpha_63_Daily_Squeeze_Gap_Expansion",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "A daily Bollinger-inside-Keltner squeeze represents low historical volatility. An aligned opening gap "
                "with 1.75x volume triggers an explosive volatility expansion cycle."
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
            "min_rvol": 1.75,
            "min_body_ratio": 0.60,
            "target_rr": 1.75,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0025, 0.0030, 0.0035],
            "min_rvol": [1.50, 1.75, 2.00],
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

        c = daily_summary["day_close"]
        h = daily_summary["day_high"]
        l = daily_summary["day_low"]

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        kc_upper = sma20 + 1.5 * atr20
        kc_lower = sma20 - 1.5 * atr20

        is_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        is_squeeze_prev = is_squeeze.shift(1)

        prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)

        out["is_squeeze_day"] = pd.Series(dates, index=out.index).map(is_squeeze_prev).ffill().fillna(False)
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
        squeeze_flags = out["is_squeeze_day"].values
        prev_closes = out["prev_day_close"].values
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0030))
        min_rvol = float(self.parameters.get("min_rvol", 1.75))
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
                    rationales[i] = "Alpha 63 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or not squeeze_flags[i] or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap = (opens[i] - prev_closes[i]) / prev_closes[i]
                rvol = volumes[i] / max(1.0, tod_vols[i])
                bar_range = highs[i] - lows[i]
                body_size = abs(closes[i] - opens[i])

                if bar_range > 0 and (body_size / bar_range) >= min_body and rvol >= min_rvol:
                    # Bullish Squeeze Gap Breakout (LONG)
                    if gap >= min_gap and closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        curr_state = 1.0
                        sl = lows[i]
                        risk = closes[i] - sl
                        tp = closes[i] + (target_rr * risk)
                        curr_sl = sl
                        curr_tp = tp
                        curr_rationale = f"Alpha 63 LONG: Daily Squeeze Gap Close={closes[i]:.1f} > PDH={prev_highs[i]:.1f} | RVOL={rvol:.2f}x"
                        signals[i] = 1.0
                        stop_loss[i] = curr_sl
                        take_profit[i] = curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

                    # Bearish Squeeze Gap Breakout (SHORT)
                    elif gap <= -min_gap and closes[i] < prev_lows[i] and closes[i] < opens[i]:
                        curr_state = -1.0
                        sl = highs[i]
                        risk = sl - closes[i]
                        tp = closes[i] - (target_rr * risk)
                        curr_sl = sl
                        curr_tp = tp
                        curr_rationale = f"Alpha 63 SHORT: Daily Squeeze Gap Close={closes[i]:.1f} < PDL={prev_lows[i]:.1f} | RVOL={rvol:.2f}x"
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
