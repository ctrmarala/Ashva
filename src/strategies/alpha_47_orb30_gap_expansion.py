"""
Ashva Quantitative Strategy: 30-Minute Opening Range Gap Expansion (Alpha 47)
Captures powerful intraday trend expansion when an overnight gap is confirmed by a 30-minute range breakout.

Hypothesis:
When an equity opens with an overnight gap (>= 0.35%) and consolidates within the initial 30 minutes (09:15-09:45 IST),
a subsequent 09:45-10:15 breakout in the gap direction with volume (RVOL >= 1.40x) confirms institutional accumulation,
triggering momentum continuation toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha47ORB30GapExpansion(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_47_ORB30_GAP_EXPANSION",
            name="Alpha_47_ORB30_Gap_Expansion",
            category="OPENING_AUCTION",
            economic_rationale=(
                "A 30-minute consolidation after an overnight gap allows initial auction noise to settle. "
                "A breakout at 09:45 with volume confirms institutional continuation flow into 15:15 close."
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
            "min_gap_pct": 0.0035,
            "min_rvol": 1.40,
            "target_rr": 1.50,
            "max_or_atr_ratio": 0.80,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0030, 0.0035, 0.0040],
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

        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()
        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()

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
        prev_closes = out["prev_day_close"].values
        daily_atrs = out["daily_atr"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0035))
        min_rvol = float(self.parameters.get("min_rvol", 1.40))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.80))

        current_day = None
        or30_high = 0.0
        or30_low = 0.0
        or30_established = False
        gap_direction = 0.0
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_0915 = pd.to_datetime("09:15:00").time()
        t_0930 = pd.to_datetime("09:30:00").time()
        t_0945 = pd.to_datetime("09:45:00").time()
        t_1015 = pd.to_datetime("10:15:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]

            if bar_date != current_day:
                current_day = bar_date
                or30_high = 0.0
                or30_low = 0.0
                or30_established = False
                gap_direction = 0.0
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""

            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 47 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            # Bar 1 (09:15) records gap and initial high/low
            if bar_time == t_0915:
                gap = (opens[i] - prev_closes[i]) / prev_closes[i]
                if abs(gap) >= min_gap:
                    gap_direction = 1.0 if gap > 0 else -1.0
                    or30_high = highs[i]
                    or30_low = lows[i]
                continue

            # Bar 2 (09:30) completes 30-min range
            if bar_time == t_0930 and gap_direction != 0.0:
                or30_high = max(or30_high, highs[i])
                or30_low = min(or30_low, lows[i])
                or30_established = True
                continue

            # Breakout Window: 09:45 to 10:15 IST
            if or30_established and (t_0945 <= bar_time <= t_1015):
                atr = daily_atrs[i]
                if pd.isna(atr) or atr <= 0:
                    continue

                or_range = or30_high - or30_low
                if or_range <= 0.01 or or_range > (max_or_atr * atr):
                    continue

                rvol = volumes[i] / max(1.0, tod_vols[i])

                # Bullish ORB-30 Gap Breakout (LONG)
                if gap_direction == 1.0 and closes[i] > or30_high and closes[i] > opens[i] and rvol >= min_rvol:
                    curr_state = 1.0
                    sl = or30_low
                    risk = max(closes[i] - sl, 0.15 * atr)
                    tp = closes[i] + (target_rr * risk)
                    curr_sl = closes[i] - risk
                    curr_tp = tp
                    curr_rationale = f"Alpha 47 LONG: ORB-30 Gap Expansion Close={closes[i]:.1f} > OR30_H={or30_high:.1f} | RVOL={rvol:.2f}x"
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Bearish ORB-30 Gap Breakout (SHORT)
                elif gap_direction == -1.0 and closes[i] < or30_low and closes[i] < opens[i] and rvol >= min_rvol:
                    curr_state = -1.0
                    sl = or30_high
                    risk = max(sl - closes[i], 0.15 * atr)
                    tp = closes[i] - (target_rr * risk)
                    curr_sl = closes[i] + risk
                    curr_tp = tp
                    curr_rationale = f"Alpha 47 SHORT: ORB-30 Gap Expansion Close={closes[i]:.1f} < OR30_L={or30_low:.1f} | RVOL={rvol:.2f}x"
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
