"""
Ashva Quantitative Strategy: Overnight Gap Opening Marubozu Momentum (Alpha 54)
Captures powerful trend continuation when an overnight gap forms a strong opening marubozu candle.

Hypothesis:
When an equity experiences an overnight gap (0.35% to 1.50%) and Bar 1 (09:15-09:30 IST) closes as a decisive
marubozu candle (body >= 70% of bar range) with heavy volume (RVOL >= 1.80x), aggressive market orders overwhelm
liquidity providers, driving clean directional continuation toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha54GapMarubozuMomentum(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_54_GAP_MARUBOZU_MOMENTUM",
            name="Alpha_54_Gap_Marubozu_Momentum",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "A 70% marubozu candle on an opening gap indicates that buyers/sellers dominated the entire 15m session "
                "from open to close. On 1.80x volume, this signals persistent programmatic execution."
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
            "min_gap_pct": 0.0035,
            "max_gap_pct": 0.0150,
            "min_rvol": 1.80,
            "min_body_ratio": 0.70,
            "target_rr": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0030, 0.0035, 0.0040],
            "min_rvol": [1.60, 1.80, 2.00],
            "min_body_ratio": [0.65, 0.70, 0.75],
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
        prev_closes = out["prev_day_close"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0035))
        max_gap = float(self.parameters.get("max_gap_pct", 0.0150))
        min_rvol = float(self.parameters.get("min_rvol", 1.80))
        min_body = float(self.parameters.get("min_body_ratio", 0.70))
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
                    rationales[i] = "Alpha 54 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue

            if bar_time == t_0915:
                gap_pct = (opens[i] - prev_closes[i]) / prev_closes[i]
                abs_gap = abs(gap_pct)

                if min_gap <= abs_gap <= max_gap:
                    bar_range = highs[i] - lows[i]
                    body_size = abs(closes[i] - opens[i])
                    rvol = volumes[i] / max(1.0, tod_vols[i])

                    if bar_range > 0 and (body_size / bar_range) >= min_body and rvol >= min_rvol:
                        # Bullish Gap Marubozu (LONG)
                        if gap_pct > 0 and closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 54 LONG: Gap Marubozu Gap=+{gap_pct*100:.2f}% | RVOL={rvol:.2f}x | Body={body_size/bar_range*100:.0f}%"
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            rationales[i] = curr_rationale
                            traded_today = True

                        # Bearish Gap Marubozu (SHORT)
                        elif gap_pct < 0 and closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 54 SHORT: Gap Marubozu Gap={gap_pct*100:.2f}% | RVOL={rvol:.2f}x | Body={body_size/bar_range*100:.0f}%"
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
