"""
Ashva Quantitative Strategy: Opening Range ATR Surge Imbalance (Alpha 64)
Captures directional intraday trend expansion when the opening 15m candle consumes >= 40% of daily ATR on 2.25x volume.

Hypothesis:
When an opening 15m candle range covers >= 40% of the entire 14-day daily ATR with heavy volume shock (RVOL >= 2.25x)
and closed decisively near its extreme (candle body >= 70% of bar range), extraordinary institutional order flow
creates strong trend momentum toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha64OpeningRangeATRSurge(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_64_OPENING_RANGE_ATR_SURGE",
            name="Alpha_64_Opening_Range_ATR_Surge",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "Covering 40% of daily ATR in 15 minutes on 2.25x volume represents an extreme order-flow imbalance "
                "that establishes a high-probability directional trend day."
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
            "min_atr_ratio": 0.40,
            "min_rvol": 2.25,
            "min_body_ratio": 0.70,
            "target_rr": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_atr_ratio": [0.35, 0.40, 0.45],
            "min_rvol": [2.00, 2.25, 2.50],
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

        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)
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
        daily_atrs = out["daily_atr"].values

        min_atr_ratio = float(self.parameters.get("min_atr_ratio", 0.40))
        min_rvol = float(self.parameters.get("min_rvol", 2.25))
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
                    rationales[i] = "Alpha 64 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or pd.isna(daily_atrs[i]) or daily_atrs[i] <= 0:
                continue

            if bar_time == t_0915:
                bar_range = highs[i] - lows[i]
                atr = daily_atrs[i]

                if bar_range >= (min_atr_ratio * atr):
                    body_size = abs(closes[i] - opens[i])
                    rvol = volumes[i] / max(1.0, tod_vols[i])

                    if (body_size / bar_range) >= min_body and rvol >= min_rvol:
                        # Bullish ATR Surge (LONG)
                        if closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 64 LONG: Range={bar_range:.1f} >= {min_atr_ratio*100:.0f}% ATR | RVOL={rvol:.2f}x"
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            rationales[i] = curr_rationale
                            traded_today = True

                        # Bearish ATR Surge (SHORT)
                        elif closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 64 SHORT: Range={bar_range:.1f} >= {min_atr_ratio*100:.0f}% ATR | RVOL={rvol:.2f}x"
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
