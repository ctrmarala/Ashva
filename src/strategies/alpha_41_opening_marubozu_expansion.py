"""
Ashva Quantitative Strategy: Opening Marubozu Institutional Momentum Surge (Alpha 41)
Captures powerful trend continuation when the 09:15-09:30 opening candle forms a full-body Marubozu bar.

Hypothesis:
When an equity's opening 15m candle forms a Marubozu structure (Body >= 75% of candle range)
accompanied by extreme relative volume (RVOL >= 2.00x), institutional block conviction creates a
high-velocity trend that extends toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha41OpeningMarubozuExpansion(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_41_OPENING_MARUBOZU_EXPANSION",
            name="Alpha_41_Opening_Marubozu_Expansion",
            category="MOMENTUM_EXPANSION",
            economic_rationale=(
                "A full-body 15m Marubozu bar (Body >= 75%) with 2.0x volume shock indicates dominant institutional control "
                "with virtually zero two-way auctioning, triggering persistent continuation into 15:15 close."
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
            "min_body_ratio": 0.75,
            "min_rvol": 2.00,
            "target_rr": 1.50,
            "min_bar_range_pct": 0.0040,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_body_ratio": [0.70, 0.75, 0.80],
            "min_rvol": [1.75, 2.00, 2.25],
            "target_rr": [1.25, 1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

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

        min_body = float(self.parameters.get("min_body_ratio", 0.75))
        min_rvol = float(self.parameters.get("min_rvol", 2.00))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        min_range_pct = float(self.parameters.get("min_bar_range_pct", 0.0040))

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
                    rationales[i] = "Alpha 41 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today:
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

                body_size = abs(closes[i] - opens[i])
                body_ratio = body_size / bar_range

                if body_ratio >= min_body:
                    # Bullish Marubozu (LONG)
                    if closes[i] > opens[i]:
                        curr_state = 1.0
                        sl = lows[i]
                        risk = closes[i] - sl
                        tp = closes[i] + (target_rr * risk)
                        curr_sl = sl
                        curr_tp = tp
                        curr_rationale = f"Alpha 41 LONG: Marubozu Body={body_ratio*100:.0f}% | RVOL={rvol:.2f}x | Range={range_pct*100:.2f}%"
                        signals[i] = 1.0
                        stop_loss[i] = curr_sl
                        take_profit[i] = curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

                    # Bearish Marubozu (SHORT)
                    elif closes[i] < opens[i]:
                        curr_state = -1.0
                        sl = highs[i]
                        risk = sl - closes[i]
                        tp = closes[i] - (target_rr * risk)
                        curr_sl = sl
                        curr_tp = tp
                        curr_rationale = f"Alpha 41 SHORT: Marubozu Body={body_ratio*100:.0f}% | RVOL={rvol:.2f}x | Range={range_pct*100:.2f}%"
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
