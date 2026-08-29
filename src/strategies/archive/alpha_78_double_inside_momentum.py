"""
Alpha 78: Double Inside Day Target Expansion (ALPHA_78_DOUBLE_INSIDE_MOMENTUM)
Exploits high-order volatility equilibrium (Mother-Child-Grandchild 2 consecutive Inside Days)
that breaks directionally on elevated morning volume with asymmetric 2.0R target.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    HypothesisStatus,
    StrategyHorizon,
    MarketMechanism,
)


class Alpha78DoubleInsideMomentum(BaseHypothesis):
    """
    Alpha 78: Double Inside Day Target Expansion
    Fires on high conviction breakout following 2 consecutive inside days.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_78_DOUBLE_INSIDE_MOMENTUM",
            name="Alpha_78_Double_Inside_Momentum",
            category="VOLATILITY_EXPANSION",
            economic_rationale=(
                "Two consecutive inside days represent extreme equilibrium compression. "
                "The subsequent opening breakout unleashes strong directional momentum toward extended targets."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.VOLATILITY,
            author="Ashva Autonomous Discovery Factory v3",
        )
        default_params = {
            "min_rvol": 1.25,
            "target_rr": 2.00,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.00, 1.25, 1.50],
            "target_rr": [1.50, 2.00, 2.50],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        if n < 50:
            out["signal"] = signals
            out["stop_loss"] = stop_loss
            out["take_profit"] = take_profit
            out["entry_rationale"] = rationales
            return out

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        is_in = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
        is_din = is_in & (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))
        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)

        out["is_din"] = pd.Series(dates, index=out.index).map(is_din).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()
        out["prev_day_high"] = pd.Series(dates, index=out.index).map(prev_high).ffill()
        out["prev_day_low"] = pd.Series(dates, index=out.index).map(prev_low).ffill()

        tod_vol = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_vol

        t_0915 = pd.to_datetime("09:15:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        prev_closes = out["prev_day_close"].values
        prev_highs = out["prev_day_high"].values
        prev_lows = out["prev_day_low"].values
        din_flags = out["is_din"].values

        min_rvol = self.parameters["min_rvol"]
        target_rr = self.parameters["target_rr"]

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

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
                rvol = volumes[i] / max(1.0, tod_vols[i])

                if din_flags[i] and rvol >= min_rvol:
                    if closes[i] > prev_highs[i] and closes[i] > opens[i]:
                        curr_state = 1.0
                        sl = lows[i]
                        risk = max(closes[i] * 0.002, closes[i] - sl)
                        tp = closes[i] + (target_rr * risk)
                        curr_sl, curr_tp = sl, tp
                        curr_rationale = f"Alpha78 LONG: Double Inside Expansion, RVOL {rvol:.1f}x, RR {target_rr}"
                        signals[i], stop_loss[i], take_profit[i] = 1.0, curr_sl, curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

                    elif closes[i] < prev_lows[i] and closes[i] < opens[i]:
                        curr_state = -1.0
                        sl = highs[i]
                        risk = max(closes[i] * 0.002, sl - closes[i])
                        tp = closes[i] - (target_rr * risk)
                        curr_sl, curr_tp = sl, tp
                        curr_rationale = f"Alpha78 SHORT: Double Inside Expansion, RVOL {rvol:.1f}x, RR {target_rr}"
                        signals[i], stop_loss[i], take_profit[i] = -1.0, curr_sl, curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
