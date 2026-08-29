"""
Alpha 76: Two-Day Momentum Gap Surge (ALPHA_76_TWO_DAY_MOMENTUM_GAP_SURGE)
Identifies 2 consecutive days of trending closes (Green/Green or Red/Red) that resolve into
an opening gap expansion on above-average morning volume.
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


class Alpha76TwoDayMomentumGapSurge(BaseHypothesis):
    """
    Alpha 76: Two-Day Momentum Gap Surge
    Exploits multi-day institutional trend momentum that continues upon morning gap open.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_76_TWO_DAY_MOMENTUM_GAP_SURGE",
            name="Alpha_76_Two_Day_Momentum_Gap_Surge",
            category="SWING_MOMENTUM",
            economic_rationale=(
                "Two consecutive days of directional institutional accumulation create multi-day trend momentum "
                "that accelerates when opened in the direction of the trend on elevated volume."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="Ashva Autonomous Discovery Factory v3",
        )
        default_params = {
            "min_gap": 0.0025,
            "max_gap": 0.0140,
            "min_rvol": 1.25,
            "min_body": 0.55,
            "target_rr": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap": [0.0020, 0.0025, 0.0040],
            "max_gap": [0.0120, 0.0140, 0.0180],
            "min_rvol": [1.00, 1.25, 1.50],
            "min_body": [0.55, 0.65],
            "target_rr": [1.25, 1.50, 1.75],
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
            day_open=("open", "first"),
            day_close=("close", "last"),
        )
        is_2g = (daily_summary["day_close"].shift(1) > daily_summary["day_open"].shift(1)) & (daily_summary["day_close"].shift(2) > daily_summary["day_open"].shift(2))
        is_2r = (daily_summary["day_close"].shift(1) < daily_summary["day_open"].shift(1)) & (daily_summary["day_close"].shift(2) < daily_summary["day_open"].shift(2))
        prev_close = daily_summary["day_close"].shift(1)

        out["is_2g"] = pd.Series(dates, index=out.index).map(is_2g).ffill().fillna(False)
        out["is_2r"] = pd.Series(dates, index=out.index).map(is_2r).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()

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
        is_2g_vals = out["is_2g"].values
        is_2r_vals = out["is_2r"].values

        min_gap = self.parameters["min_gap"]
        max_gap = self.parameters["max_gap"]
        min_rvol = self.parameters["min_rvol"]
        min_body = self.parameters["min_body"]
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
                gap_pct = (opens[i] - prev_closes[i]) / prev_closes[i]
                abs_gap = abs(gap_pct)
                rvol = volumes[i] / max(1.0, tod_vols[i])
                bar_range = highs[i] - lows[i]
                body_size = abs(closes[i] - opens[i])
                body_ratio = (body_size / bar_range) if bar_range > 0 else 0.0

                if min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= min_body:
                    if is_2g_vals[i] and gap_pct > 0 and closes[i] > opens[i]:
                        curr_state = 1.0
                        sl = lows[i]
                        risk = max(closes[i] * 0.002, closes[i] - sl)
                        tp = closes[i] + (target_rr * risk)
                        curr_sl, curr_tp = sl, tp
                        curr_rationale = f"Alpha76 LONG: 2G Trend Day Gap Surge +{gap_pct*100:.2f}%, RVOL {rvol:.1f}x"
                        signals[i], stop_loss[i], take_profit[i] = 1.0, curr_sl, curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

                    elif is_2r_vals[i] and gap_pct < 0 and closes[i] < opens[i]:
                        curr_state = -1.0
                        sl = highs[i]
                        risk = max(closes[i] * 0.002, sl - closes[i])
                        tp = closes[i] - (target_rr * risk)
                        curr_sl, curr_tp = sl, tp
                        curr_rationale = f"Alpha76 SHORT: 2R Trend Day Gap Surge {gap_pct*100:.2f}%, RVOL {rvol:.1f}x"
                        signals[i], stop_loss[i], take_profit[i] = -1.0, curr_sl, curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
