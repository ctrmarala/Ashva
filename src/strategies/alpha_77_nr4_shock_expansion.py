"""
Alpha 77: NR4 Volatility Shock Expansion (ALPHA_77_NR4_SHOCK_EXPANSION)
Combines 4-day volatility contraction (NR4) with an opening 10-day volume outlier shock.
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


class Alpha77NR4ShockExpansion(BaseHypothesis):
    """
    Alpha 77: NR4 Volatility Shock Expansion
    Combines NR4 contraction with opening volume shock >= 1.2x 10-day bar-1 maximum.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_77_NR4_SHOCK_EXPANSION",
            name="Alpha_77_NR4_Shock_Expansion",
            category="VOLATILITY_EXPANSION",
            economic_rationale=(
                "NR4 4-day range compression followed by an opening bar volume exceeding 10-day bar-1 maximum "
                "creates high probability multi-hour expansion."
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
            "min_gap": 0.0035,
            "max_gap": 0.0140,
            "vol_mult": 1.20,
            "min_body": 0.55,
            "target_rr": 1.50,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap": [0.0020, 0.0035, 0.0050],
            "max_gap": [0.0120, 0.0140, 0.0160],
            "vol_mult": [1.00, 1.20, 1.50],
            "min_body": [0.55, 0.65],
            "target_rr": [1.25, 1.50, 2.00],
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
        daily_range = daily_summary["day_high"] - daily_summary["day_low"]
        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)

        is_nr4_series = pd.Series(False, index=daily_summary.index)
        for d_i in range(4, len(daily_summary)):
            if daily_range.iloc[d_i - 1] < daily_range.iloc[d_i - 4:d_i - 1].min():
                is_nr4_series.iloc[d_i] = True

        out["is_nr4"] = pd.Series(dates, index=out.index).map(is_nr4_series).ffill().fillna(False)
        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()
        out["prev_day_high"] = pd.Series(dates, index=out.index).map(prev_high).ffill()
        out["prev_day_low"] = pd.Series(dates, index=out.index).map(prev_low).ffill()

        # 10-day max bar 1 volume
        bar1_mask = out["time_str"] == "09:15"
        b1_vols = out.loc[bar1_mask, "volume"]
        b1_dates = out.loc[bar1_mask].index.date
        b1_series = pd.Series(b1_vols.values, index=b1_dates)
        max_b1_10d = b1_series.rolling(10, min_periods=5).max().shift(1)
        out["bar1_10d_max_vol"] = pd.Series(dates, index=out.index).map(max_b1_10d).ffill()

        t_0915 = pd.to_datetime("09:15:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        prev_closes = out["prev_day_close"].values
        vol_outliers = out["bar1_10d_max_vol"].values
        nr4_flags = out["is_nr4"].values

        min_gap = self.parameters["min_gap"]
        max_gap = self.parameters["max_gap"]
        vol_mult = self.parameters["vol_mult"]
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
                bar_range = highs[i] - lows[i]
                body_size = abs(closes[i] - opens[i])
                body_ratio = (body_size / bar_range) if bar_range > 0 else 0.0

                if (
                    nr4_flags[i]
                    and min_gap <= abs_gap <= max_gap
                    and not pd.isna(vol_outliers[i])
                    and volumes[i] >= (vol_mult * vol_outliers[i])
                    and body_ratio >= min_body
                ):
                    if gap_pct > 0 and closes[i] > opens[i]:
                        curr_state = 1.0
                        sl = lows[i]
                        risk = max(closes[i] * 0.002, closes[i] - sl)
                        tp = closes[i] + (target_rr * risk)
                        curr_sl, curr_tp = sl, tp
                        curr_rationale = f"Alpha77 LONG: NR4 + Vol Surge {volumes[i]/vol_outliers[i]:.1f}x 10d Max, Gap +{gap_pct*100:.2f}%"
                        signals[i], stop_loss[i], take_profit[i] = 1.0, curr_sl, curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

                    elif gap_pct < 0 and closes[i] < opens[i]:
                        curr_state = -1.0
                        sl = highs[i]
                        risk = max(closes[i] * 0.002, sl - closes[i])
                        tp = closes[i] - (target_rr * risk)
                        curr_sl, curr_tp = sl, tp
                        curr_rationale = f"Alpha77 SHORT: NR4 + Vol Surge {volumes[i]/vol_outliers[i]:.1f}x 10d Max, Gap {gap_pct*100:.2f}%"
                        signals[i], stop_loss[i], take_profit[i] = -1.0, curr_sl, curr_tp
                        rationales[i] = curr_rationale
                        traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
