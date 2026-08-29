"""
Ashva Quantitative Strategy: 10-Day Volume Outlier Opening Drive (Alpha 50)
Captures powerful trend continuation when the 09:15 opening candle volume breaks the 10-day opening volume high.

Hypothesis:
When an equity's opening 15m candle volume exceeds the maximum opening 15m volume of the prior 10 sessions
by >= 25% with strong directional body (>= 60%), massive institutional order-flow concentration drives persistent
trending drift toward a 1.50R target, squared off by 15:15 IST.
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


class Alpha50OutlierVolumeOpeningDrive(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_50_OUTLIER_VOLUME_OPENING_DRIVE",
            name="Alpha_50_Outlier_Volume_Opening_Drive",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "When opening 15m volume surpasses the 10-day maximum opening volume by 25%, it represents institutional "
                "conviction that overrides intra-day mean-reversion forces, yielding high-probability drift."
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
            "vol_outlier_ratio": 1.25,
            "min_body_ratio": 0.60,
            "target_rr": 1.50,
            "min_bar_range_pct": 0.0035,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "vol_outlier_ratio": [1.15, 1.25, 1.40],
            "min_body_ratio": [0.55, 0.60, 0.65],
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

        # Rolling 10-session maximum of 09:15 volume (Shifted 1 session)
        bar1_mask = out["time_str"] == "09:15"
        bar1_vols = out.loc[bar1_mask, "volume"]
        bar1_dates = out.loc[bar1_mask].index.date
        bar1_daily_series = pd.Series(bar1_vols.values, index=bar1_dates)
        rolling_max_10d = bar1_daily_series.rolling(10, min_periods=5).max().shift(1)
        out["bar1_10d_max_vol"] = pd.Series(dates, index=out.index).map(rolling_max_10d).ffill()

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
        bar1_10d_maxs = out["bar1_10d_max_vol"].values

        outlier_ratio = float(self.parameters.get("vol_outlier_ratio", 1.25))
        min_body = float(self.parameters.get("min_body_ratio", 0.60))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        min_range_pct = float(self.parameters.get("min_bar_range_pct", 0.0035))

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
                    rationales[i] = "Alpha 50 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today:
                continue

            if bar_time == t_0915 and not np.isnan(bar1_10d_maxs[i]) and bar1_10d_maxs[i] > 0:
                if volumes[i] >= outlier_ratio * bar1_10d_maxs[i]:
                    bar_range = highs[i] - lows[i]
                    if bar_range <= 0 or opens[i] <= 0:
                        continue

                    range_pct = bar_range / opens[i]
                    if range_pct < min_range_pct:
                        continue

                    body_size = abs(closes[i] - opens[i])
                    if (body_size / bar_range) >= min_body:
                        # Bullish Outlier Drive (LONG)
                        if closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 50 LONG: 10d Outlier Vol={volumes[i]:,.0f} > 10dMax={bar1_10d_maxs[i]:,.0f} | Body={body_size/bar_range*100:.0f}%"
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            rationales[i] = curr_rationale
                            traded_today = True

                        # Bearish Outlier Drive (SHORT)
                        elif closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            curr_rationale = f"Alpha 50 SHORT: 10d Outlier Vol={volumes[i]:,.0f} > 10dMax={bar1_10d_maxs[i]:,.0f} | Body={body_size/bar_range*100:.0f}%"
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
