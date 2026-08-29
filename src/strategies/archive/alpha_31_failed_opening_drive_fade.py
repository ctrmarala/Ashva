"""
Ashva Quantitative Alpha 31: Failed Opening Drive Auction Rejection Fade
Hypothesis:
    When a stock gaps or drives aggressively on the 09:15 opening bar but forms an
    extreme rejection shadow (upper wick >= 50% on gap up or lower wick >= 50% on gap down),
    institutional liquidity providers have absorbed aggressive retail flow. When the 09:30
    bar confirms acceptance back toward fair value, fading the failed opening drive exhibits
    high-probability mean-reversion drift toward VWAP / prior-day close.

Mechanism:
    Opening auction liquidity absorption and failed expansion mean-reversion fade.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    StrategyHorizon,
    MarketMechanism,
)
from src.features.indicators import TechnicalIndicators


class Alpha31FailedOpeningDriveFade(BaseHypothesis):
    """
    Alpha 31: Opening Auction Absorption & Failed Drive Reversal Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        meta = HypothesisMetadata(
            hypothesis_id="alpha_31",
            name="ALPHA_31_FAILED_OPENING_DRIVE_FADE",
            category="OPENING_MICROSTRUCTURE_FADE",
            economic_rationale=(
                "Aggressive retail gap-chasing is absorbed by institutional limit orders at the opening. "
                "The resulting rejection wick signals order exhaustion, and trapped buyers/sellers liquidate, "
                "driving price back to fair-value VWAP."
            ),
            target_instruments=["NIFTY50_LIQUID"],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
        )
        super().__init__(meta, parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_wick_ratio": [0.45, 0.50, 0.60],
            "min_bar1_range_atr": [0.25, 0.35, 0.45],
            "min_bar1_rvol": [1.10, 1.25, 1.40],
            "target_rr": [1.25, 1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                out.index = pd.to_datetime(out.index)

        dates = out.index.date
        times = out.index.time

        # 1. Daily ATR (14-day) anchored to prior days
        daily_df = out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily_df) >= 14:
            daily_atr_df = TechnicalIndicators.add_atr(daily_df, period=14)
            daily_atr_prev = daily_atr_df["atr_14"].shift(1)
            atr_map = daily_atr_prev.to_dict()
            out["daily_atr"] = [atr_map.get(pd.Timestamp(d), np.nan) for d in dates]
        else:
            out["daily_atr"] = (out["high"] - out["low"]).rolling(14).mean()

        out["daily_atr"] = out["daily_atr"].ffill().bfill()

        # 2. Time-of-Day Mean Volume Baseline
        tod_rolling = out.groupby(times)["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        # 3. Intraday Anchored VWAP
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = (out["cum_pv"] / out["cum_v"].replace(0, np.nan)).bfill().ffill()

        # Strategy Hyperparameters
        min_wick_pct = float(self.parameters.get("min_wick_ratio", 0.50))
        min_range_atr = float(self.parameters.get("min_bar1_range_atr", 0.35))
        min_rvol = float(self.parameters.get("min_bar1_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))

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
        vwaps = out["vwap"].values

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        bar1_open = 0.0
        bar1_high = 0.0
        bar1_low = 0.0
        bar1_close = 0.0
        bar1_vol = 0.0
        bar1_recorded = False

        t_0915 = pd.to_datetime("09:15:00").time()
        t_0930 = pd.to_datetime("09:30:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(1, n):
            bar_date = dates[i]
            bar_time = times[i]

            # Reset on new trading day
            if bar_date != current_day:
                current_day = bar_date
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""
                bar1_recorded = False

            # Intraday 15:15 EOD Square-Off
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 31 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across holding bars
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]
            c_vwap = vwaps[i]

            if pd.isna(c_atr) or c_atr <= 0:
                continue

            # Record Bar 1 (09:15 to 09:30)
            if bar_time == t_0915:
                bar1_open = c_open
                bar1_high = c_high
                bar1_low = c_low
                bar1_close = c_close
                bar1_vol = c_vol
                bar1_recorded = True
                continue

            # Evaluate Reversal at Bar 2 (09:30)
            if bar_time == t_0930 and bar1_recorded and not traded_today:
                bar1_range = bar1_high - bar1_low
                if bar1_range <= 0.01:
                    continue

                rvol_1 = bar1_vol / max(1.0, tod_vols[i - 1])
                range_ratio = bar1_range / c_atr

                # Require significant opening expansion and volume
                if range_ratio < min_range_atr or rvol_1 < min_rvol:
                    continue

                upper_wick = bar1_high - max(bar1_open, bar1_close)
                lower_wick = min(bar1_open, bar1_close) - bar1_low
                upper_wick_ratio = upper_wick / bar1_range
                lower_wick_ratio = lower_wick / bar1_range

                # Case 1: Failed Upward Expansion (Long Upper Rejection Shadow) -> SHORT Fade
                if (upper_wick_ratio >= min_wick_pct) and (c_close < c_open) and (c_close < bar1_close):
                    curr_state = -1.0
                    stop_dist = max(bar1_high - c_close + 0.10 * c_atr, 0.30 * c_atr)
                    curr_sl = c_close + stop_dist
                    curr_tp = c_close - (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 31 SHORT FADE: Bar1 UpperWick={upper_wick_ratio*100:.0f}% (Range={range_ratio:.2f} ATR) | "
                        f"Bar1 RVOL={rvol_1:.2f}x | Bar2 Bearish Close={c_close:.1f} | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
                    )
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Case 2: Failed Downward Expansion (Long Lower Rejection Shadow) -> LONG Fade
                elif (lower_wick_ratio >= min_wick_pct) and (c_close > c_open) and (c_close > bar1_close):
                    curr_state = 1.0
                    stop_dist = max(c_close - bar1_low + 0.10 * c_atr, 0.30 * c_atr)
                    curr_sl = c_close - stop_dist
                    curr_tp = c_close + (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 31 LONG FADE: Bar1 LowerWick={lower_wick_ratio*100:.0f}% (Range={range_ratio:.2f} ATR) | "
                        f"Bar1 RVOL={rvol_1:.2f}x | Bar2 Bullish Close={c_close:.1f} | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
                    )
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
