"""
Ashva Quantitative Strategy: NR4 Daily Volatility Compression Opening Expansion (Alpha 35)
Captures directional intraday trend expansion following a 4-day volatility contraction (NR4).

Hypothesis:
When an equity experiences an NR4 condition (Day T-1 range strictly narrower than Days T-4, T-3, and T-2),
compression stores structural volatility energy. On the subsequent session, an opening range breakout (09:30-10:30 IST)
with elevated relative volume (RVOL >= 1.25x) initiates a directional continuation trend toward a 1.50R target,
squared off by 15:15 IST.
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


class Alpha35NR4OpeningExpansion(BaseHypothesis):
    """
    NR4 Daily Volatility Compression Opening Expansion (Alpha 35):
    1. NR4 Condition: Daily Range(T-1) < min(Daily Range(T-4 ... T-2)) strictly from completed prior sessions.
    2. Opening Range (09:15-09:30 IST): High, Low, Range.
    3. Breakout Window (09:30-10:30 IST):
       - Long: 15m Close > OR15 High + Bullish Bar (Close > Open) + RVOL >= 1.25x.
       - Short: 15m Close < OR15 Low + Bearish Bar (Close < Open) + RVOL >= 1.25x.
    4. Risk & Execution: Next-bar open fill, Stop at OR Extreme (Low for Long, High for Short), Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_35_NR4_OPENING_EXPANSION",
            name="Alpha_35_NR4_Opening_Expansion",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "Following 3 consecutive sessions of contracting daily range (NR4 condition), compressed volatility "
                "stores kinetic market energy. When price breaks out of the initial 15m opening bar with above-average volume, "
                "a directional volatility expansion trend unfolds through the intraday session."
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
            "min_rvol": 1.25,
            "target_rr": 1.50,
            "max_or_atr_ratio": 0.80,
            "min_or_atr_ratio": 0.15,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25, 1.35],
            "target_rr": [1.25, 1.50, 2.00],
            "max_or_atr_ratio": [0.60, 0.80],
            "min_or_atr_ratio": [0.10, 0.15],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 35.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Build Daily Canvas strictly from completed prior sessions (Shifted 1 session)
        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )

        daily_range = daily_summary["day_high"] - daily_summary["day_low"]

        # NR4 calculation: Day T-1 range is strictly smaller than the prior 3 days (T-4 to T-2)
        is_nr4 = pd.Series(False, index=daily_summary.index)
        for i in range(4, len(daily_summary)):
            t_minus_1_range = daily_range.iloc[i - 1]
            prior_3_ranges = daily_range.iloc[i - 4:i - 1]
            if t_minus_1_range < prior_3_ranges.min():
                is_nr4.iloc[i] = True

        # Daily ATR(14) (Shifted 1 session)
        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        out["is_nr4_day"] = pd.Series(dates, index=out.index).map(is_nr4).ffill().fillna(False)
        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()

        # 2. 20-Session TOD Rolling Volume Baseline (Shifted 1 session)
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
        nr4_flags = out["is_nr4_day"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_or_atr_ratio = float(self.parameters.get("max_or_atr_ratio", 0.80))
        min_or_atr_ratio = float(self.parameters.get("min_or_atr_ratio", 0.15))

        current_day = None
        or_high = 0.0
        or_low = 0.0
        or_established = False
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
            hour = bar_time.hour
            minute = bar_time.minute

            if bar_date != current_day:
                current_day = bar_date
                or_high = 0.0
                or_low = 0.0
                or_established = False
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""

            # Square off at 15:15 IST
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 35 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across intraday bars
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
            is_nr4_today = nr4_flags[i]

            # 1. Establish 09:15 Opening Range (Bar 1)
            if bar_time == t_0915:
                or_high = c_high
                or_low = c_low
                or_established = True
                continue

            # 2. Breakout Evaluation Window (09:30 to 10:30 IST on NR4 Days)
            if (not or_established) or (not is_nr4_today) or traded_today or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10 and minute <= 30):
                or_range = or_high - or_low
                if or_range <= 0.01 or or_range > (max_or_atr_ratio * c_atr) or or_range < (min_or_atr_ratio * c_atr):
                    continue

                rvol = c_vol / max(1.0, c_tod)

                # Bullish NR4 Breakout (LONG)
                if (c_close > or_high) and (c_close > c_open) and (rvol >= min_rvol):
                    curr_state = 1.0
                    sl_price = or_low
                    stop_dist = max(c_close - sl_price, 0.15 * c_atr)
                    curr_sl = c_close - stop_dist
                    curr_tp = c_close + (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 35 NR4 LONG: NR4 Expansion Close={c_close:.1f} > OR_High={or_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
                    )
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Bearish NR4 Breakout (SHORT)
                elif (c_close < or_low) and (c_close < c_open) and (rvol >= min_rvol):
                    curr_state = -1.0
                    sl_price = or_high
                    stop_dist = max(sl_price - c_close, 0.15 * c_atr)
                    curr_sl = c_close + stop_dist
                    curr_tp = c_close - (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 35 NR4 SHORT: NR4 Expansion Close={c_close:.1f} < OR_Low={or_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
                    )
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