"""
Ashva Quantitative Strategy: Daily NR7 Opening Volatility Expansion (Alpha 15 - NR7-ORB)
Captures powerful directional trend expansion following a multi-session volatility contraction (NR7).

Hypothesis:
When a stock's daily trading range is the narrowest of the last 7 sessions (NR7), price compression reaches
maximum potential energy. On the subsequent session, an opening range breakout with volume expansion triggers
institutional delta-hedging and momentum follow-through to a 1.50R target.
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


class Alpha15NR7VolatilityExpansion(BaseHypothesis):
    """
    Daily NR7 Opening Volatility Expansion (Alpha 15):
    1. NR7 Condition: Daily Range(T-1) < min(Daily Range(T-7 ... T-2)) strictly from completed sessions.
    2. Opening Range (09:15-09:30 IST): High, Low, Range.
    3. Breakout Window (09:30-11:00 IST):
       - Long: 15m Close > OR15 High + Bullish Candle + RVOL >= 1.20x.
       - Short: 15m Close < OR15 Low + Bearish Candle + RVOL >= 1.20x.
    4. Execution & Risk: Next-bar open fill, Stop at OR15 Midpoint, Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_15_NR7_VOLATILITY_EXPANSION",
            name="Alpha_15_NR7_Volatility_Expansion",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "When a stock's daily range is the narrowest of the last 7 sessions (NR7), volatility "
                "compression reaches maximum potential energy. On the subsequent session, an opening "
                "range breakout with volume expansion triggers aggressive momentum follow-through."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )
        params = parameters or {
            "min_rvol": 1.20,                  # Volume >= 1.20x shifted TOD baseline
            "target_rr": 1.50,                 # 1.50R target multiple
            "max_or_atr_ratio": 0.50,          # OR15 range <= 0.50 * Daily ATR
            "stop_mode": "midpoint",           # "midpoint" or "opposite"
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00],
            "max_or_atr_ratio": [0.40, 0.50, 0.60],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 15.
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

        # NR7 calculation: Day T-1 range is strictly smaller than the prior 6 days (T-7 to T-2)
        is_nr7 = pd.Series(False, index=daily_summary.index)
        for i in range(7, len(daily_summary)):
            t_minus_1_range = daily_range.iloc[i - 1]
            prior_6_ranges = daily_range.iloc[i - 7:i - 1]
            if t_minus_1_range < prior_6_ranges.min():
                is_nr7.iloc[i] = True

        # Daily ATR(14) (Shifted 1 session)
        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        out["is_nr7_day"] = pd.Series(dates, index=out.index).map(is_nr7).ffill().fillna(False)
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
        nr7_flags = out["is_nr7_day"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.50))
        stop_mode = self.parameters.get("stop_mode", "midpoint")

        current_day = None
        or_high = 0.0
        or_low = 0.0
        or_established = False
        traded_today = False

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

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]
            is_nr7_today = nr7_flags[i]

            # -------------------------------------------------------------
            # 1. Establish 09:15 Opening Range (Bar 1)
            # -------------------------------------------------------------
            if hour == 9 and minute == 15:
                or_high = c_high
                or_low = c_low
                or_established = True
                continue

            # -------------------------------------------------------------
            # 2. Breakout Evaluation Window (09:30 to 11:00 IST on NR7 Days)
            # -------------------------------------------------------------
            if (not or_established) or (not is_nr7_today) or traded_today or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute == 0):
                or_range = or_high - or_low
                if or_range <= 0.01 or or_range > (max_or_atr * c_atr):
                    continue

                or_mid = (or_high + or_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish NR7 Breakout (LONG)
                if (c_close > or_high) and (c_close > c_open) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    sl_price = or_mid if stop_mode == "midpoint" else or_low
                    stop_dist = max(c_close - sl_price, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 15 NR7 LONG: NR7 Expansion Close={c_close:.1f} > OR_High={or_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish NR7 Breakdown (SHORT)
                elif (c_close < or_low) and (c_close < c_open) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    sl_price = or_mid if stop_mode == "midpoint" else or_high
                    stop_dist = max(sl_price - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 15 NR7 SHORT: NR7 Expansion Close={c_close:.1f} < OR_Low={or_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
