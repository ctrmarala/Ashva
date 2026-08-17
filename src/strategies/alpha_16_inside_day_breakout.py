"""
Ashva Quantitative Strategy: Inside Day Morning Breakout (Alpha 16 - ID-ORB)
Captures high-probability trend continuation following an Inside Day consolidation equilibrium.

Hypothesis:
An Inside Day (High(T-1) < High(T-2) and Low(T-1) > Low(T-2)) represents total market balance and institutional
order accumulation. When the subsequent session breaks above the Inside Day High or below the Inside Day Low
during the morning session with volume surge, the price momentum expands rapidly toward a 1.50R target.
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


class Alpha16InsideDayBreakout(BaseHypothesis):
    """
    Inside Day Morning Breakout (Alpha 16):
    1. Inside Day Condition: High(T-1) < High(T-2) and Low(T-1) > Low(T-2) strictly from completed sessions.
    2. Morning Breakout Window (09:15-11:00 IST):
       - Long: 15m Close > High(T-1) + Bullish Candle + RVOL >= 1.20x.
       - Short: 15m Close < Low(T-1) + Bearish Candle + RVOL >= 1.20x.
    3. Execution & Risk: Next-bar open fill, Stop at Inside Day Midpoint, Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_16_INSIDE_DAY_BREAKOUT",
            name="Alpha_16_Inside_Day_Breakout",
            category="EQUILIBRIUM_BREAKOUT_MOMENTUM",
            economic_rationale=(
                "An Inside Day represents market compression and institutional accumulation. "
                "When the subsequent session breaks out of the Inside Day boundary during the "
                "morning session with elevated volume, momentum follow-through is highly probable."
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
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 16.
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

        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)
        prev2_high = daily_summary["day_high"].shift(2)
        prev2_low = daily_summary["day_low"].shift(2)

        # Inside Day: Day T-1 is strictly inside Day T-2
        is_inside_day = (prev_high < prev2_high) & (prev_low > prev2_low)
        id_midpoint = (prev_high + prev_low) / 2.0

        # Daily ATR(14) (Shifted 1 session)
        prev_close = daily_summary["day_close"].shift(1)
        prev_prev_close = daily_summary["day_close"].shift(2)
        tr1 = prev_high - prev_low
        tr2 = (prev_high - prev_prev_close).abs()
        tr3 = (prev_low - prev_prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean()

        out["is_inside_day"] = pd.Series(dates, index=out.index).map(is_inside_day).ffill().fillna(False)
        out["id_high"] = pd.Series(dates, index=out.index).map(prev_high).ffill()
        out["id_low"] = pd.Series(dates, index=out.index).map(prev_low).ffill()
        out["id_mid"] = pd.Series(dates, index=out.index).map(id_midpoint).ffill()
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
        id_flags = out["is_inside_day"].values
        id_highs = out["id_high"].values
        id_lows = out["id_low"].values
        id_mids = out["id_mid"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))

        current_day = None
        traded_today = False

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today or not id_flags[i]:
                continue

            # Evaluate strictly in morning window (09:15 to 11:00)
            if (hour == 9 and minute >= 15) or (hour == 10) or (hour == 11 and minute == 0):
                c_close = closes[i]
                c_open = opens[i]
                c_vol = volumes[i]
                c_tod = tod_vols[i]
                c_atr = daily_atrs[i]
                c_id_high = id_highs[i]
                c_id_low = id_lows[i]
                c_id_mid = id_mids[i]

                if pd.isna(c_id_high) or pd.isna(c_atr) or c_atr <= 0:
                    continue

                rvol = c_vol / max(1.0, c_tod)

                # Bullish Inside Day Breakout (LONG)
                if (c_close > c_id_high) and (c_close > c_open) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    stop_dist = max(c_close - c_id_mid, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 16 ID LONG: Close={c_close:.1f} > ID_High={c_id_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish Inside Day Breakdown (SHORT)
                elif (c_close < c_id_low) and (c_close < c_open) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    stop_dist = max(c_id_mid - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 16 ID SHORT: Close={c_close:.1f} < ID_Low={c_id_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
