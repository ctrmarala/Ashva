"""
Ashva Quantitative Strategy: 14:00 Power Hour Momentum Expansion (Alpha 19)
Captures late-day institutional market-on-close (MOC) and NAV rebalancing momentum.

Hypothesis:
When a stock spends the entire session between 09:30 and 13:45 in tight consolidation (Session Range <= 0.40 * Daily ATR),
a 14:00 breakout with volume surge (RVOL >= 1.50x) triggers aggressive institutional close-of-day order execution,
driving rapid trend expansion through the 15:15 close.
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


class Alpha19PowerHourMomentum(BaseHypothesis):
    """
    14:00 Power Hour Momentum Expansion (Alpha 19):
    1. Consolidation Gate (09:30-13:45 IST): Session Range <= 0.40 * Daily ATR.
    2. Power Hour Window (14:00-14:30 IST):
       - Long: 15m Close > Session High + Bullish Candle + RVOL >= 1.50x.
       - Short: 15m Close < Session Low + Bearish Candle + RVOL >= 1.50x.
    3. Execution & Risk: Next-bar open fill, Stop at Session Midpoint, Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_19_POWER_HOUR_MOMENTUM",
            name="Alpha_19_Power_Hour_Momentum",
            category="LATE_SESSION_MOMENTUM",
            economic_rationale=(
                "Late-day institutional NAV balancing and Market-On-Close order execution create strong "
                "directional momentum in stocks breaking out of all-day consolidation boxes after 14:00."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )
        params = parameters or {
            "min_rvol": 1.50,                  # Volume >= 1.50x shifted TOD baseline
            "target_rr": 1.50,                 # 1.50R target multiple
            "max_session_atr_ratio": 0.40,     # Session range <= 0.40 * Daily ATR
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.30, 1.50, 1.70],
            "target_rr": [1.50, 2.00],
            "max_session_atr_ratio": [0.35, 0.40, 0.45],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 19.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Daily ATR(14) (Shifted 1 session)
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

        min_rvol = float(self.parameters.get("min_rvol", 1.50))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_session_atr = float(self.parameters.get("max_session_atr_ratio", 0.40))

        current_day = None
        sess_high = 0.0
        sess_low = 999999.0
        traded_today = False

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            if bar_date != current_day:
                current_day = bar_date
                sess_high = 0.0
                sess_low = 999999.0
                traded_today = False

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]

            # -------------------------------------------------------------
            # 1. Track All-Day Session Consolidation (09:15 to 13:45 IST)
            # -------------------------------------------------------------
            if (hour < 14) or (hour == 13 and minute <= 45):
                sess_high = max(sess_high, c_high)
                sess_low = min(sess_low, c_low)
                continue

            # -------------------------------------------------------------
            # 2. Power Hour Breakout Window (14:00 to 14:30 IST)
            # -------------------------------------------------------------
            if traded_today or sess_high <= 0 or sess_low >= 999999.0 or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 14 and minute <= 30):
                sess_range = sess_high - sess_low
                if sess_range <= 0.01 or sess_range > (max_session_atr * c_atr):
                    continue

                sess_mid = (sess_high + sess_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish Power Hour Breakout (LONG)
                if (c_close > sess_high) and (c_close > c_open) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    stop_dist = max(c_close - sess_mid, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 19 POWER LONG: Close={c_close:.1f} > Sess_High={sess_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish Power Hour Breakdown (SHORT)
                elif (c_close < sess_low) and (c_close < c_open) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    stop_dist = max(sess_mid - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 19 POWER SHORT: Close={c_close:.1f} < Sess_Low={sess_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
