"""
Ashva Quantitative Strategy: Daily High-Frequency Momentum Scanner (Alpha 23 - Velocity-50)
Engineered for reliable 5% to 10% monthly ROI through high trade frequency (30-50 trades/month)
across liquid Indian equities.

Hypothesis:
A low-frequency alpha (1 trade/month) suffers from severe statistical variance where luck dominates returns.
To achieve reliable, consistent 5-10% monthly ROI, the strategy must achieve high sample velocity (1-2 trades/day)
by taking opening range momentum breakouts across liquid leaders with 1.75R payoff and tight invalidation stops.
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


class Alpha23Velocity50Scanner(BaseHypothesis):
    """
    Daily High-Frequency Momentum Scanner (Alpha 23):
    1. Daily Range Normalization: Daily ATR(14) strictly from completed sessions.
    2. High-Frequency Breakout Trigger (09:30 to 11:00 IST):
       - Long: 15m Close > Bar 1 High + RVOL >= 1.15x.
       - Short: 15m Close < Bar 1 Low + RVOL >= 1.15x.
    3. Asymmetric Compounding: Target = 1.75R (Risk:Reward = 1:1.75), Stop at Bar 1 Midpoint.
    4. Sizing & High Frequency: 1 trade/day per active stock, 15:15 EOD exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_23_VELOCITY_50_SCANNER",
            name="Alpha_23_Velocity_50_Scanner",
            category="HIGH_FREQUENCY_PORTFOLIO_MOMENTUM",
            economic_rationale=(
                "Single low-frequency trades have high statistical variance. High monthly ROI requires "
                "consistent daily sample execution (1-2 trades/day) across liquid equities taking opening "
                "range breakouts with 1.75R asymmetric payoff and tight midpoint stops."
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
            "min_rvol": 1.15,                  # 1.15x minimum RVOL for high-frequency participation
            "target_rr": 1.75,                 # 1.75R target multiple
            "max_or_atr_ratio": 0.55,          # Maximum Bar 1 Range <= 0.55 * Daily ATR
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.15, 1.20],
            "target_rr": [1.50, 1.75, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 23.
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

        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)
        prev_prev_close = daily_summary["day_close"].shift(2)

        tr1 = prev_high - prev_low
        tr2 = (prev_high - prev_prev_close).abs()
        tr3 = (prev_low - prev_prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean()

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

        min_rvol = float(self.parameters.get("min_rvol", 1.15))
        target_rr = float(self.parameters.get("target_rr", 1.75))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.55))

        current_day = None
        or_high = 0.0
        or_low = 0.0
        or_valid = False
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
                or_valid = False
                traded_today = False

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]

            # -------------------------------------------------------------
            # 1. Establish 09:15 Opening Range (Bar 1)
            # -------------------------------------------------------------
            if hour == 9 and minute == 15:
                if pd.isna(c_atr) or c_atr <= 0:
                    continue

                bar1_range = c_high - c_low
                if (bar1_range > 0.01) and (bar1_range <= max_or_atr * c_atr):
                    or_high = c_high
                    or_low = c_low
                    or_valid = True
                continue

            # -------------------------------------------------------------
            # 2. Breakout Evaluation Window (09:30 to 11:00 IST)
            # -------------------------------------------------------------
            if (not or_valid) or traded_today or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute == 0):
                or_mid = (or_high + or_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish Breakout (LONG)
                if (c_close > or_high) and (c_close > c_open) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    stop_dist = max(c_close - or_mid, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 23 VELOCITY LONG: Breakout Close={c_close:.1f} > OR_High={or_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.2f} RR)"
                    )
                    traded_today = True

                # Bearish Breakdown (SHORT)
                elif (c_close < or_low) and (c_close < c_open) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    stop_dist = max(or_mid - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 23 VELOCITY SHORT: Breakdown Close={c_close:.1f} < OR_Low={or_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.2f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
