"""
Ashva Quantitative Strategy: High-Velocity Portfolio Momentum (Alpha 21 - HV-Momentum)
Engineered for high monthly ROI generation via dynamic multi-asset selection, 2.0R asymmetric payoff,
and daily high-velocity execution across top liquid market leaders.

Hypothesis:
To generate institutional-grade 5-10% monthly ROI, systematic trading must operate at high statistical velocity:
1. Daily Cross-Sectional Leader Selection (Top 2 most explosive opening momentum stocks).
2. Clean 15m Opening Range Breakout with RVOL >= 1.30x.
3. Asymmetric 2.0R Target with tight Invalidation Stop at OR15 Midpoint.
4. Active intraday risk allocation (1.5% account risk per trade with MIS leverage).
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


class Alpha21HighVelocityMomentum(BaseHypothesis):
    """
    High-Velocity Portfolio Momentum (Alpha 21):
    1. Opening Directional Filter (09:15-09:30): Bar 1 Range >= 0.35 * Daily ATR, Body >= 50%, RVOL >= 1.25x.
    2. High-Velocity Breakout Window (09:30 to 10:45 IST):
       - Long: 15m Close > OR15 High + Bullish Candle.
       - Short: 15m Close < OR15 Low + Bearish Candle.
    3. Asymmetric Payoff: Target = 2.0R (Risk:Reward = 1:2.0), Stop at OR15 Midpoint.
    4. Intraday MIS Leverage & Sizing: 1.5% Risk per trade, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_21_HIGH_VELOCITY_MOMENTUM",
            name="Alpha_21_High_Velocity_Momentum",
            category="HIGH_VELOCITY_PORTFOLIO_MOMENTUM",
            economic_rationale=(
                "High monthly ROI requires high statistical trade velocity and asymmetric payoff (2.0R). "
                "By capturing early-morning institutional momentum breakouts on top liquid leaders with "
                "a tight midpoint invalidation stop, the strategy maximizes monthly capital compounding."
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
            "min_or_atr_ratio": 0.25,          # Min Bar 1 Range >= 0.25 * Daily ATR (adequate volatility)
            "max_or_atr_ratio": 0.60,          # Max Bar 1 Range <= 0.60 * Daily ATR (prevent exhaustion)
            "min_body_ratio": 0.45,            # Bar 1 Body >= 45% of range
            "min_rvol": 1.20,                  # RVOL >= 1.20x shifted TOD baseline
            "target_rr": 2.00,                 # 2.0R target multiple for asymmetric compounding
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_or_atr_ratio": [0.20, 0.25, 0.30],
            "min_rvol": [1.15, 1.20, 1.30],
            "target_rr": [1.75, 2.00, 2.50],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 21.
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

        min_or_atr = float(self.parameters.get("min_or_atr_ratio", 0.25))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.60))
        min_body = float(self.parameters.get("min_body_ratio", 0.45))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 2.00))

        current_day = None
        or_high = 0.0
        or_low = 0.0
        or_valid = False
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
                or_valid = False
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""

            # Intraday 15:15 EOD Square-Off
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 21 EXIT: Intraday 15:15 EOD Square-Off"
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

            # -------------------------------------------------------------
            # 1. Establish 09:15 Opening Range (Bar 1)
            # -------------------------------------------------------------
            if bar_time == t_0915:
                if pd.isna(c_atr) or c_atr <= 0:
                    continue

                bar1_range = c_high - c_low
                bar1_body = abs(c_close - c_open)
                bar1_body_ratio = bar1_body / max(bar1_range, 0.01)
                bar1_atr_ratio = bar1_range / c_atr

                # Validate Bar 1 expansion quality
                if (min_or_atr <= bar1_atr_ratio <= max_or_atr) and (bar1_body_ratio >= min_body):
                    or_high = c_high
                    or_low = c_low
                    or_valid = True
                continue

            # -------------------------------------------------------------
            # 2. High-Velocity Breakout Window (09:30 to 10:45 IST)
            # -------------------------------------------------------------
            if (not or_valid) or traded_today or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10 and minute <= 45):
                or_mid = (or_high + or_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish Breakout (LONG)
                if (c_close > or_high) and (c_close > c_open) and (rvol >= min_rvol):
                    curr_state = 1.0
                    stop_dist = max(c_close - or_mid, 0.15 * c_atr)
                    curr_sl = c_close - stop_dist
                    curr_tp = c_close + (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 21 HV LONG: Breakout Close={c_close:.1f} > OR_High={or_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                    )
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Bearish Breakdown (SHORT)
                elif (c_close < or_low) and (c_close < c_open) and (rvol >= min_rvol):
                    curr_state = -1.0
                    stop_dist = max(or_mid - c_close, 0.15 * c_atr)
                    curr_sl = c_close + stop_dist
                    curr_tp = c_close - (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 21 HV SHORT: Breakdown Close={c_close:.1f} < OR_Low={or_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                    )
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
