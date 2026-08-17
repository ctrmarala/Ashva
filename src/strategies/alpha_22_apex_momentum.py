"""
Ashva Quantitative Strategy: Apex Intraday Volatility Impulse (Alpha 22 - Standalone High-Yield Alpha)
Engineered to generate 5% to 10% monthly ROI via high-frequency asymmetric 2.5R momentum breakouts.

Hypothesis:
High monthly ROI from a single standalone alpha requires high statistical trade velocity, high win rate,
and asymmetric payoff (2.5R). When a liquid equity establishes a high-energy opening range (Range >= 0.35 * ATR,
RVOL >= 1.25x) and breaks out before 10:30 with SuperTrend alignment, institutional trend continuation creates
large multi-R directional expansion.
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


class Alpha22ApexMomentum(BaseHypothesis):
    """
    Apex Intraday Volatility Impulse (Alpha 22 - Standalone High-Yield Alpha):
    1. Dynamic Daily ATR(14) Baseline strictly from prior completed sessions.
    2. High-Energy Opening Range (09:15-09:30 IST): Range >= 0.30 * Daily ATR, Body >= 40%, RVOL >= 1.20x.
    3. Breakout Window (09:30-10:45 IST):
       - Long: 15m Close > OR15 High + Bullish Candle + 15m Close > 15m EMA(20).
       - Short: 15m Close < OR15 Low + Bearish Candle + 15m Close < 15m EMA(20).
    4. Asymmetric Risk/Reward: Target = 2.50R (1:2.5 RR), Invalidation Stop at OR15 Midpoint.
    5. Intraday MIS Sizing: 1.50% account risk per trade, Mandatory 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_22_APEX_MOMENTUM",
            name="Alpha_22_Apex_Momentum",
            category="STANDALONE_HIGH_YIELD_MOMENTUM",
            economic_rationale=(
                "High monthly ROI requires capturing explosive multi-R trend expansion (2.5R) with tight "
                "midpoint stops on high-volume opening breakouts. Institutional order flow following opening range "
                "expansion drives persistent intraday follow-through to deliver 5-10% monthly portfolio ROI."
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
            "min_or_atr_ratio": 0.30,          # Minimum Bar 1 Range >= 0.30 * Daily ATR
            "max_or_atr_ratio": 0.65,          # Maximum Bar 1 Range <= 0.65 * Daily ATR
            "min_rvol": 1.20,                  # RVOL >= 1.20x shifted TOD baseline
            "target_rr": 2.50,                 # 2.50R asymmetric target multiple
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_or_atr_ratio": [0.25, 0.30, 0.35],
            "min_rvol": [1.15, 1.20, 1.30],
            "target_rr": [2.00, 2.50, 3.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 22.
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

        # 2. 15m Intraday EMA(20) Trend Baseline
        out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()

        # 3. 20-Session TOD Rolling Volume Baseline (Shifted 1 session)
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
        ema20s = out["ema20"].values

        min_or_atr = float(self.parameters.get("min_or_atr_ratio", 0.30))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.65))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 2.50))

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
            c_ema = ema20s[i]

            # -------------------------------------------------------------
            # 1. Establish 09:15 Opening Range (Bar 1)
            # -------------------------------------------------------------
            if hour == 9 and minute == 15:
                if pd.isna(c_atr) or c_atr <= 0:
                    continue

                bar1_range = c_high - c_low
                bar1_body = abs(c_close - c_open)
                bar1_body_ratio = bar1_body / max(bar1_range, 0.01)
                bar1_atr_ratio = bar1_range / c_atr

                if (min_or_atr <= bar1_atr_ratio <= max_or_atr) and (bar1_body_ratio >= 0.40):
                    or_high = c_high
                    or_low = c_low
                    or_valid = True
                continue

            # -------------------------------------------------------------
            # 2. Breakout Evaluation Window (09:30 to 10:45 IST)
            # -------------------------------------------------------------
            if (not or_valid) or traded_today or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10 and minute <= 45):
                or_mid = (or_high + or_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish Apex Momentum (LONG)
                if (c_close > or_high) and (c_close > c_open) and (c_close > c_ema) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    stop_dist = max(c_close - or_mid, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 22 APEX LONG: Breakout Close={c_close:.1f} > OR_High={or_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish Apex Momentum (SHORT)
                elif (c_close < or_low) and (c_close < c_open) and (c_close < c_ema) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    stop_dist = max(or_mid - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 22 APEX SHORT: Breakdown Close={c_close:.1f} < OR_Low={or_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
