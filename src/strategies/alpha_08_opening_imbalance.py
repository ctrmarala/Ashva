"""
Ashva Quantitative Strategy: Opening Price Acceptance & Directional Imbalance (Alpha 08)
Captures continuation momentum following a strong directional first 15-minute candle
with minimal adverse excursion from the opening price (Open ≈ Low / Open ≈ High).

Hypothesis:
When a liquid NSE stock exhibits a strong directional first 15-minute candle with minimal
adverse excursion from the opening price, expanded range and elevated volume, the directional
imbalance tends to persist into the subsequent session.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class Alpha08OpeningImbalance(BaseHypothesis):
    """
    Opening Price Acceptance & Directional Imbalance (Alpha 08):
    1. First 15m Bar (09:15-09:30): Evaluates Open, High, Low, Close, and Volume.
    2. Minimal Adverse Excursion (Near-Zero Wick):
       - Bullish (Open ≈ Low): Lower Wick <= 10% of candle range.
       - Bearish (Open ≈ High): Upper Wick <= 10% of candle range.
    3. Directional Dominance: Candle Body >= 70% of candle range.
    4. Range Expansion: Candle Range >= 0.40 * Daily ATR(14) (shifted prior daily ATR).
    5. Volume Confirmation: 09:15 Volume >= 1.20x shifted 20-session TOD baseline.
    6. Execution & Risk: Entry at 09:30 next-bar open, Stop at Bar 1 extreme, Target 1.5R, EOD 15:15 exit.
    7. Frequency: Maximum 1 trade per stock per day.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_08_OPENING_IMBALANCE",
            name="Alpha_08_Opening_Imbalance",
            category="OPENING_DIRECTIONAL_IMBALANCE",
            economic_rationale=(
                "When a liquid NSE stock exhibits a strong directional first 15-minute candle "
                "with minimal adverse excursion from the opening price, expanded range and elevated volume, "
                "the directional imbalance tends to persist into the subsequent session."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            author="AshvaQuantLab",
        )
        params = parameters or {
            "max_adverse_wick_ratio": 0.10,    # Adverse wick <= 10% of total candle range
            "min_body_ratio": 0.70,            # Directional body >= 70% of total candle range
            "min_range_atr_ratio": 0.40,       # Candle range >= 0.40 * Daily ATR(14)
            "min_rvol": 1.20,                  # Volume >= 1.20x shifted TOD baseline
            "target_rr": 1.50,                 # Exactly 1:1.50 Risk-to-Reward ratio
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "max_adverse_wick_ratio": [0.08, 0.10, 0.12],
            "min_body_ratio": [0.65, 0.70, 0.75],
            "min_range_atr_ratio": [0.35, 0.40, 0.45],
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 08.
        Generates discrete entry pulses (+1.0 / -1.0) on 09:15 bar, filled at 09:30 open.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Compute Daily ATR(14) strictly from completed prior sessions (Shifted 1 session)
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
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)  # Shifted 1 session (Zero Look-Ahead)

        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()

        # 2. Compute 20-Session Time-of-Day (TOD) Rolling Volume Baseline (Shifted 1 session to prevent look-ahead)
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

        max_wick = float(self.parameters.get("max_adverse_wick_ratio", 0.10))
        min_body = float(self.parameters.get("min_body_ratio", 0.70))
        min_range_atr = float(self.parameters.get("min_range_atr_ratio", 0.40))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))

        for i in range(n):
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            # Alpha 08 evaluates ONLY the first 15m candle of the session (09:15 IST)
            if hour == 9 and minute == 15:
                c_open = opens[i]
                c_high = highs[i]
                c_low = lows[i]
                c_close = closes[i]
                c_vol = volumes[i]
                c_tod_vol = tod_vols[i]
                c_daily_atr = daily_atrs[i]

                candle_range = c_high - c_low
                if candle_range <= 0.01 or pd.isna(c_daily_atr) or c_daily_atr <= 0:
                    continue

                # Range expansion filter against prior Daily ATR
                if candle_range < (min_range_atr * c_daily_atr):
                    continue

                # Volume expansion filter against shifted TOD baseline
                rvol = c_vol / max(1.0, c_tod_vol)
                if rvol < min_rvol:
                    continue

                body_size = abs(c_close - c_open)
                body_ratio = body_size / candle_range

                if body_ratio < min_body:
                    continue

                # -------------------------------------------------------------
                # Bullish Imbalance: Open ≈ Low (Minimal Lower Wick)
                # -------------------------------------------------------------
                if c_close > c_open:
                    lower_wick = c_open - c_low
                    lower_wick_ratio = lower_wick / candle_range

                    if lower_wick_ratio <= max_wick:
                        # Discrete Entry Pulse on 09:15 bar (filled at 09:30 next_open)
                        signals[i] = 1.0
                        stop_dist = max(c_close - c_low, 0.05)
                        stop_loss[i] = c_low  # Stop at exact Bar-1 extreme
                        take_profit[i] = c_close + (target_rr * stop_dist)
                        rationales[i] = (
                            f"Alpha 08 LONG: Open~Low (LWick={lower_wick_ratio*100:.1f}%) | "
                            f"Body={body_ratio*100:.1f}% | Range={candle_range:.1f} ({candle_range/c_daily_atr*100:.1f}% ATR) | "
                            f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                        )

                # -------------------------------------------------------------
                # Bearish Imbalance: Open ≈ High (Minimal Upper Wick)
                # -------------------------------------------------------------
                elif c_close < c_open:
                    upper_wick = c_high - c_open
                    upper_wick_ratio = upper_wick / candle_range

                    if upper_wick_ratio <= max_wick:
                        # Discrete Entry Pulse on 09:15 bar (filled at 09:30 next_open)
                        signals[i] = -1.0
                        stop_dist = max(c_high - c_close, 0.05)
                        stop_loss[i] = c_high  # Stop at exact Bar-1 extreme
                        take_profit[i] = c_close - (target_rr * stop_dist)
                        rationales[i] = (
                            f"Alpha 08 SHORT: Open~High (UWick={upper_wick_ratio*100:.1f}%) | "
                            f"Body={body_ratio*100:.1f}% | Range={candle_range:.1f} ({candle_range/c_daily_atr*100:.1f}% ATR) | "
                            f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                        )

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
