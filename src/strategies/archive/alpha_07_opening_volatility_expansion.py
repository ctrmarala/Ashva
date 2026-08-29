"""
Ashva Quantitative Strategy: Opening Volatility Expansion (Alpha 07)
Captures directional momentum expansion following an unusually compressed 30-minute opening structure.

Hypothesis:
When the first 30 minutes establish unusually compressed volatility (OR30_Range <= 0.50 * Daily ATR(14)),
the subsequent volume-confirmed expansion of that range tends to produce strong continuation
in the direction of the initial breakout with a highly favorable risk-to-reward ratio.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class Alpha07OpeningVolatilityExpansion(BaseHypothesis):
    """
    Opening Volatility Expansion (Alpha 07):
    1. 30m Range (OR30): High/Low from 09:15 to 09:45 IST.
    2. Compression Gate: OR30_Range <= 0.50 * Daily ATR(14) (shifted prior daily ATR).
    3. Breakout Window: 09:45 to 11:30 IST.
    4. Trigger: 15m close outside OR30 + same-direction candle body + RVOL >= 1.20x TOD baseline.
    5. Risk/Reward: Stop at opposite OR30 boundary, exact 2R target (1:2.0), EOD 15:15 exit.
    6. Frequency: Maximum 1 trade per stock per day.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_07_OPENING_VOLATILITY_EXPANSION",
            name="Alpha_07_Opening_Volatility_Expansion",
            category="VOLATILITY_COMPRESSION_EXPANSION",
            economic_rationale=(
                "When the first 30 minutes establish unusually compressed volatility "
                "(OR30_Range <= 0.50 * Daily ATR(14)), the subsequent volume-confirmed expansion "
                "of that range tends to produce strong continuation in the direction of the initial "
                "breakout with a highly favorable risk-to-reward ratio."
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
            "max_compression_atr_ratio": 0.50, # OR30_Range must be <= 0.50 * Daily ATR(14)
            "min_rvol": 1.20,                  # Breakout volume >= 1.20x TOD baseline
            "target_rr": 2.00,                 # Exactly 1:2.00 Risk-to-Reward ratio
            "max_breakout_hour": 11,           # Breakout window ends at 11:30 IST
            "max_breakout_minute": 30,
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "max_compression_atr_ratio": [0.40, 0.50, 0.60],
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00, 2.50],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 07.
        Generates discrete entry pulses (+1.0 / -1.0) on breakout trigger bars, 0.0 otherwise.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Compute Daily ATR(14) strictly from completed prior sessions
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

        # 2. Compute 20-Session Time-of-Day (TOD) Rolling Volume Baseline
        out["tod_mean_vol"] = out.groupby("time_str")["volume"].transform(
            lambda s: s.rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])

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

        max_comp_ratio = float(self.parameters.get("max_compression_atr_ratio", 0.50))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 2.00))
        max_hour = int(self.parameters.get("max_breakout_hour", 11))
        max_min = int(self.parameters.get("max_breakout_minute", 30))

        current_day = None
        or30_high = 0.0
        or30_low = 0.0
        is_compressed = False
        traded_today = False
        bar_count_today = 0
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(n):
            ts = timestamps[i]
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            # Reset on new session
            if bar_date != current_day:
                current_day = bar_date
                or30_high = 0.0
                or30_low = 0.0
                is_compressed = False
                traded_today = False
                bar_count_today = 0
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""

            # Intraday 15:15 EOD Square-Off
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 07 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across intraday bars
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            bar_count_today += 1
            c_high = highs[i]
            c_low = lows[i]
            c_open = opens[i]
            c_close = closes[i]
            c_vol = volumes[i]
            c_tod_vol = tod_vols[i]
            c_daily_atr = daily_atrs[i]

            # -------------------------------------------------------------
            # Phase 1: 30-Minute Range Construction (09:15 & 09:30 bars)
            # -------------------------------------------------------------
            if hour == 9 and minute == 15:
                or30_high = c_high
                or30_low = c_low
                continue
            elif hour == 9 and minute == 30:
                or30_high = max(or30_high, c_high)
                or30_low = min(or30_low, c_low)
                or30_range = or30_high - or30_low

                # Evaluate Volatility Compression against prior Daily ATR(14)
                if pd.notna(c_daily_atr) and c_daily_atr > 0:
                    if or30_range <= (max_comp_ratio * c_daily_atr):
                        is_compressed = True
                    else:
                        is_compressed = False
                continue

            # Skip if not compressed or already traded today
            if not is_compressed or traded_today:
                continue

            # -------------------------------------------------------------
            # Phase 2: Volatility Expansion Trigger (09:45 to 11:30 IST)
            # -------------------------------------------------------------
            if (hour > max_hour) or (hour == max_hour and minute > max_min):
                continue

            rvol = c_vol / max(1.0, c_tod_vol)
            or30_range = or30_high - or30_low

            # Bullish Breakout Expansion (LONG)
            if (c_close > or30_high) and (c_close > c_open) and (rvol >= min_rvol):
                curr_state = 1.0
                stop_dist = max(c_close - or30_low, 0.15 * c_daily_atr)
                curr_sl = c_close - stop_dist
                curr_tp = c_close + (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 07 LONG: OR30 Breakout Close={c_close:.1f} > High={or30_high:.1f} | "
                    f"OR30 Range={or30_range:.1f} ({or30_range/c_daily_atr*100:.1f}% Daily ATR) | "
                    f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                )
                signals[i] = 1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

            # Bearish Breakdown Expansion (SHORT)
            elif (c_close < or30_low) and (c_close < c_open) and (rvol >= min_rvol):
                curr_state = -1.0
                stop_dist = max(or30_high - c_close, 0.15 * c_daily_atr)
                curr_sl = c_close + stop_dist
                curr_tp = c_close - (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 07 SHORT: OR30 Breakdown Close={c_close:.1f} < Low={or30_low:.1f} | "
                    f"OR30 Range={or30_range:.1f} ({or30_range/c_daily_atr*100:.1f}% Daily ATR) | "
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
