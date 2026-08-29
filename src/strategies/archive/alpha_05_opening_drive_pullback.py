"""
Ashva Quantitative Strategy: Opening Drive Pullback Continuation (Alpha 05)
Captures continuation momentum following a high-conviction opening directional impulse
and a shallow, low-volume value retracement.

Hypothesis:
A strong opening directional move followed by a shallow, low-volume pullback is more
likely to continue than reverse, because the initial directional imbalance remains intact
while the counter-move is dominated by profit-taking rather than new institutional information.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class Alpha05OpeningDrivePullback(BaseHypothesis):
    """
    Opening Drive Pullback Continuation (Alpha 05):
    1. Drive Identification: First qualifying 15m bar (09:15 or 09:30) with >=60% body,
       >=0.60 ATR range, and RVOL >= 1.30x TOD baseline.
    2. Pullback Zone: 38.2% to 61.8% Fibonacci retracement of drive range, holding VWAP & Drive Origin.
    3. Volume Exhaustion: Pullback bar volume <= 70% of drive volume.
    4. Resumption Trigger: Next 15m bar closes back in drive direction beyond the pullback high/low.
    5. Risk/Reward: Stop beyond pullback swing extreme, 1:2.0 target, EOD 15:15 square-off.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_05_OPENING_DRIVE_PULLBACK",
            name="Alpha_05_Opening_Drive_Pullback",
            category="MICROSTRUCTURE_MOMENTUM_PULLBACK",
            economic_rationale=(
                "A strong opening directional move followed by a shallow, low-volume pullback "
                "is more likely to continue than reverse, because the initial directional imbalance "
                "remains intact while the counter-move is dominated by profit-taking rather than new information."
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
            "min_body_ratio": 0.60,         # Body must be >= 60% of total candle range
            "min_atr_ratio": 0.60,          # Range must be >= 0.60 * ATR(14)
            "min_rvol": 1.30,               # Opening bar volume >= 1.30x 20-session baseline
            "fib_low": 0.382,               # Shallow pullback zone start
            "fib_high": 0.618,              # Shallow pullback zone end
            "pullback_vol_max_ratio": 0.70, # Pullback candle volume <= 70% of drive volume
            "target_rr": 2.0,               # 1:2.0 Risk-Reward target
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_body_ratio": [0.55, 0.60, 0.65],
            "min_rvol": [1.20, 1.30, 1.40],
            "pullback_vol_max_ratio": [0.65, 0.70, 0.80],
            "target_rr": [1.5, 2.0, 2.5],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 05.
        Generates discrete entry pulses (+1.0 / -1.0) on trigger bars, 0.0 otherwise.
        """
        out = df.copy()

        # 1. Compute Indicators
        out = TI.add_atr(out, period=14)
        atr_col = "atr_14"

        # Intraday VWAP
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        dates = pd.to_datetime(out.index).date
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = out["cum_pv"] / out["cum_v"].replace(0, np.nan)
        out["vwap"] = out["vwap"].bfill().ffill()

        # 20-Session Time-of-Day (TOD) Rolling Volume Baseline
        timestamps = pd.to_datetime(out.index)
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]
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
        vwaps = out["vwap"].values
        atrs = out[atr_col].values
        tod_vols = out["tod_mean_vol"].values

        min_body = float(self.parameters.get("min_body_ratio", 0.60))
        min_atr_r = float(self.parameters.get("min_atr_ratio", 0.60))
        min_rvol = float(self.parameters.get("min_rvol", 1.30))
        fib_low = float(self.parameters.get("fib_low", 0.382))
        fib_high = float(self.parameters.get("fib_high", 0.618))
        vol_max_r = float(self.parameters.get("pullback_vol_max_ratio", 0.70))
        target_rr = float(self.parameters.get("target_rr", 2.0))

        # State tracking per session
        current_day = None
        drive_side = None
        drive_high = 0.0
        drive_low = 0.0
        drive_vol = 0.0
        pullback_extreme = 0.0
        in_pullback_zone = False
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(n):
            ts = df.index[i]
            bar_date = ts.date()
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            # Reset state on new session
            if bar_date != current_day:
                current_day = bar_date
                drive_side = None
                drive_high = 0.0
                drive_low = 0.0
                drive_vol = 0.0
                pullback_extreme = 0.0
                in_pullback_zone = False
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
                    rationales[i] = "Alpha 05 EXIT: Intraday 15:15 EOD Square-Off"
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
            c_vwap = vwaps[i]
            c_atr = atrs[i]
            c_tod = tod_vols[i]

            # 1. Phase 1: Drive Bar Identification (Bar 1 @ 09:15 or Bar 2 @ 09:30)
            if drive_side is None and not traded_today:
                if (hour == 9 and minute in [15, 30]):
                    bar_range = c_high - c_low
                    body = abs(c_close - c_open)
                    rvol = c_vol / max(1.0, c_tod)

                    if bar_range >= (min_atr_r * c_atr) and rvol >= min_rvol and bar_range > 0:
                        body_ratio = body / bar_range
                        if body_ratio >= min_body:
                            if c_close > c_open:
                                drive_side = "BULLISH"
                                drive_high = c_high
                                drive_low = c_low
                                drive_vol = c_vol
                                pullback_extreme = c_low
                            elif c_close < c_open:
                                drive_side = "BEARISH"
                                drive_high = c_high
                                drive_low = c_low
                                drive_vol = c_vol
                                pullback_extreme = c_high
                    continue

            # If no drive was established in early morning, skip the rest of the day
            if drive_side is None or traded_today:
                continue

            # Hard cutoff: Entries only between 09:45 and 12:00
            if (hour < 9 or (hour == 9 and minute < 45)) or (hour > 12):
                continue

            drive_range = drive_high - drive_low
            if drive_range <= 0.01:
                continue

            # 2. Phase 2: Pullback Evaluation
            if drive_side == "BULLISH":
                ret38 = drive_high - (fib_low * drive_range)
                ret62 = drive_high - (fib_high * drive_range)

                # Check if pullback violated drive origin or broken below VWAP
                if c_low < drive_low or c_close < c_vwap:
                    drive_side = None  # Drive invalidated
                    continue

                # Check if current bar is in shallow pullback zone with volume dry-up
                is_low_vol = (c_vol <= vol_max_r * drive_vol)
                if (c_low <= ret38) and is_low_vol:
                    in_pullback_zone = True
                    pullback_extreme = min(pullback_extreme if pullback_extreme > 0 else c_low, c_low)

                # 3. Phase 3: Continuation Resumption Trigger
                if in_pullback_zone and (c_close > opens[i]) and (c_close > highs[i - 1]):
                    curr_state = 1.0
                    stop_dist = max(c_close - pullback_extreme, 0.40 * c_atr)
                    curr_sl = c_close - stop_dist
                    curr_tp = c_close + (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 05 LONG: DriveRange=Rs {drive_range:.1f} | Retrac={(drive_high - c_low)/drive_range*100:.1f}% | "
                        f"SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                    )
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True
                    drive_side = None

            elif drive_side == "BEARISH":
                ret38 = drive_low + (fib_low * drive_range)
                ret62 = drive_low + (fib_high * drive_range)

                # Check if pullback violated drive origin or broken above VWAP
                if c_high > drive_high or c_close > c_vwap:
                    drive_side = None  # Drive invalidated
                    continue

                # Check if current bar is in shallow pullback zone with volume dry-up
                is_low_vol = (c_vol <= vol_max_r * drive_vol)
                if (c_high >= ret38) and is_low_vol:
                    in_pullback_zone = True
                    pullback_extreme = max(pullback_extreme, c_high)

                # 3. Phase 3: Continuation Resumption Trigger
                if in_pullback_zone and (c_close < opens[i]) and (c_close < lows[i - 1]):
                    curr_state = -1.0
                    stop_dist = max(pullback_extreme - c_close, 0.40 * c_atr)
                    curr_sl = c_close + stop_dist
                    curr_tp = c_close - (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 05 SHORT: DriveRange=Rs {drive_range:.1f} | Retrac={(c_high - drive_low)/drive_range*100:.1f}% | "
                        f"SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                    )
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True
                    drive_side = None

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
