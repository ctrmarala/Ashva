"""
Ashva Quantitative Strategy: Previous-Day High/Low Sweep & Failed Auction Reversal (Alpha 06)
Captures intraday mean-reversion following a failed breakout at the prior day's key liquidity boundaries (PDH / PDL).

Hypothesis:
When a liquid NSE stock briefly breaks the previous day's high or low, fails to sustain the breakout,
and closes back inside the prior day's range with strong rejection, the failed breakout represents trapped
breakout participants and tends to mean-revert intraday.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class Alpha06PDHPDLSweep(BaseHypothesis):
    """
    Previous-Day High/Low Sweep & Failed Auction Reversal (Alpha 06):
    1. Reference Levels: Previous completed session PDH, PDL, PDC calculated with zero look-ahead.
    2. Boundary Sweep: Price pokes beyond PDH (or PDL) by <= 0.50 * ATR between 09:15 and 13:00.
    3. Rejection Confirmation: Bar closes back inside the prior day's range with >=30% rejection wick
       or directional candle body.
    4. Independent Level Safeguard: Maximum 1 trade per boundary (PDH/PDL) per stock per day.
    5. Risk/Reward: Stop beyond sweep extreme (+0.10 ATR), exact 1:1.50 target, EOD 15:15 square-off.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_06_PDH_PDL_SWEEP",
            name="Alpha_06_PDH_PDL_Sweep",
            category="AUCTION_LIQUIDITY_REVERSAL",
            economic_rationale=(
                "When a liquid NSE stock briefly breaks the previous day's high or low, fails to sustain "
                "the breakout, and closes back inside the prior day's range with strong rejection, the failed "
                "breakout represents trapped breakout participants and tends to mean-revert intraday."
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
            "max_sweep_atr_ratio": 0.50,    # Max penetration beyond PDH/PDL (must not exceed 0.50 ATR)
            "min_wick_ratio": 0.30,         # Rejection wick must be >= 30% of total candle range
            "stop_atr_buffer": 0.10,        # Stop buffer beyond sweep extreme (0.10 ATR)
            "target_rr": 1.50,              # Exactly 1:1.50 Risk-to-Reward ratio
            "max_sweep_hour": 13,           # Sweep detection allowed until 13:00 IST
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "max_sweep_atr_ratio": [0.40, 0.50, 0.60],
            "min_wick_ratio": [0.25, 0.30, 0.35],
            "stop_atr_buffer": [0.05, 0.10, 0.15],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 06.
        Generates discrete entry pulses (+1.0 / -1.0) on trigger bars, 0.0 otherwise.
        """
        out = df.copy()

        # 1. Compute Rolling ATR
        out = TI.add_atr(out, period=14)
        atr_col = "atr_14"

        # 2. Compute Prior Day High (PDH), Low (PDL), and Close (PDC)
        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time

        # Build daily aggregations strictly using completed prior sessions
        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )
        daily_shifted = daily_summary.shift(1)  # Shift by 1 session to prevent look-ahead bias

        # Map back to 15m intraday bars
        out["pdh"] = pd.Series(dates, index=out.index).map(daily_shifted["day_high"])
        out["pdl"] = pd.Series(dates, index=out.index).map(daily_shifted["day_low"])
        out["pdc"] = pd.Series(dates, index=out.index).map(daily_shifted["day_close"])

        # Forward fill any initial NaNs
        out["pdh"] = out["pdh"].ffill()
        out["pdl"] = out["pdl"].ffill()
        out["pdc"] = out["pdc"].ffill()

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        atrs = out[atr_col].values
        pdhs = out["pdh"].values
        pdls = out["pdl"].values

        max_sweep_atr = float(self.parameters.get("max_sweep_atr_ratio", 0.50))
        min_wick_r = float(self.parameters.get("min_wick_ratio", 0.30))
        sl_buffer = float(self.parameters.get("stop_atr_buffer", 0.10))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_hour = int(self.parameters.get("max_sweep_hour", 13))

        current_day = None
        pdh_traded_today = False
        pdl_traded_today = False

        for i in range(1, n):
            ts = timestamps[i]
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            # Reset tracking on new daily session
            if bar_date != current_day:
                current_day = bar_date
                pdh_traded_today = False
                pdl_traded_today = False

            c_pdh = pdhs[i]
            c_pdl = pdls[i]
            c_atr = atrs[i]

            # Require valid prior day levels and ATR
            if pd.isna(c_pdh) or pd.isna(c_pdl) or pd.isna(c_atr) or c_atr <= 0:
                continue

            # Sweep detection window: 09:15 to 13:00 IST
            if (hour > max_hour) or (hour == max_hour and minute > 0):
                continue

            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]
            c_range = max(c_high - c_low, 0.01)

            # -------------------------------------------------------------
            # Case 1: Bearish Sweep at PDH (Failed Bullish Breakout -> SHORT)
            # -------------------------------------------------------------
            if not pdh_traded_today and (c_high > c_pdh):
                # 1. Sweep Depth Filter: High must not exceed PDH + (0.50 * ATR)
                sweep_depth = c_high - c_pdh
                if sweep_depth <= (max_sweep_atr * c_atr):
                    # 2. Inside Close Filter: Bar closes strictly back inside prior range
                    if c_close < c_pdh:
                        # 3. Rejection Wick / Body Signature
                        upper_wick = c_high - max(c_open, c_close)
                        upper_wick_pct = upper_wick / c_range
                        is_bearish_body = (c_close < c_open)

                        if (upper_wick_pct >= min_wick_r) or is_bearish_body:
                            # Discrete Entry Pulse
                            signals[i] = -1.0
                            sl_price = c_high + (sl_buffer * c_atr)
                            stop_dist = max(sl_price - c_close, 0.30 * c_atr)
                            stop_loss[i] = c_close + stop_dist
                            take_profit[i] = c_close - (target_rr * stop_dist)
                            rationales[i] = (
                                f"Alpha 06 SHORT: PDH Sweep={c_high:.1f} (PDH={c_pdh:.1f}) | "
                                f"Wick={upper_wick_pct*100:.1f}% | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                            )
                            pdh_traded_today = True
                            continue

            # -------------------------------------------------------------
            # Case 2: Bullish Sweep at PDL (Failed Bearish Breakdown -> LONG)
            # -------------------------------------------------------------
            if not pdl_traded_today and (c_low < c_pdl):
                # 1. Sweep Depth Filter: Low must not exceed PDL - (0.50 * ATR)
                sweep_depth = c_pdl - c_low
                if sweep_depth <= (max_sweep_atr * c_atr):
                    # 2. Inside Close Filter: Bar closes strictly back inside prior range
                    if c_close > c_pdl:
                        # 3. Rejection Wick / Body Signature
                        lower_wick = min(c_open, c_close) - c_low
                        lower_wick_pct = lower_wick / c_range
                        is_bullish_body = (c_close > c_open)

                        if (lower_wick_pct >= min_wick_r) or is_bullish_body:
                            # Discrete Entry Pulse
                            signals[i] = 1.0
                            sl_price = c_low - (sl_buffer * c_atr)
                            stop_dist = max(c_close - sl_price, 0.30 * c_atr)
                            stop_loss[i] = c_close - stop_dist
                            take_profit[i] = c_close + (target_rr * stop_dist)
                            rationales[i] = (
                                f"Alpha 06 LONG: PDL Sweep={c_low:.1f} (PDL={c_pdl:.1f}) | "
                                f"Wick={lower_wick_pct*100:.1f}% | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                            )
                            pdl_traded_today = True
                            continue

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
