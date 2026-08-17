"""
Ashva Quantitative Strategy: Alpha 04 — Gap & Go (alpha_04_gap_and_go)
Hypothesis:
When a liquid NSE stock opens with a meaningful overnight gap and early trading confirms
that the gap represents genuine new information rather than an immediate liquidity imbalance,
the initial directional move tends to continue intraday.
"""

from typing import Dict, List, Any, Optional
from datetime import time
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class Alpha04GapAndGo(BaseHypothesis):
    """
    Alpha 04: Gap & Go (alpha_04_gap_and_go)
    Identifies institutional overnight gap continuations confirmed by early volume and gap holding.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_04_GAP_AND_GO",
            name="alpha_04_gap_and_go",
            category="INTRADAY_MOMENTUM_GAP_CONTINUATION",
            economic_rationale=(
                "When a liquid NSE stock opens with a meaningful overnight gap and early trading confirms "
                "that the gap represents genuine new information rather than an immediate liquidity imbalance, "
                "the initial directional move tends to continue intraday. High opening volume and price holding "
                "above/below previous day's close confirms institutional consensus."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN",
                "AXISBANK", "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL"
            ],
            timeframe="15m",
            author="AshvaQuantLab",
        )
        params = parameters or {
            "min_gap_pct": 0.50,         # Minimum 0.50% overnight gap size
            "volume_mult": 1.30,         # First 15m volume >= 1.3x 20-period Volume SMA
            "min_adx": 16.0,             # Minimum directional trend momentum
            "rr_ratio": 2.0,             # 2.0R Take Profit
            "sl_buffer_atr": 0.5,        # Buffer beyond first bar extreme
            "entry_time": "09:30",       # Triggered at 09:30 AM after first bar confirmation
            "eod_exit_time": "15:15",
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.40, 0.50, 0.70],
            "volume_mult": [1.10, 1.30, 1.50],
            "min_adx": [14.0, 16.0, 18.0],
            "rr_ratio": [1.5, 2.0, 2.5],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates gap continuation signals when opening gap holds with elevated volume.
        """
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column")

        min_gap = float(self.parameters.get("min_gap_pct", 0.50))
        vol_mult = float(self.parameters.get("volume_mult", 1.30))
        min_adx = float(self.parameters.get("min_adx", 16.0))
        rr = float(self.parameters.get("rr_ratio", 2.0))
        sl_buffer = float(self.parameters.get("sl_buffer_atr", 0.5))

        # Technical Indicators
        out = TI.add_atr(out, period=14)
        out = TI.add_adx(out, period=14)
        out["vol_sma20"] = out["volume"].rolling(window=20, min_periods=5).mean()

        # Intraday Anchored VWAP
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        dates = pd.to_datetime(out.index).date
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = out["cum_pv"] / out["cum_v"].replace(0, np.nan)
        out["vwap"] = out["vwap"].bfill().ffill()
        out.drop(columns=["cum_pv", "cum_v"], inplace=True)

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        vols = out["volume"].values
        vol_smas = out["vol_sma20"].values
        vwaps = out["vwap"].values
        adxs = out["adx_14"].values
        atrs = out["atr_14"].values
        timestamps = out.index
        times = [ts.time() for ts in timestamps]

        t_0915 = time(9, 15)
        t_0930 = time(9, 30)
        t_1515 = time(15, 15)

        curr_state = 0.0
        entry_price = 0.0
        curr_sl = 0.0
        curr_tp = 0.0

        current_day = None
        prev_day_close = np.nan
        first_bar_open = np.nan
        first_bar_high = np.nan
        first_bar_low = np.nan
        first_bar_close = np.nan
        first_bar_vol = np.nan
        first_bar_vol_sma = np.nan
        first_bar_adx = np.nan
        trade_taken_today = False

        for i in range(1, n):
            d = timestamps[i].date()
            t = times[i]
            c_price = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = vols[i]
            c_vol_sma = vol_smas[i]
            c_vwap = vwaps[i]
            c_adx = adxs[i]
            c_atr = atrs[i]

            # New Trading Day Initialization
            if current_day != d:
                prev_day_close = closes[i - 1]  # Previous session closing price
                current_day = d
                first_bar_open = np.nan
                first_bar_high = np.nan
                first_bar_low = np.nan
                first_bar_close = np.nan
                first_bar_vol = np.nan
                first_bar_vol_sma = np.nan
                first_bar_adx = np.nan
                trade_taken_today = False

            # Capture First 15m Bar (09:15 - 09:30 AM)
            if t == t_0915:
                first_bar_open = c_open
                first_bar_high = c_high
                first_bar_low = c_low
                first_bar_close = c_price
                first_bar_vol = c_vol
                first_bar_vol_sma = c_vol_sma
                first_bar_adx = c_adx

            # Intraday EOD Exit
            if t >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "alpha_04_gap_and_go EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Gap & Go Trigger at 09:30 AM (Bar 2 Open / Breakout)
            if (curr_state == 0.0) and not trade_taken_today and not np.isnan(prev_day_close) and not np.isnan(first_bar_close) and (t == t_0930):
                gap_pct = ((first_bar_open - prev_day_close) / prev_day_close) * 100.0
                vol_ok = (first_bar_vol >= vol_mult * first_bar_vol_sma) if not np.isnan(first_bar_vol_sma) and first_bar_vol_sma > 0 else True
                adx_ok = (c_adx >= min_adx) if not np.isnan(c_adx) else True

                # 1. GAP UP & GO (BULLISH GAP CONTINUATION)
                # - Meaningful Gap Up: Open >= Prev Close + 0.50%
                # - Gap Holding: First bar Low did NOT fill the gap (Low >= Prev Close)
                # - Bullish Acceptance: First bar Close > Open and Close > VWAP
                # - Institutional Volume: Volume >= 1.3x SMA20
                if (gap_pct >= min_gap) and (first_bar_low >= prev_day_close) and (first_bar_close > first_bar_open) and (c_price > c_vwap) and vol_ok and adx_ok:
                    curr_state = 1.0
                    entry_price = c_open
                    curr_sl = min(first_bar_low - (sl_buffer * c_atr), prev_day_close)
                    risk = entry_price - curr_sl
                    if risk > 0:
                        curr_tp = entry_price + (rr * risk)
                        trade_taken_today = True

                        signals[i] = 1.0
                        stop_loss[i] = curr_sl
                        take_profit[i] = curr_tp
                        rationales[i] = (
                            f"alpha_04_gap_and_go LONG (GAP UP & GO): Gap={gap_pct:+.2f}% | "
                            f"Vol={first_bar_vol:.0f} (>={vol_mult}x SMA) | ADX={c_adx:.1f} | SL={curr_sl:.1f} | TP={curr_tp:.1f}"
                        )

                # 2. GAP DOWN & GO (BEARISH GAP CONTINUATION)
                # - Meaningful Gap Down: Open <= Prev Close - 0.50%
                # - Gap Holding: First bar High did NOT fill the gap (High <= Prev Close)
                # - Bearish Acceptance: First bar Close < Open and Close < VWAP
                # - Institutional Volume: Volume >= 1.3x SMA20
                elif (gap_pct <= -min_gap) and (first_bar_high <= prev_day_close) and (first_bar_close < first_bar_open) and (c_price < c_vwap) and vol_ok and adx_ok:
                    curr_state = -1.0
                    entry_price = c_open
                    curr_sl = max(first_bar_high + (sl_buffer * c_atr), prev_day_close)
                    risk = curr_sl - entry_price
                    if risk > 0:
                        curr_tp = entry_price - (rr * risk)
                        trade_taken_today = True

                        signals[i] = -1.0
                        stop_loss[i] = curr_sl
                        take_profit[i] = curr_tp
                        rationales[i] = (
                            f"alpha_04_gap_and_go SHORT (GAP DOWN & GO): Gap={gap_pct:+.2f}% | "
                            f"Vol={first_bar_vol:.0f} (>={vol_mult}x SMA) | ADX={c_adx:.1f} | SL={curr_sl:.1f} | TP={curr_tp:.1f}"
                        )

            # In Position: Monitor TP / SL
            elif curr_state == 1.0:
                if c_high >= curr_tp or c_low <= curr_sl:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = f"alpha_04_gap_and_go EXIT LONG: {'Target Hit (+2R)' if c_high >= curr_tp else 'Stop Loss Hit'}"
                else:
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
            elif curr_state == -1.0:
                if c_low <= curr_tp or c_high >= curr_sl:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = f"alpha_04_gap_and_go EXIT SHORT: {'Target Hit (+2R)' if c_low <= curr_tp else 'Stop Loss Hit'}"
                else:
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
