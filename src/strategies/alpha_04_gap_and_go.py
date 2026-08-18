"""
Ashva Quantitative Strategy: Alpha 04 — Gap & Go (alpha_04_gap_and_go)
Institutional Overnight Gap Continuation Engine.
Incorporate:
1. Upper & lower gap bounds (0.50% to 2.50%).
2. Time-of-Day (TOD) 09:15 Relative Volume (RVOL) benchmark.
3. Explicit previous-session final-bar close extraction.
4. Execution price alignment at 09:30 open.
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
                "When a liquid NSE stock opens with a meaningful overnight gap (0.50% to 2.50%) "
                "and early trading confirms that the gap represents genuine new information rather than "
                "an immediate liquidity imbalance, the initial directional move tends to continue intraday."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN",
                "AXISBANK", "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL"
            ],
            timeframe="15m",
            author="AshvaQuantLab",
        )
        params = parameters or {
            "min_gap_pct": 0.50,         # Minimum 0.50% overnight gap
            "max_gap_pct": 2.50,         # Maximum 2.50% gap cap (avoid freak runaway gap distortions)
            "rvol_mult": 1.30,           # Opening 09:15 volume >= 1.30x 20-session TOD average
            "min_adx": 16.0,             # Minimum directional trend momentum
            "rr_ratio": 2.0,             # 2.0R Take Profit
            "sl_buffer_atr": 0.5,        # Buffer beyond first bar extreme
            "min_atr_pct": 0.80,         # Minimum 0.80% normalized ATR (avoid dead/sluggish stocks)
            "max_atr_pct": 3.50,         # Maximum 3.50% normalized ATR (avoid extreme erratic whip)
            "eod_exit_time": "15:15",
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.40, 0.50, 0.60],
            "max_gap_pct": [2.0, 2.5, 3.0],
            "rvol_mult": [1.10, 1.30, 1.50],
            "min_adx": [14.0, 16.0, 18.0],
            "rr_ratio": [1.5, 2.0, 2.5],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates gap continuation signals with TOD relative volume and explicit prior-session close.
        """
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column")

        min_gap = float(self.parameters.get("min_gap_pct", 0.50))
        max_gap = float(self.parameters.get("max_gap_pct", 2.50))
        rvol_m = float(self.parameters.get("rvol_mult", 1.30))
        min_adx = float(self.parameters.get("min_adx", 16.0))
        rr = float(self.parameters.get("rr_ratio", 2.0))
        sl_buffer = float(self.parameters.get("sl_buffer_atr", 0.5))
        min_atr_pct = float(self.parameters.get("min_atr_pct", 0.80))
        max_atr_pct = float(self.parameters.get("max_atr_pct", 3.50))

        # 1. Technical Indicators
        out = TI.add_atr(out, period=14)
        out = TI.add_adx(out, period=14)

        # 2. Intraday Anchored VWAP
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        dates_series = pd.to_datetime(out.index).date
        out["cum_pv"] = pv.groupby(dates_series).cumsum()
        out["cum_v"] = out["volume"].groupby(dates_series).cumsum()
        out["vwap"] = out["cum_pv"] / out["cum_v"].replace(0, np.nan)
        out["vwap"] = out["vwap"].bfill().ffill()
        out.drop(columns=["cum_pv", "cum_v"], inplace=True)

        # 3. Explicit Prior Session Last Close Mapping & 09:15 TOD Relative Volume
        unique_dates = sorted(list(set(dates_series)))
        prior_close_map = {}
        tod_0915_vol_map = {}

        # Build daily 09:15 volumes list for TOD benchmark
        daily_0915_vols = []
        for d in unique_dates:
            day_mask = (dates_series == d)
            day_df = out[day_mask]
            if not day_df.empty:
                # Store 09:15 bar volume
                t0915_rows = day_df[day_df.index.time == time(9, 15)]
                v0915 = t0915_rows["volume"].iloc[0] if not t0915_rows.empty else np.nan
                daily_0915_vols.append(v0915)
                # Compute rolling 20-session TOD volume average
                past_vols = [v for v in daily_0915_vols[-21:-1] if not np.isnan(v)]
                tod_0915_vol_map[d] = np.mean(past_vols) if past_vols else v0915

        # Build explicit prior-day final close map
        for idx in range(1, len(unique_dates)):
            prev_d = unique_dates[idx - 1]
            curr_d = unique_dates[idx]
            prev_day_df = out[dates_series == prev_d]
            if not prev_day_df.empty:
                prior_close_map[curr_d] = prev_day_df["close"].iloc[-1]

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
        first_bar_open = np.nan
        first_bar_high = np.nan
        first_bar_low = np.nan
        first_bar_close = np.nan
        first_bar_vol = np.nan
        trade_taken_today = False

        for i in range(1, n):
            d = timestamps[i].date()
            t = times[i]
            c_price = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = vols[i]
            c_vwap = vwaps[i]
            c_adx = adxs[i]
            c_atr = atrs[i]

            # Day boundary reset
            if current_day != d:
                current_day = d
                first_bar_open = np.nan
                first_bar_high = np.nan
                first_bar_low = np.nan
                first_bar_close = np.nan
                first_bar_vol = np.nan
                trade_taken_today = False

            # Capture First 15m Bar (09:15 - 09:30 AM)
            if t == t_0915:
                first_bar_open = c_open
                first_bar_high = c_high
                first_bar_low = c_low
                first_bar_close = c_price
                first_bar_vol = c_vol

            # Intraday EOD Exit
            if t >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "alpha_04_gap_and_go EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Gap & Go Trigger at 09:30 AM (Bar 2 Open)
            if (curr_state == 0.0) and not trade_taken_today and (d in prior_close_map) and not np.isnan(first_bar_close) and (t == t_0930):
                prev_day_close = prior_close_map[d]
                tod_benchmark_vol = tod_0915_vol_map.get(d, first_bar_vol)

                gap_pct = ((first_bar_open - prev_day_close) / prev_day_close) * 100.0
                abs_gap = abs(gap_pct)
                rvol = (first_bar_vol / tod_benchmark_vol) if tod_benchmark_vol and tod_benchmark_vol > 0 else 1.0
                norm_atr = (c_atr / prev_day_close) * 100.0 if prev_day_close > 0 else 0.0

                gap_valid = (min_gap <= abs_gap <= max_gap) and (min_atr_pct <= norm_atr <= max_atr_pct)
                rvol_ok = (rvol >= rvol_m)
                adx_ok = (c_adx >= min_adx) if not np.isnan(c_adx) else True

                # 1. GAP UP & GO (BULLISH CONTINUATION)
                # - Gap between 0.50% and 2.50%
                # - First bar Low >= Prev Close (Gap holds)
                # - First bar Close > Open and Close > VWAP
                # - TOD 09:15 RVOL >= 1.30x
                if gap_valid and (gap_pct > 0) and (first_bar_low >= prev_day_close) and (first_bar_close > first_bar_open) and (c_price > c_vwap) and rvol_ok and adx_ok:
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
                            f"alpha_04_gap_and_go LONG: Gap={gap_pct:+.2f}% | "
                            f"RVOL_0915={rvol:.2f}x | ADX={c_adx:.1f} | Entry={entry_price:.1f} | SL={curr_sl:.1f} | TP={curr_tp:.1f}"
                        )

                # 2. GAP DOWN & GO (BEARISH CONTINUATION)
                # - Gap between -0.50% and -2.50%
                # - First bar High <= Prev Close (Gap holds)
                # - First bar Close < Open and Close < VWAP
                # - TOD 09:15 RVOL >= 1.30x
                elif gap_valid and (gap_pct < 0) and (first_bar_high <= prev_day_close) and (first_bar_close < first_bar_open) and (c_price < c_vwap) and rvol_ok and adx_ok:
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
                            f"alpha_04_gap_and_go SHORT: Gap={gap_pct:+.2f}% | "
                            f"RVOL_0915={rvol:.2f}x | ADX={c_adx:.1f} | Entry={entry_price:.1f} | SL={curr_sl:.1f} | TP={curr_tp:.1f}"
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
