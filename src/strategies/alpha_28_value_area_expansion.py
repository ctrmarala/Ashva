"""
Ashva Quantitative Alpha 28: Intraday Volume-Weighted Value Area Expansion
Hypothesis:
    In auction market theory, the morning session (09:15 to 11:15 IST) establishes the
    initial Value Area (VA) containing 70% of the morning's traded volume, bounded by
    Value Area High (VAH) and Value Area Low (VAL). When price consolidates within the
    Value Area and subsequently breaks out outside VAH or VAL with elevated volume,
    the market is rejecting fair value, initiating a multi-hour directional discovery drift.

Mechanism:
    Auction Market Theory price discovery outside the morning 70% Volume Profile Value Area.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    StrategyHorizon,
    MarketMechanism,
)
from src.features.indicators import TechnicalIndicators


class Alpha28ValueAreaExpansion(BaseHypothesis):
    """
    Alpha 28: Value Area High/Low Breakout & Discovery Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        meta = HypothesisMetadata(
            hypothesis_id="alpha_28",
            name="ALPHA_28_VALUE_AREA_EXPANSION",
            category="AUCTION_MARKET_THEORY",
            economic_rationale=(
                "When price accepts outside the morning 70% Value Area with strong volume, "
                "it signifies that larger timeframe participants have shifted fair value perception, "
                "causing liquidity providers to retreat and creating directional momentum into the close."
            ),
            target_instruments=["NIFTY50_LIQUID"],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        super().__init__(meta, parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "value_area_pct": [0.68, 0.70, 0.75],
            "min_rvol": [1.10, 1.25, 1.40],
            "min_body_ratio": [0.50, 0.60],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                out.index = pd.to_datetime(out.index)

        dates = out.index.date
        times = out.index.time

        # 1. Daily ATR (14-day) anchored to prior days
        daily_df = out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily_df) >= 14:
            daily_atr_df = TechnicalIndicators.add_atr(daily_df, period=14)
            daily_atr_prev = daily_atr_df["atr_14"].shift(1)
            atr_map = daily_atr_prev.to_dict()
            out["daily_atr"] = [atr_map.get(pd.Timestamp(d), np.nan) for d in dates]
        else:
            out["daily_atr"] = (out["high"] - out["low"]).rolling(14).mean()

        out["daily_atr"] = out["daily_atr"].ffill().bfill()

        # 2. Time-of-Day Mean Volume Baseline
        tod_rolling = out.groupby(times)["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        # Strategy Hyperparameters
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        min_body_pct = float(self.parameters.get("min_body_ratio", 0.50))
        target_rr = float(self.parameters.get("target_rr", 1.50))

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

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        # Value Area Tracking
        morning_bars_highs = []
        morning_bars_lows = []
        morning_bars_vols = []
        morning_bars_closes = []
        vah = 0.0
        val = 0.0
        va_established = False

        t_1115 = pd.to_datetime("11:15:00").time()
        t_1330 = pd.to_datetime("13:30:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(1, n):
            bar_date = dates[i]
            bar_time = times[i]

            # Reset on new trading day
            if bar_date != current_day:
                current_day = bar_date
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""
                morning_bars_highs = []
                morning_bars_lows = []
                morning_bars_vols = []
                morning_bars_closes = []
                vah = 0.0
                val = 0.0
                va_established = False

            # Intraday 15:15 EOD Square-Off
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 28 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across holding bars
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

            if pd.isna(c_atr) or c_atr <= 0:
                continue

            # Build Morning Value Area (09:15 to 11:15)
            if bar_time < t_1115:
                morning_bars_highs.append(c_high)
                morning_bars_lows.append(c_low)
                morning_bars_vols.append(c_vol)
                morning_bars_closes.append(c_close)
                continue

            # Establish Value Area at 11:15
            if not va_established:
                if len(morning_bars_closes) >= 6:
                    # Approximate Value Area as Volume-Weighted Mean +/- 1.0 Volume-Weighted Std Dev (covers ~68-70% volume)
                    v_weights = np.array(morning_bars_vols, dtype=float)
                    p_closes = np.array(morning_bars_closes, dtype=float)
                    total_v = np.sum(v_weights)
                    if total_v > 0:
                        vwap_m = np.sum(p_closes * v_weights) / total_v
                        vw_var = np.sum(v_weights * ((p_closes - vwap_m) ** 2)) / total_v
                        vw_std = np.sqrt(vw_var)
                        vah = vwap_m + vw_std
                        val = vwap_m - vw_std
                        va_established = True
                if not va_established:
                    continue

            if traded_today or bar_time > t_1330:
                continue

            # Check Expansion / Breakout Candidate
            rvol = c_vol / max(1.0, c_tod)
            if rvol < min_rvol:
                continue

            bar_range = c_high - c_low
            body = abs(c_close - c_open)
            body_ratio = body / max(0.01, bar_range)

            if body_ratio < min_body_pct:
                continue

            # Case 1: Bullish Value Area High (VAH) Breakout
            if (c_close > vah) and (c_close > c_open) and (c_open <= vah or c_low <= vah):
                curr_state = 1.0
                stop_dist = max(c_close - vah + 0.10 * c_atr, 0.25 * c_atr)
                curr_sl = c_close - stop_dist
                curr_tp = c_close + (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 28 VAH EXPANSION: Close={c_close:.1f} > VAH={vah:.1f} | VAL={val:.1f} | "
                    f"Body={body_ratio*100:.0f}% | RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
                )
                signals[i] = 1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

            # Case 2: Bearish Value Area Low (VAL) Breakdown
            elif (c_close < val) and (c_close < c_open) and (c_open >= val or c_high >= val):
                curr_state = -1.0
                stop_dist = max(val - c_close + 0.10 * c_atr, 0.25 * c_atr)
                curr_sl = c_close + stop_dist
                curr_tp = c_close - (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 28 VAL BREAKDOWN: Close={c_close:.1f} < VAL={val:.1f} | VAH={vah:.1f} | "
                    f"Body={body_ratio*100:.0f}% | RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
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
