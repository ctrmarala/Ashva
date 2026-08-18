"""
Ashva Quantitative Alpha 30: 30-Minute Volatility Squeeze Trend Continuation
Hypothesis:
    On 30-minute bars, when Bollinger Bands (20, 2.0) compress inside Keltner Channels (20, 1.5 ATR),
    the market has entered an extreme volatility squeeze. When the squeeze releases with an expansion
    bar aligned with the 30-minute EMA20 trend direction, the stock tends to experience sustained,
    low-noise directional continuation into the close, generating high per-trade payoff ratios.

Mechanism:
    30-minute volatility cycle transition (Squeeze to Expansion) aligned with intermediate trend.
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


class Alpha30MiddaySqueezeTrend(BaseHypothesis):
    """
    Alpha 30: 30m Volatility Squeeze & Trend Continuation Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        meta = HypothesisMetadata(
            hypothesis_id="alpha_30",
            name="ALPHA_30_MIDDAY_SQUEEZE_TREND",
            category="VOLATILITY_SQUEEZE",
            economic_rationale=(
                "30-minute consolidation builds coiled institutional energy. When the squeeze fires "
                "in the direction of the underlying EMA trend, larger participants enter the market, "
                "producing large-range trend bars that overcome Indian statutory friction."
            ),
            target_instruments=["NIFTY50_LIQUID"],
            timeframe="30m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        super().__init__(meta, parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "bb_length": [20],
            "bb_std": [2.0],
            "kc_mult": [1.5],
            "target_rr": [1.50, 2.00],
            "max_entry_hour": [13, 14],
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

        # 1. Resample to 30m if input is 15m or directly calculate indicators
        # Calculate Bollinger Bands (20, 2.0) and Keltner Channels (20, 10, 1.5)
        out = TechnicalIndicators.add_bollinger_bands(out, window=20, num_std=2.0)
        out = TechnicalIndicators.add_keltner_channels(out, ema_period=20, atr_period=10, multiplier=1.5)
        out = TechnicalIndicators.add_ema(out, period=20, price_col="close")
        
        # Squeeze condition: BB Upper < KC Upper AND BB Lower > KC Lower
        out["is_squeeze"] = (out["bb_upper_20"] < out["kc_upper_20"]) & (out["bb_lower_20"] > out["kc_lower_20"])
        out["squeeze_prev"] = out["is_squeeze"].shift(1).fillna(False)

        # 2. Daily ATR for Stop Sizing
        daily_df = out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily_df) >= 14:
            daily_atr_df = TechnicalIndicators.add_atr(daily_df, period=14)
            daily_atr_prev = daily_atr_df["atr_14"].shift(1)
            atr_map = daily_atr_prev.to_dict()
            out["daily_atr"] = [atr_map.get(pd.Timestamp(d), np.nan) for d in dates]
        else:
            out["daily_atr"] = out["atr_20"]

        out["daily_atr"] = out["daily_atr"].ffill().bfill()

        # Strategy Hyperparameters
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_entry_hour = int(self.parameters.get("max_entry_hour", 14))

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        ema20s = out["ema_20"].values
        daily_atrs = out["daily_atr"].values
        is_squeezes = out["is_squeeze"].values
        prev_squeezes = out["squeeze_prev"].values
        bb_uppers = out["bb_upper_20"].values
        bb_lowers = out["bb_lower_20"].values

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_0945 = pd.to_datetime("09:45:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(25, n):
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

            # Intraday 15:15 EOD Square-Off
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 30 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across holding bars
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or bar_time < t_0945 or bar_time.hour > max_entry_hour:
                continue

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_ema = ema20s[i]
            c_atr = daily_atrs[i]
            c_squeeze = is_squeezes[i]
            p_squeeze = prev_squeezes[i]
            b_up = bb_uppers[i]
            b_low = bb_lowers[i]

            if pd.isna(c_atr) or c_atr <= 0 or pd.isna(c_ema):
                continue

            # Squeeze Release Event: Was in Squeeze previously, now expanding out of squeeze
            squeeze_fired = (p_squeeze and not c_squeeze) or (not c_squeeze and (c_close > b_up or c_close < b_low))
            if not squeeze_fired:
                continue

            # Case 1: Bullish Squeeze Firing (Close > EMA20 and Close > Open)
            if (c_close > c_ema) and (c_close > c_open) and (c_close >= b_up):
                curr_state = 1.0
                stop_dist = max(c_close - c_ema + 0.10 * c_atr, 0.30 * c_atr)
                curr_sl = c_close - stop_dist
                curr_tp = c_close + (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 30 SQUEEZE LONG: Close={c_close:.1f} > BB_Up={b_up:.1f} | EMA20={c_ema:.1f} | "
                    f"SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                )
                signals[i] = 1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

            # Case 2: Bearish Squeeze Firing (Close < EMA20 and Close < Open)
            elif (c_close < c_ema) and (c_close < c_open) and (c_close <= b_low):
                curr_state = -1.0
                stop_dist = max(c_ema - c_close + 0.10 * c_atr, 0.30 * c_atr)
                curr_sl = c_close + stop_dist
                curr_tp = c_close - (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 30 SQUEEZE SHORT: Close={c_close:.1f} < BB_Low={b_low:.1f} | EMA20={c_ema:.1f} | "
                    f"SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
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
