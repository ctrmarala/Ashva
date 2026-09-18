"""
Ashva Quantitative Strategy: 2-Period RSI Statistical Exhaustion Mean Reversion (Alpha 98)
Category: STATISTICAL_MEAN_REVERSION
Market Mechanism: MEAN_REVERSION

Hypothesis:
In range-bound/consolidating equities, extreme short-term stretched conditions (Daily RSI(2) <= 10.0 for Long,
or RSI(2) >= 90.0 for Short) represent temporary liquidity vacuum and panic retail capitulation.
When the morning opening bar (09:15-09:30 IST) prints a sharp price rejection wick (>= 40% wick in direction of extension)
and closes in the reversal direction with RVOL >= 1.20x, institutional mean-reversion engines harvest the spread
back toward the 20-period moving average. This operates completely uncorrelated to trend-breakout alphas.
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
from src.strategies.base import BaseStrategy
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha98RSI2StatisticalExhaustionReversion(BaseHypothesis, BaseStrategy):
    strategy_id = "98_alpha"
    hypothesis_id = "98_alpha"
    name = "98_alpha - 2-Period RSI Statistical Exhaustion Mean Reversion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.20,
            "min_body": 0.50,
            "min_wick": 0.35,
            "rsi_oversold": 12.0,
            "rsi_overbought": 88.0,
            "target_rr": 1.80,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="98_alpha",
            name="98_alpha - 2-Period RSI Statistical Exhaustion Mean Reversion",
            category="STATISTICAL_MEAN_REVERSION",
            economic_rationale=(
                "Extreme 2-period RSI stretch (<= 12 or >= 88) combined with morning rejection wicks in non-trending "
                "equities triggers rapid institutional mean-reversion back toward daily fair value."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="98_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25],
            "target_rr": [1.60, 1.80],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)

        if n < 80:
            out["signal"] = signals
            out["stop_loss"] = stop_loss
            out["take_profit"] = take_profit
            return out

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        c = daily_summary["day_close"]
        rng = h - l

        # Compute Daily RSI(2)
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(2, min_periods=2).mean()
        avg_loss = loss.rolling(2, min_periods=2).mean()
        rs = avg_gain / np.maximum(avg_loss, 1e-8)
        rsi2 = 100.0 - (100.0 / (1.0 + rs))

        atr14 = rng.shift(2).rolling(14, min_periods=10).mean()

        c1 = c.shift(1)
        h1 = h.shift(1)
        l1 = l.shift(1)
        rng1 = rng.shift(1)
        rsi1 = rsi2.shift(1)

        rsi_os = float(self.parameters.get("rsi_oversold", 12.0))
        rsi_ob = float(self.parameters.get("rsi_overbought", 88.0))

        is_oversold_reversal = (rsi1 <= rsi_os)
        is_overbought_reversal = (rsi1 >= rsi_ob)

        is_bull_reversion_map = is_oversold_reversal.to_dict()
        is_bear_reversion_map = is_overbought_reversal.to_dict()
        prev_h_map = h1.to_dict()
        prev_l_map = l1.to_dict()
        prev_c_map = c1.to_dict()

        vol_sma20 = out["volume"].rolling(20, min_periods=5).mean().fillna(out["volume"])
        rvol_series = out["volume"] / np.maximum(vol_sma20, 1.0)

        tr = np.maximum(
            out["high"] - out["low"],
            np.maximum(
                abs(out["high"] - out["close"].shift(1).fillna(out["open"])),
                abs(out["low"] - out["close"].shift(1).fillna(out["open"])),
            ),
        )
        atr_series = tr.rolling(14, min_periods=5).mean().fillna(tr)

        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        min_body = float(self.parameters.get("min_body", 0.50))
        min_wick = float(self.parameters.get("min_wick", 0.35))
        target_rr = float(self.parameters.get("target_rr", 1.80))

        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        closes = out["close"].values
        time_strs = out["time_str"].values
        rvol_arr = rvol_series.values
        atr_arr = atr_series.values

        pos = 0.0
        cur_sl = 0.0
        cur_tp = 0.0
        traded_today = False

        for i in range(1, n):
            t_str = time_strs[i]
            c_date = dates[i]
            p_date = dates[i - 1]

            if c_date != p_date:
                pos = 0.0
                cur_sl = 0.0
                cur_tp = 0.0
                traded_today = False

            if t_str >= "15:15":
                pos = 0.0
                signals[i] = 0.0
                stop_loss[i] = 0.0
                take_profit[i] = 0.0
                continue

            if pos != 0.0:
                h_i = highs[i]
                l_i = lows[i]

                if pos > 0:  # Long
                    if l_i <= cur_sl or h_i >= cur_tp:
                        pos = 0.0
                        signals[i] = 0.0
                        stop_loss[i] = 0.0
                        take_profit[i] = 0.0
                        continue
                    else:
                        signals[i] = 1.0
                        stop_loss[i] = cur_sl
                        take_profit[i] = cur_tp
                        continue
                elif pos < 0:  # Short
                    if h_i >= cur_sl or l_i <= cur_tp:
                        pos = 0.0
                        signals[i] = 0.0
                        stop_loss[i] = 0.0
                        take_profit[i] = 0.0
                        continue
                    else:
                        signals[i] = -1.0
                        stop_loss[i] = cur_sl
                        take_profit[i] = cur_tp
                        continue

            is_bull = is_bull_reversion_map.get(c_date, False)
            is_bear = is_bear_reversion_map.get(c_date, False)
            prev_h = prev_h_map.get(c_date, np.nan)
            prev_l = prev_l_map.get(c_date, np.nan)
            prev_c = prev_c_map.get(c_date, np.nan)

            if pd.isna(prev_h) or pd.isna(prev_l) or pd.isna(prev_c):
                signals[i] = 0.0
                stop_loss[i] = 0.0
                take_profit[i] = 0.0
                continue

            # Morning Reversion Entry (09:15 or 09:30 bar)
            if t_str in ("09:15", "09:30") and not traded_today:
                o = opens[i]
                h_bar = highs[i]
                l_bar = lows[i]
                c_bar = closes[i]
                rvol = rvol_arr[i]
                atr = atr_arr[i]
                bar_rng = h_bar - l_bar

                if bar_rng > 0:
                    # BULLISH MEAN REVERSION (Oversold Bounce):
                    # 1. Extreme RSI(2) oversold on T-1
                    # 2. Lower rejection wick >= min_wick or solid green candle
                    # 3. Bar closes higher than open
                    # 4. Volume confirms absorption: RVOL >= min_rvol
                    lower_wick = (min(o, c_bar) - l_bar) / bar_rng
                    body_bull = (c_bar - o) / bar_rng

                    if is_bull and c_bar > o and (lower_wick >= min_wick or body_bull >= min_body) and rvol >= min_rvol:
                        sl = max(l_bar - (0.2 * atr), c_bar - (1.1 * atr))
                        risk = c_bar - sl
                        if (0.0030 * c_bar) <= risk <= (0.0250 * c_bar):
                            tp = c_bar + (risk * target_rr)
                            pos = 1.0
                            cur_sl = sl
                            cur_tp = tp
                            traded_today = True
                            signals[i] = 1.0
                            stop_loss[i] = sl
                            take_profit[i] = tp
                            continue

                    # BEARISH MEAN REVERSION (Overbought Fade):
                    upper_wick = (h_bar - max(o, c_bar)) / bar_rng
                    body_bear = (o - c_bar) / bar_rng

                    if is_bear and c_bar < o and (upper_wick >= min_wick or body_bear >= min_body) and rvol >= min_rvol:
                        sl = min(h_bar + (0.2 * atr), c_bar + (1.1 * atr))
                        risk = sl - c_bar
                        if (0.0030 * c_bar) <= risk <= (0.0250 * c_bar):
                            tp = c_bar - (risk * target_rr)
                            pos = -1.0
                            cur_sl = sl
                            cur_tp = tp
                            traded_today = True
                            signals[i] = -1.0
                            stop_loss[i] = sl
                            take_profit[i] = tp
                            continue

            signals[i] = 0.0
            stop_loss[i] = 0.0
            take_profit[i] = 0.0

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        return out

    def on_bar(self, event: BarEvent) -> Optional[SignalEvent]:
        sym = event.symbol
        if sym not in self._current_pos:
            self._current_pos[sym] = 0.0
        return None
