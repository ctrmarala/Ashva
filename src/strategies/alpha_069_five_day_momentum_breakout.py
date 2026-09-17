"""
Ashva Quantitative Strategy: 5-Day High/Low Momentum Breakout (Alpha 69)
Category: MULTI_DAY_MOMENTUM_BREAKOUT
Market Mechanism: MOMENTUM_EXPANSION

Hypothesis:
When a liquid equity in a confirmed daily uptrend (Close > 50-day SMA) breaks out above its 5-day rolling high
on the 09:15-09:30 IST morning candle with anomalous volume (RVOL >= 1.25x), institutional demand drives
sustained intraday price discovery. This strategy operates independently of inside-day compression regimes.
"""

from typing import Dict, List, Any, Optional
from datetime import time
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


class Alpha69FiveDayMomentumBreakout(BaseHypothesis, BaseStrategy):
    strategy_id = "69_alpha"
    hypothesis_id = "69_alpha"
    name = "69_alpha — 5-Day Momentum Breakout"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.25,
            "min_gap": 0.0030,
            "target_rr": 1.65,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="69_alpha",
            name="69_alpha — 5-Day Momentum Breakout",
            category="MULTI_DAY_MOMENTUM_BREAKOUT",
            economic_rationale=(
                "Fresh 5-day multi-session high breakouts on strong morning volume trigger institutional "
                "momentum continuation that expands throughout the trading day."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="69_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.20, 1.30],
            "target_rr": [1.50, 1.75],
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

        # 5-day rolling high and low shifted strictly by 1 day (zero lookahead)
        roll_h5 = h.shift(1).rolling(5, min_periods=4).max().to_dict()
        roll_l5 = l.shift(1).rolling(5, min_periods=4).min().to_dict()
        sma50_map = c.shift(1).rolling(50, min_periods=20).mean().to_dict()
        prior_close_map = c.shift(1).to_dict()

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

        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        min_gap = float(self.parameters.get("min_gap", 0.0030))
        target_rr = float(self.parameters.get("target_rr", 1.65))

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

            h5 = roll_h5.get(c_date, np.nan)
            l5 = roll_l5.get(c_date, np.nan)
            sma50 = sma50_map.get(c_date, np.nan)
            prev_c = prior_close_map.get(c_date, np.nan)

            if pd.isna(h5) or pd.isna(l5) or pd.isna(sma50) or pd.isna(prev_c):
                signals[i] = 0.0
                stop_loss[i] = 0.0
                take_profit[i] = 0.0
                continue

            # Morning Breakout on 09:15 or 09:30 bar
            if t_str in ["09:15", "09:30"] and not traded_today:
                o = opens[i]
                h_bar = highs[i]
                l_bar = lows[i]
                c_bar = closes[i]
                rvol = rvol_arr[i]
                atr = atr_arr[i]

                # BULLISH 5-DAY BREAKOUT:
                # 1. Close > 5-day High
                # 2. Macro Trend: Close > 50 SMA
                # 3. Volume: RVOL >= min_rvol
                # 4. Confirmation: Close > Open and positive gap
                if c_bar > h5 and c_bar > sma50 and rvol >= min_rvol and c_bar > o and (o >= prev_c * (1.0 + min_gap)):
                    sl = max(l_bar, c_bar - (1.1 * atr))
                    risk = c_bar - sl
                    if (0.0035 * c_bar) <= risk <= (0.0250 * c_bar):
                        tp = c_bar + (risk * target_rr)
                        pos = 1.0
                        cur_sl = sl
                        cur_tp = tp
                        traded_today = True
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = tp
                        continue

                # BEARISH 5-DAY BREAKDOWN:
                # 1. Close < 5-day Low
                # 2. Macro Trend: Close < 50 SMA
                # 3. Volume: RVOL >= min_rvol
                # 4. Confirmation: Close < Open and negative gap
                if c_bar < l5 and c_bar < sma50 and rvol >= min_rvol and c_bar < o and (o <= prev_c * (1.0 - min_gap)):
                    sl = min(h_bar, c_bar + (1.1 * atr))
                    risk = sl - c_bar
                    if (0.0035 * c_bar) <= risk <= (0.0250 * c_bar):
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
