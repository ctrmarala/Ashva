"""
Ashva Quantitative Strategy: Opening Range RVOL Directional Breakout (Alpha 67)
Category: OPENING_RANGE_BREAKOUT
Market Mechanism: MOMENTUM_EXPANSION

Hypothesis:
An opening range breakout (09:15 to 09:30 IST) accompanied by anomalous institutional Relative Volume (RVOL >= 1.5x)
in the direction of the 50 EMA trend signals aggressive order flow imbalance.
Entering at the close of the confirming breakout bar between 09:30 and 11:30 IST captures strong midday trend expansion.
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


class Alpha67OpeningRangeRVOLBreakout(BaseHypothesis, BaseStrategy):
    strategy_id = "67_alpha"
    hypothesis_id = "67_alpha"
    name = "67_alpha — Opening Range RVOL Directional Breakout"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.50,
            "target_rr": 1.70,
            "min_risk_pct": 0.0035,
            "max_risk_pct": 0.0250,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="67_alpha",
            name="67_alpha — Opening Range RVOL Directional Breakout",
            category="OPENING_RANGE_BREAKOUT",
            economic_rationale=(
                "Early morning opening range expansion backed by >= 1.5x RVOL reflects institutional directional conviction. "
                "Aligning with the 50 EMA filters false shakes and drives sustained expansion into afternoon auctions."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="67_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.40, 1.60],
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

        ema50_series = out["close"].ewm(span=50, adjust=False).mean()
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

        min_rvol = float(self.parameters.get("min_rvol", 1.50))
        target_rr = float(self.parameters.get("target_rr", 1.70))
        min_risk_pct = float(self.parameters.get("min_risk_pct", 0.0035))
        max_risk_pct = float(self.parameters.get("max_risk_pct", 0.0250))

        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        closes = out["close"].values
        time_strs = out["time_str"].values
        ema50_arr = ema50_series.values
        rvol_arr = rvol_series.values
        atr_arr = atr_series.values

        pos = 0.0
        cur_sl = 0.0
        cur_tp = 0.0
        
        # Day session state
        cur_day_orb_high = -1.0
        cur_day_orb_low = -1.0
        traded_today = False

        for i in range(1, n):
            t_str = time_strs[i]
            c_date = dates[i]
            p_date = dates[i - 1]

            # Day boundary reset
            if c_date != p_date:
                pos = 0.0
                cur_sl = 0.0
                cur_tp = 0.0
                cur_day_orb_high = -1.0
                cur_day_orb_low = -1.0
                traded_today = False

            # Capture 09:15 opening bar range (the first 15m candle)
            if t_str == "09:15":
                cur_day_orb_high = highs[i]
                cur_day_orb_low = lows[i]

            # 15:15 EOD square-off
            if t_str >= "15:15":
                pos = 0.0
                signals[i] = 0.0
                stop_loss[i] = 0.0
                take_profit[i] = 0.0
                continue

            # In position: check intrabar barriers
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

            # Entry condition: 09:30 to 11:30, only 1 trade per stock per day
            if "09:30" <= t_str <= "11:30" and not traded_today and cur_day_orb_high > 0:
                o = opens[i]
                h = highs[i]
                l = lows[i]
                c = closes[i]
                ema50 = ema50_arr[i]
                rvol = rvol_arr[i]
                atr = atr_arr[i]

                # BULLISH ORB BREAKOUT:
                # 1. Close > ORB High
                # 2. RVOL >= min_rvol
                # 3. Macro Trend: Close > EMA50
                if c > cur_day_orb_high and rvol >= min_rvol and c > ema50 and c > o:
                    sl = max(cur_day_orb_low, c - (1.1 * atr))
                    risk = c - sl
                    if (min_risk_pct * c) <= risk <= (max_risk_pct * c):
                        tp = c + (risk * target_rr)
                        pos = 1.0
                        cur_sl = sl
                        cur_tp = tp
                        traded_today = True
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = tp
                        continue

                # BEARISH ORB BREAKOUT:
                # 1. Close < ORB Low
                # 2. RVOL >= min_rvol
                # 3. Macro Trend: Close < EMA50
                if c < cur_day_orb_low and rvol >= min_rvol and c < ema50 and c < o:
                    sl = min(cur_day_orb_high, c + (1.1 * atr))
                    risk = sl - c
                    if (min_risk_pct * c) <= risk <= (max_risk_pct * c):
                        tp = c - (risk * target_rr)
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
