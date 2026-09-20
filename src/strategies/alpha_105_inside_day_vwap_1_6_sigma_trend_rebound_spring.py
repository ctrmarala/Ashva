"""
Ashva Quantitative Strategy: Inside Day VWAP 1.6-Sigma Trend Rebound Spring (105_alpha)
Category: STATISTICAL_VWAP_MEAN_REVERSION
Market Mechanism: MEAN_REVERSION

Hypothesis:
Inside Day compression tests lower/upper VWAP bands in 50 SMA trend, sparking sharp mean reversion.
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


class Alpha105InsideDayVwapSpring(BaseHypothesis, BaseStrategy):
    strategy_id = "105_alpha"
    hypothesis_id = "105_alpha"
    name = "105_alpha — Inside Day VWAP 1.6-Sigma Trend Rebound Spring"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "max_gap": 0.0250,
            "min_rvol": 1.20,
            "min_body": 0.60,
            "target_rr": 1.65,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="105_alpha",
            name="105_alpha — Inside Day VWAP 1.6-Sigma Trend Rebound Spring",
            category="STATISTICAL_VWAP_MEAN_REVERSION",
            economic_rationale="Inside Day compression tests lower/upper VWAP bands in 50 SMA trend, sparking sharp mean reversion.",
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="105_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25],
            "target_rr": [1.55, 1.75],
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
            day_open=("open", "first"),
            day_vol=("volume", "sum"),
        )
        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        c = daily_summary["day_close"]
        rng = h - l
        atr14 = rng.shift(2).rolling(14, min_periods=10).mean()
        sma50 = c.shift(2).rolling(50, min_periods=30).mean()
        daily_atr_map = atr14.to_dict()
        daily_sma_map = sma50.to_dict()

        if "inside" == "inside":
            coil_flags = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
        elif "inside" == "dual_inside":
            coil_flags = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2)) & (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))
        elif "inside" == "3day":
            coil_flags = (c.shift(1) > c.shift(2)) & (c.shift(2) > c.shift(3))
        else:
            coil_flags = (rng.shift(1) <= 0.8 * atr14)
        coil_map = coil_flags.to_dict()

        # Cumulative VWAP
        pv = out["close"] * out["volume"]
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_vol"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = out["cum_pv"] / np.maximum(out["cum_vol"], 1.0)
        
        diff_sq = (out["close"] - out["vwap"]) ** 2 * out["volume"]
        cum_diff_sq = diff_sq.groupby(dates).cumsum()
        out["vwap_std"] = np.sqrt(cum_diff_sq / np.maximum(out["cum_vol"], 1.0))

        target_rr = float(self.parameters.get("target_rr", 1.65))
        current_day = None
        traded_today = False

        for i in range(1, n):
            d = dates[i]
            t_str = out["time_str"].iloc[i]

            if d != current_day:
                current_day = d
                traded_today = False

            if traded_today:
                continue

            if not ("09:30" <= t_str <= "11:30"):
                continue

            cur_atr = daily_atr_map.get(d, np.nan)
            cur_sma = daily_sma_map.get(d, np.nan)
            is_coiled = coil_map.get(d, False)
            if pd.isna(cur_atr) or cur_atr <= 0 or not is_coiled or pd.isna(cur_sma):
                continue

            vwap = out["vwap"].iloc[i]
            v_std = out["vwap_std"].iloc[i]
            if v_std <= 0:
                continue

            o = out["open"].iloc[i]
            h_bar = out["high"].iloc[i]
            l_bar = out["low"].iloc[i]
            c_bar = out["close"].iloc[i]

            # Long Setup: Macro Bull (Close > 50 SMA), dip tests lower VWAP band (-1.5 to -2.0 sigma) and bounces bullishly
            lower_band = vwap - (1.6 * v_std)
            if c_bar > cur_sma and l_bar <= lower_band and c_bar > lower_band and c_bar > o:
                sl = l_bar
                risk = max(c_bar * 0.0025, c_bar - sl)
                signals[i] = 1.0
                stop_loss[i] = sl
                take_profit[i] = c_bar + (target_rr * risk)
                traded_today = True
                continue

            # Short Setup: Macro Bear (Close < 50 SMA), surge tests upper VWAP band (+1.6 sigma) and rejects bearishly
            upper_band = vwap + (1.6 * v_std)
            if c_bar < cur_sma and h_bar >= upper_band and c_bar < upper_band and c_bar < o:
                sl = h_bar
                risk = max(c_bar * 0.0025, sl - c_bar)
                signals[i] = -1.0
                stop_loss[i] = sl
                take_profit[i] = c_bar - (target_rr * risk)
                traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        return out

    def on_bar(self, event: BarEvent) -> List[SignalEvent]:
        sym = event.symbol
        t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)
        if t_str >= "15:15" and pos != 0.0:
            self._current_pos[sym] = 0.0
            return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
        return []
