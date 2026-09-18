"""
Ashva Quantitative Strategy: 2-Day Momentum Contraction Spring (Alpha 96)
Category: MOMENTUM_CONTRACTION_SPRING
Market Mechanism: MOMENTUM_EXPANSION

Hypothesis:
When an equity exhibits a strong momentum expansion on Day T-2 (Close[T-2] > High[T-3] for Long, or < Low[T-3] for Short)
followed by an ultra-tight volatility contraction on Day T-1 (Range[T-1] <= 0.65x ATR) that holds the breakout gain
(Close[T-1] >= Close[T-2] for Long, or Close[T-1] <= Close[T-2] for Short) above the 50 SMA (below 50 SMA for Short),
institutions are absorbing float before the next continuation impulse.
A confirming morning opening breakout (09:15-09:30 IST) with gap >= 0.35%, solid candle body >= 65%,
and RVOL >= 1.25x triggers an asymmetric high-expectancy trend expansion.
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


class Alpha96TwoDayMomentumContractionSpring(BaseHypothesis, BaseStrategy):
    strategy_id = "96_alpha"
    hypothesis_id = "96_alpha"
    name = "96_alpha - 2-Day Momentum Contraction Spring"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "min_rvol": 1.25,
            "min_body": 0.65,
            "max_range_atr": 0.65,
            "target_rr": 1.90,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="96_alpha",
            name="96_alpha - 2-Day Momentum Contraction Spring",
            category="MOMENTUM_CONTRACTION_SPRING",
            economic_rationale=(
                "Holding breakout gains on an ultra-narrow contraction day (<= 0.65x ATR) after a prior thrust "
                "proves supply absorption at new levels, creating explosive continuation upon Day T morning gap breakouts."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="96_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.15, 1.25],
            "target_rr": [1.70, 1.90],
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

        sma50 = c.shift(1).rolling(50, min_periods=20).mean()
        atr14 = rng.shift(2).rolling(14, min_periods=10).mean()

        c1 = c.shift(1)
        c2 = c.shift(2)
        h3 = h.shift(3)
        l3 = l.shift(3)
        h1 = h.shift(1)
        l1 = l.shift(1)
        rng1 = rng.shift(1)

        is_bull_setup = (c2 > h3) & (c1 >= c2) & (c1 > sma50)
        is_bear_setup = (c2 < l3) & (c1 <= c2) & (c1 < sma50)
        max_rng_ratio = float(self.parameters.get("max_range_atr", 0.65))
        is_contracted = (rng1 <= (atr14 * max_rng_ratio))

        is_bull_map = (is_bull_setup & is_contracted).to_dict()
        is_bear_map = (is_bear_setup & is_contracted).to_dict()
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

        min_gap = float(self.parameters.get("min_gap", 0.0035))
        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        min_body = float(self.parameters.get("min_body", 0.65))
        target_rr = float(self.parameters.get("target_rr", 1.90))

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

            is_bull = is_bull_map.get(c_date, False)
            is_bear = is_bear_map.get(c_date, False)
            prev_h = prev_h_map.get(c_date, np.nan)
            prev_l = prev_l_map.get(c_date, np.nan)
            prev_c = prev_c_map.get(c_date, np.nan)

            if pd.isna(prev_h) or pd.isna(prev_l) or pd.isna(prev_c):
                signals[i] = 0.0
                stop_loss[i] = 0.0
                take_profit[i] = 0.0
                continue

            # Morning Breakout on 09:15 bar
            if t_str == "09:15" and not traded_today:
                o = opens[i]
                h_bar = highs[i]
                l_bar = lows[i]
                c_bar = closes[i]
                rvol = rvol_arr[i]
                atr = atr_arr[i]
                bar_rng = h_bar - l_bar

                if bar_rng > 0:
                    body_bull = (c_bar - o) / bar_rng
                    if is_bull and (o >= prev_c * (1.0 + min_gap)) and c_bar > prev_h and body_bull >= min_body and rvol >= min_rvol:
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

                    body_bear = (o - c_bar) / bar_rng
                    if is_bear and (o <= prev_c * (1.0 - min_gap)) and c_bar < prev_l and body_bear >= min_body and rvol >= min_rvol:
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
