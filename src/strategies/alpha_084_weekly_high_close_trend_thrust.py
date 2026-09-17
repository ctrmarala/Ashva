"""
Ashva Quantitative Strategy: 5-Day High-Close Structural Trend Thrust (Alpha 84)
Category: MULTI_DAY_STRUCTURAL_BREAKOUT
Market Mechanism: MOMENTUM_EXPANSION

Hypothesis:
When a liquid equity sets a new 5-day highest close (Close[T-1] >= max(Close[T-2..T-6]) for Long,
or Close[T-1] <= min(Close[T-2..T-6]) for Short) with orderly volatility (Range[T-1] <= 0.70x ATR)
and macro trend alignment (Close[T-1] > 50 SMA), institutional accumulation continues into Day T.
A confirming morning directional breakout (09:15-09:30 IST) with gap >= 0.35%, solid candle body >= 65%,
and RVOL >= 1.25x triggers strong intraday continuation that operates orthogonally to inside-day regimes.
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


class Alpha84WeeklyHighCloseTrendThrust(BaseHypothesis, BaseStrategy):
    strategy_id = "84_alpha"
    hypothesis_id = "84_alpha"
    name = "84_alpha - 5-Day High-Close Structural Trend Thrust"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "min_rvol": 1.25,
            "min_body": 0.65,
            "max_range_atr": 0.70,
            "target_rr": 1.90,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="84_alpha",
            name="84_alpha - 5-Day High-Close Structural Trend Thrust",
            category="MULTI_DAY_STRUCTURAL_BREAKOUT",
            economic_rationale=(
                "Setting a 5-day high close on compressed range (<= 0.70x ATR) confirms institutional accumulation without exhaustion. "
                "Morning gap continuation on Day T captures explosive directional momentum."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="84_alpha", parameters=merged)
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
        h1 = h.shift(1)
        l1 = l.shift(1)
        rng1 = rng.shift(1)

        # 5-day rolling highest/lowest close shifted by 2 (comparing T-1 close to previous 5 days T-2 to T-6)
        max_c_5d = c.shift(2).rolling(5, min_periods=4).max()
        min_c_5d = c.shift(2).rolling(5, min_periods=4).min()

        is_5d_bull = (c1 >= max_c_5d) & (c1 > sma50)
        is_5d_bear = (c1 <= min_c_5d) & (c1 < sma50)
        max_rng_ratio = float(self.parameters.get("max_range_atr", 0.70))
        is_orderly = (rng1 <= (atr14 * max_rng_ratio))

        is_bull_trend_map = (is_5d_bull & is_orderly).to_dict()
        is_bear_trend_map = (is_5d_bear & is_orderly).to_dict()
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

            is_bull = is_bull_trend_map.get(c_date, False)
            is_bear = is_bear_trend_map.get(c_date, False)
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
