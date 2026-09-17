"""
Ashva Quantitative Strategy: Camarilla Multi-Day Range Expansion Breakout (Alpha 71)
Category: CAMARILLA_RANGE_EXPANSION
Market Mechanism: BREAKOUT

Hypothesis:
When price opens and expands beyond the 5-day Camarilla H4 (Breakout Resistance) or L4 (Breakdown Support) levels
on the morning 09:15-09:30 IST candle with anomalous volume (RVOL >= 1.25x) and trend alignment (50 SMA),
institutional liquidity imbalances trigger high-velocity expansion. This strategy operates orthogonally to inside-day regimes.
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


class Alpha71CamarillaMultiDayRangeExpansion(BaseHypothesis, BaseStrategy):
    strategy_id = "71_alpha"
    hypothesis_id = "71_alpha"
    name = "71_alpha — Camarilla Multi-Day Range Expansion Breakout"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap": 0.0035,
            "min_rvol": 1.25,
            "min_body": 0.60,
            "target_rr": 1.80,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="71_alpha",
            name="71_alpha — Camarilla Multi-Day Range Expansion Breakout",
            category="CAMARILLA_RANGE_EXPANSION",
            economic_rationale=(
                "5-day Camarilla H4/L4 boundary breaches with anomalous volume capture institutional inventory expansion "
                "independent of single-day compression setups."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="71_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.20, 1.30],
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

        # 5-day Camarilla levels shifted strictly by 1 day (zero lookahead)
        h5 = h.shift(1).rolling(5, min_periods=4).max()
        l5 = l.shift(1).rolling(5, min_periods=4).min()
        c1 = c.shift(1)
        rng5 = h5 - l5

        cam_h4 = (c1 + rng5 * (1.1 / 2.0)).to_dict()
        cam_l4 = (c1 - rng5 * (1.1 / 2.0)).to_dict()
        sma50_map = c1.rolling(50, min_periods=20).mean().to_dict()
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
        min_body = float(self.parameters.get("min_body", 0.60))
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

            h4 = cam_h4.get(c_date, np.nan)
            l4 = cam_l4.get(c_date, np.nan)
            sma50 = sma50_map.get(c_date, np.nan)
            prev_c = prev_c_map.get(c_date, np.nan)

            if pd.isna(h4) or pd.isna(l4) or pd.isna(sma50) or pd.isna(prev_c):
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
                    # BULLISH CAMARILLA H4 BREAKOUT:
                    # 1. Close > Camarilla H4
                    # 2. Close > 50 SMA
                    # 3. Gap >= min_gap
                    # 4. Solid Bullish Body >= min_body
                    # 5. Volume: RVOL >= min_rvol
                    body_bull = (c_bar - o) / bar_rng
                    if c_bar > h4 and c_bar > sma50 and (o >= prev_c * (1.0 + min_gap)) and body_bull >= min_body and rvol >= min_rvol:
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

                    # BEARISH CAMARILLA L4 BREAKDOWN:
                    # 1. Close < Camarilla L4
                    # 2. Close < 50 SMA
                    # 3. Gap <= -min_gap
                    # 4. Solid Bearish Body >= min_body
                    # 5. Volume: RVOL >= min_rvol
                    body_bear = (o - c_bar) / bar_rng
                    if c_bar < l4 and c_bar < sma50 and (o <= prev_c * (1.0 - min_gap)) and body_bear >= min_body and rvol >= min_rvol:
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
