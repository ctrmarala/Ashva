"""
Ashva Quantitative Strategy: Intraday VWAP & EMA Trend Pullback Continuation (Alpha 66)
Category: INTRADAY_TREND_PULLBACK
Market Mechanism: MOMENTUM_PULLBACK_REVERSAL

Hypothesis:
In liquid Indian equities, institutional algorithms accumulate or distribute relative to VWAP.
When a stock is in an established intraday trend (Price > VWAP, 20 EMA > 50 EMA on 15m),
shallow pullbacks into the 20 EMA / VWAP dynamic support band between 09:45 and 13:30 IST
represent institutional re-entry liquidity rather than trend reversals.
Rejection hammers reclaiming the 20 EMA provide high-expectancy trend continuation with tight risk.
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


class Alpha66IntradayVWAPTrendPullback(BaseHypothesis, BaseStrategy):
    strategy_id = "66_alpha"
    hypothesis_id = "66_alpha"
    name = "66_alpha — Intraday VWAP & EMA Trend Pullback"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "target_rr": 1.75,
            "min_risk_pct": 0.0030,
            "max_risk_pct": 0.0250,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="66_alpha",
            name="66_alpha — Intraday VWAP & EMA Trend Pullback",
            category="INTRADAY_TREND_PULLBACK",
            economic_rationale=(
                "Institutional algorithmic accumulation creates dynamic support at VWAP and the 20 EMA. "
                "Intraday pullbacks that test and reject the 20 EMA in the direction of the 50 EMA trend "
                "offer high-expectancy asymmetric continuation."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="66_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "target_rr": [1.60, 1.80],
            "min_risk_pct": [0.0025, 0.0035],
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

        # Technical Indicators
        typical = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical * out["volume"]
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_vol"] = out["volume"].groupby(dates).cumsum()
        vwap_series = out["cum_pv"] / np.maximum(out["cum_vol"], 1.0)

        ema20_series = out["close"].ewm(span=20, adjust=False).mean()
        ema50_series = out["close"].ewm(span=50, adjust=False).mean()

        tr = np.maximum(
            out["high"] - out["low"],
            np.maximum(
                abs(out["high"] - out["close"].shift(1).fillna(out["open"])),
                abs(out["low"] - out["close"].shift(1).fillna(out["open"])),
            ),
        )
        atr_series = tr.rolling(14, min_periods=5).mean().fillna(tr)

        target_rr = float(self.parameters.get("target_rr", 1.75))
        min_risk_pct = float(self.parameters.get("min_risk_pct", 0.0030))
        max_risk_pct = float(self.parameters.get("max_risk_pct", 0.0250))

        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        closes = out["close"].values
        time_strs = out["time_str"].values
        vwap_arr = vwap_series.values
        ema20_arr = ema20_series.values
        ema50_arr = ema50_series.values
        atr_arr = atr_series.values

        pos = 0.0
        cur_sl = 0.0
        cur_tp = 0.0

        for i in range(20, n):
            t_str = time_strs[i]
            c_date = dates[i]
            p_date = dates[i - 1]

            # Day boundary reset
            if c_date != p_date:
                pos = 0.0
                cur_sl = 0.0
                cur_tp = 0.0

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

            # Check new entry eligibility (Between 09:45 and 13:30)
            if "09:45" <= t_str <= "13:30":
                o = opens[i]
                h = highs[i]
                l = lows[i]
                c = closes[i]
                vwap = vwap_arr[i]
                ema20 = ema20_arr[i]
                ema50 = ema50_arr[i]
                atr = atr_arr[i]

                # BULLISH SETUP:
                # 1. Macro trend: EMA20 > EMA50 and Close > VWAP
                # 2. Pullback: Low <= EMA20 * 1.002 (touched/tested EMA20)
                # 3. Rejection: Close > EMA20 and Close > Open (bullish recovery)
                if ema20 > ema50 and c > vwap and l <= (ema20 * 1.002) and c > ema20 and c > o:
                    sl = min(l, ema20 - (0.4 * atr))
                    risk = c - sl
                    if (min_risk_pct * c) <= risk <= (max_risk_pct * c):
                        tp = c + (risk * target_rr)
                        pos = 1.0
                        cur_sl = sl
                        cur_tp = tp
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = tp
                        continue

                # BEARISH SETUP:
                # 1. Macro trend: EMA20 < EMA50 and Close < VWAP
                # 2. Pullback: High >= EMA20 * 0.998 (touched/tested EMA20)
                # 3. Rejection: Close < EMA20 and Close < Open (bearish rejection)
                if ema20 < ema50 and c < vwap and h >= (ema20 * 0.998) and c < ema20 and c < o:
                    sl = max(h, ema20 + (0.4 * atr))
                    risk = sl - c
                    if (min_risk_pct * c) <= risk <= (max_risk_pct * c):
                        tp = c - (risk * target_rr)
                        pos = -1.0
                        cur_sl = sl
                        cur_tp = tp
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
