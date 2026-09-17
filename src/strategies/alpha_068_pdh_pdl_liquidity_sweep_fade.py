"""
Ashva Quantitative Strategy: Prior Day High/Low Liquidity Sweep Mean Reversion (Alpha 68)
Category: LIQUIDITY_SWEEP_MEAN_REVERSION
Market Mechanism: MEAN_REVERSION

Hypothesis:
Breakouts beyond Prior Day High (PDH) or Prior Day Low (PDL) frequently trigger retail stop orders.
When institutional participation fails to support the breakout and price closes back inside the prior day range
between 09:30 and 13:45 IST, smart money absorbs the trapped liquidity.
Fading the failed breakout targets mean reversion to the prior day value area with defined asymmetric risk.
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


class Alpha68PDHPDLLiquiditySweepFade(BaseHypothesis, BaseStrategy):
    strategy_id = "68_alpha"
    hypothesis_id = "68_alpha"
    name = "68_alpha — Prior Day High/Low Liquidity Sweep Fade"

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
            hypothesis_id="68_alpha",
            name="68_alpha — Prior Day High/Low Liquidity Sweep Fade",
            category="LIQUIDITY_SWEEP_MEAN_REVERSION",
            economic_rationale=(
                "Failed breakouts of prior-day extremes trap retail momentum participants. "
                "Immediate mean reversion back into the prior-day value area provides high win-rate reversion."
            ),
            target_instruments=[],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
        )
        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="68_alpha", parameters=merged)
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

        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
        pdh_map = daily_summary["day_high"].shift(1).to_dict()
        pdl_map = daily_summary["day_low"].shift(1).to_dict()

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

            pdh = pdh_map.get(c_date, np.nan)
            pdl = pdl_map.get(c_date, np.nan)

            if pd.isna(pdh) or pd.isna(pdl):
                signals[i] = 0.0
                stop_loss[i] = 0.0
                take_profit[i] = 0.0
                continue

            # Sweep window: 09:30 to 13:45
            if "09:30" <= t_str <= "13:45" and not traded_today:
                o = opens[i]
                h = highs[i]
                l = lows[i]
                c = closes[i]
                atr = atr_arr[i]

                # BEARISH SWEEP (FADE BREAKOUT ABOVE PDH):
                # 1. Bar High pierced above PDH
                # 2. Bar Close failed and fell back below PDH
                # 3. Bar is bearish (Close < Open)
                if h > pdh and c < pdh and c < o:
                    sl = h + (0.15 * atr)
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

                # BULLISH SWEEP (FADE BREAKDOWN BELOW PDL):
                # 1. Bar Low pierced below PDL
                # 2. Bar Close reclaimed and finished above PDL
                # 3. Bar is bullish (Close > Open)
                if l < pdl and c > pdl and c > o:
                    sl = l - (0.15 * atr)
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
