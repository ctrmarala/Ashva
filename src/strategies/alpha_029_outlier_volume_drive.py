"""
Ashva Quantitative Alpha Strategy — Alpha 29: Outlier Volume Morning Drive (OVD-15M)
Alpha ID: 29_alpha
Version: v1.0.0
Author: AshvaQuantLab

Economic Hypothesis:
An opening 15-minute bar with extreme relative volume (RVOL >= 2.0x) and strong directional body (>= 60%)
reflects aggressive institutional block accumulation or liquidation. Entering on the drive close with the
09:15 candle extreme as a structural stop captures the multi-hour continuation with high payoff asymmetry.

Contract Specification:
- Mechanism: VOLATILITY
- Timeframe: 15m
- Horizon: INTRADAY (15:15 IST hard square-off)
- Pure 1-bar impulse signal.
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


class Alpha29OutlierVolumeDrive(BaseHypothesis, BaseStrategy):
    """
    Alpha 29: Outlier Volume Morning Drive (OVD-15M).
    Pure 1-bar impulse signal with 15m candle extreme stop loss and 1.5R target.
    """

    strategy_id = "29_alpha"
    hypothesis_id = "29_alpha"
    name = "29_alpha — Outlier Volume Morning Drive"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 2.00,
            "min_body": 0.60,
            "min_range_pct": 0.0035,
            "target_rr": 1.50,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="29_alpha",
            name="29_alpha — Outlier Volume Morning Drive",
            category="VOLUME_SHOCK_MOMENTUM",
            economic_rationale=(
                "Extreme opening volume shock (RVOL >= 2.0x) with strong body drive. "
                "Pure impulse signal with tight 15m candle stop loss."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.VOLATILITY,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="29_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.80, 2.20],
            "target_rr": [1.50, 1.75],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)

        if n < 50:
            out["signal"] = signals; out["stop_loss"] = stop_loss; out["take_profit"] = take_profit
            return out

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        tod_vol = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_vol

        t_0915 = time(9, 15)

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values

        min_rvol = float(self.parameters.get("min_rvol", 2.00))
        min_body = float(self.parameters.get("min_body", 0.60))
        min_rng = float(self.parameters.get("min_range_pct", 0.0035))
        target_rr = float(self.parameters.get("target_rr", 1.50))

        current_day = None
        traded_today = False

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today:
                continue

            if bar_time == t_0915:
                rvol = volumes[i] / max(1.0, tod_vols[i])
                bar_range = highs[i] - lows[i]
                range_pct = bar_range / opens[i] if opens[i] > 0 else 0.0
                body_size = abs(closes[i] - opens[i])
                body_ratio = (body_size / bar_range) if bar_range > 0 else 0.0

                if rvol >= min_rvol and range_pct >= min_rng and body_ratio >= min_body:
                    # Bullish Outlier Volume Drive
                    if closes[i] > opens[i]:
                        sl = lows[i]
                        risk = max(closes[i] * 0.0025, closes[i] - sl)
                        signals[i] = 1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] + (target_rr * risk)
                        traded_today = True

                    # Bearish Outlier Volume Drive
                    elif closes[i] < opens[i]:
                        sl = highs[i]
                        risk = max(closes[i] * 0.0025, sl - closes[i])
                        signals[i] = -1.0
                        stop_loss[i] = sl
                        take_profit[i] = closes[i] - (target_rr * risk)
                        traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        return out

    def on_bar(self, event: BarEvent) -> List[SignalEvent]:
        sym = event.symbol; t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)
        if t_str >= "15:15" and pos != 0.0:
            self._current_pos[sym] = 0.0
            return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
        return []
