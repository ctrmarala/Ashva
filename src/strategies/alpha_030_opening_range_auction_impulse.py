"""
Ashva Quantitative Alpha Strategy — Alpha 30: Opening Range Auction Impulse (ORA-15M)
Alpha ID: 30_alpha
Version: v1.0.0
Author: AshvaQuantLab

Economic Hypothesis:
The first 15-minute bar (09:15-09:30 IST) establishes the initial auction balance.
When price decisively breaks the opening range with RVOL >= 1.30x and ADX(14) >= 18.0, institutional
imbalance forces directional momentum toward a 1.6R target with stop loss anchored at the opening range midpoint.

Contract Specification:
- Mechanism: BREAKOUT
- Timeframe: 15m
- Horizon: INTRADAY (15:15 IST hard square-off)
- Pure 1-bar impulse signal.
"""

from typing import Dict, List, Any, Optional
from datetime import time
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    StrategyHorizon,
    MarketMechanism,
)
from src.strategies.base import BaseStrategy
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha30OpeningRangeAuctionImpulse(BaseHypothesis, BaseStrategy):
    """
    Alpha 30: Opening Range Auction Impulse (ORA-15M).
    Pure 1-bar impulse signal with opening range midpoint stop loss and 1.6R target.
    """

    strategy_id = "30_alpha"
    hypothesis_id = "30_alpha"
    name = "30_alpha — Opening Range Auction Impulse"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_rvol": 1.30,
            "min_adx": 18.0,
            "min_or_pct": 0.0030,
            "max_or_pct": 0.0120,
            "target_rr": 1.60,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="30_alpha",
            name="30_alpha — Opening Range Auction Impulse",
            category="INTRADAY_AUCTION_BREAKOUT",
            economic_rationale=(
                "First 15m auction balance breakout with volume confirmation and ADX trend filter. "
                "Pure impulse signal with opening range midpoint stop loss."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="30_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.25, 1.35],
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

        out = TI.add_adx(out, period=14)
        out = TI.add_sma(out, period=20, price_col="volume", col_name="vol_sma20")

        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        dates = pd.to_datetime(out.index).date
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = (out["cum_pv"] / out["cum_v"].replace(0, np.nan)).bfill().ffill()

        times = [ts.time() for ts in pd.to_datetime(out.index)]
        t_0915 = time(9, 15)
        t_1300 = time(13, 0)

        closes = out["close"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        vol_smas = out["vol_sma20"].values
        vwaps = out["vwap"].values
        adxs = out["adx_14"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.30))
        min_adx = float(self.parameters.get("min_adx", 18.0))
        min_or = float(self.parameters.get("min_or_pct", 0.0030))
        max_or = float(self.parameters.get("max_or_pct", 0.0120))
        target_rr = float(self.parameters.get("target_rr", 1.60))

        current_date = None
        orb_high = 0.0
        orb_low = 0.0
        trade_taken_today = False

        for i in range(n):
            d = dates[i]
            t = times[i]

            if d != current_date:
                current_date = d
                orb_high = 0.0
                orb_low = 0.0
                trade_taken_today = False

            if t == t_0915:
                orb_high = highs[i]
                orb_low = lows[i]
                continue

            if orb_high > orb_low and not trade_taken_today and (t <= t_1300):
                or_range = orb_high - orb_low
                or_pct = or_range / orb_low if orb_low > 0 else 0.0

                if min_or <= or_pct <= max_or:
                    range_mid = (orb_high + orb_low) / 2.0
                    c_vol_sma = vol_smas[i]
                    vol_ok = (volumes[i] >= min_rvol * c_vol_sma) if not np.isnan(c_vol_sma) and c_vol_sma > 0 else True
                    adx_ok = (adxs[i] >= min_adx) if not np.isnan(adxs[i]) else True

                    # Bullish Breakout
                    if closes[i] > orb_high and closes[i] > vwaps[i] and vol_ok and adx_ok:
                        curr_sl = range_mid
                        risk_dist = max(closes[i] * 0.0025, closes[i] - curr_sl)
                        signals[i] = 1.0
                        stop_loss[i] = curr_sl
                        take_profit[i] = closes[i] + (target_rr * risk_dist)
                        trade_taken_today = True

                    # Bearish Breakdown
                    elif closes[i] < orb_low and closes[i] < vwaps[i] and vol_ok and adx_ok:
                        curr_sl = range_mid
                        risk_dist = max(closes[i] * 0.0025, curr_sl - closes[i])
                        signals[i] = -1.0
                        stop_loss[i] = curr_sl
                        take_profit[i] = closes[i] - (target_rr * risk_dist)
                        trade_taken_today = True

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
