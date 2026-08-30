"""
Ashva Quantitative Alpha Strategy — Alpha 5: Morning Volatility Expansion (MVE)
Alpha ID: 5_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
In Indian cash equities, institutional order flow and directional expansion predominantly occur during
the morning session (09:30 - 11:30 IST) following pre-market positioning and opening price discovery.
Entering explosive breakouts exclusively in this morning window from pre-compressed volatility bases
provides 4 to 5 hours of unhindered trend runway, allowing trades to reach asymmetric +3.5% targets
before the afternoon chop and 15:15 square-off.

Economic Mechanism:
Morning institutional participation establishes the day's trend structure. By cutting off all new entries
past 11:30 IST, we eliminate low-runway late afternoon trades that time-exit flat while incurring full
statutory friction. Trailing stops and wider 3.5:1 profit targets maximize gain capture on genuine trend days.

Contract Specification:
- Entry Window: Strictly 09:30 to 11:30 IST (Zero afternoon entries).
- Compression Filter: ATR14 <= 0.95x ATR SMA20 (Pre-expansion coiling).
- Long Entry: Close > Opening Range High (09:15-09:45 High), Volume >= 2.2x Vol SMA20, Close > EMA50.
- Short Entry: Close < Opening Range Low (09:15-09:45 Low), Volume >= 2.2x Vol SMA20, Close < EMA50.
- Stop Loss: 0.90% from entry price.
- Profit Target: 3.20% from entry price (3.55 : 1 Reward-to-Risk).
- Dynamic Trailing: Once MFE >= 1.50%, stop moves to breakeven (entry price + 0.20% buffer).
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 18 bars holding).
- Universe: Dynamic active universe (77 equities). No hardcoded instruments.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    HypothesisStatus,
    StrategyHorizon,
    MarketMechanism,
)
from src.strategies.base import BaseStrategy
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha5MorningMomentumExpansion(BaseHypothesis, BaseStrategy):
    """
    Alpha 5: Morning Volatility Expansion (MVE).
    Ultra-selective morning-only expansion strategy with asymmetric runway.
    """

    strategy_id = "5_alpha"
    hypothesis_id = "5_alpha"
    name = "5_alpha — Morning Momentum Expansion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "squeeze_atr_ratio": 0.95,        # Pre-expansion ATR ratio vs SMA20(ATR)
            "orb_end_time": "09:45",          # Opening range formation end
            "vol_surge_mult": 2.2,            # Abnormal volume surge threshold
            "trend_ema_span": 50,             # Macro trend alignment filter
            "stop_loss_pct": 0.0090,          # 0.90% initial stop loss
            "take_profit_pct": 0.0320,        # 3.20% take profit (3.55:1 RR)
            "trailing_trigger_pct": 0.0150,   # Activate breakeven stop at +1.50% profit
            "max_holding_bars": 18,           # Max holding duration
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:45",      # Entry window opens
            "entry_end_time": "11:30",        # Strict morning entry cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="5_alpha",
            name="5_alpha — Morning Momentum Expansion",
            category="MORNING_EXPANSION",
            economic_rationale=(
                "Captures institutional morning momentum expansions (09:45-11:30 IST) following volatility "
                "compression. Strict morning entry cutoff provides full multi-hour trend runway to reach asymmetric "
                "3.5:1 profit targets while eliminating late-day flat time-exit friction."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="5_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}
        self._trail_active: Dict[str, bool] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "squeeze_atr_ratio": [0.90, 0.95, 1.00],
            "vol_surge_mult": [1.8, 2.2, 2.6],
            "stop_loss_pct": [0.008, 0.009, 0.010],
            "take_profit_pct": [0.028, 0.032, 0.038],
            "max_holding_bars": [14, 18, 22],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates leak-free directional signals across historical OHLCV data.
        Returns DataFrame with 'signal', 'stop_loss', and 'take_profit' columns.
        """
        if df.empty or len(df) < 60:
            out = df.copy()
            out["signal"] = 0.0
            out["stop_loss"] = 0.0
            out["take_profit"] = 0.0
            return out

        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        out.sort_index(inplace=True)

        squeeze_ratio = float(self.parameters.get("squeeze_atr_ratio", 0.95))
        vol_mult = float(self.parameters.get("vol_surge_mult", 2.2))
        orb_time = str(self.parameters.get("orb_end_time", "09:45"))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0090))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0320))
        trail_trigger = float(self.parameters.get("trailing_trigger_pct", 0.0150))
        max_bars = int(self.parameters.get("max_holding_bars", 18))
        entry_start = str(self.parameters.get("entry_start_time", "09:45"))
        entry_end = str(self.parameters.get("entry_end_time", "11:30"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time indicators (shift(1) to guarantee zero lookahead)
        high_prev = out["high"].shift(1)
        low_prev = out["low"].shift(1)
        close_prev = out["close"].shift(1)
        vol_prev = out["volume"].shift(1)

        # 1. ATR 14 & ATR Moving Average (20)
        tr1 = high_prev - low_prev
        tr2 = (high_prev - close_prev.shift(1)).abs()
        tr3 = (low_prev - close_prev.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_sma20 = atr14.rolling(20).mean()

        # 2. Volume SMA20
        vol_sma20 = vol_prev.rolling(20).mean()

        # 3. Trend EMA50
        ema50 = close_prev.ewm(span=50, adjust=False).mean()

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        dates = pd.Series(out.index.date, index=out.index)
        grouped = out.groupby(dates)

        for date_val, group in grouped:
            group_len = len(group)
            if group_len < 2:
                continue

            pos = 0.0
            entry_px = 0.0
            sl_px = 0.0
            tp_px = 0.0
            bars_held = 0
            traded_today = False
            trail_active = False

            # Compute Opening Range (09:15 to 09:45)
            orb_bars = group[group.index.strftime("%H:%M") <= orb_time]
            if orb_bars.empty:
                continue
            orb_high = float(orb_bars["high"].max())
            orb_low = float(orb_bars["low"].min())

            for i in range(group_len):
                curr_bar = group.iloc[i]
                idx_pos = out.index.get_loc(curr_bar.name)
                t_str = curr_bar.name.strftime("%H:%M")
                c_px = float(curr_bar["close"])
                h_px = float(curr_bar["high"])
                l_px = float(curr_bar["low"])

                # Square-off at or past 15:15 IST
                if t_str >= square_off:
                    if pos != 0.0:
                        pos = 0.0
                        entry_px = 0.0
                        sl_px = 0.0
                        tp_px = 0.0
                        bars_held = 0
                        trail_active = False
                    signals[idx_pos] = 0.0
                    continue

                if pos != 0.0:
                    bars_held += 1

                    # Dynamic Trailing stop to breakeven + buffer once +1.5% profit reached
                    if pos > 0:
                        if h_px >= entry_px * (1.0 + trail_trigger):
                            trail_active = True
                            sl_px = max(sl_px, entry_px * 1.0020)

                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0
                            trail_active = False
                    elif pos < 0:
                        if l_px <= entry_px * (1.0 - trail_trigger):
                            trail_active = True
                            sl_px = min(sl_px, entry_px * 0.9980)

                        if h_px >= sl_px or l_px <= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0
                            trail_active = False

                    signals[idx_pos] = pos
                    stop_losses[idx_pos] = sl_px
                    take_profits[idx_pos] = tp_px

                else:
                    # Strict Morning Entry Window (09:45 to 11:30 IST)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        a14 = atr14.iloc[idx_pos]
                        a_sma = atr_sma20.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]
                        e50 = ema50.iloc[idx_pos]

                        if pd.notna(a14) and pd.notna(a_sma) and pd.notna(v_sma) and a_sma > 0 and v_sma > 0:
                            is_squeezed = (a14 <= a_sma * squeeze_ratio)
                            is_vol_surge = (v_prev >= v_sma * vol_mult)

                            # Bullish Morning Breakout
                            if (c_px > orb_high) and is_squeezed and is_vol_surge and (c_px > e50):
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True
                                trail_active = False

                            # Bearish Morning Breakdown
                            elif (c_px < orb_low) and is_squeezed and is_vol_surge and (c_px < e50):
                                pos = -1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 + sl_pct)
                                tp_px = entry_px * (1.0 - tp_pct)
                                bars_held = 0
                                traded_today = True
                                trail_active = False

                    signals[idx_pos] = pos
                    stop_losses[idx_pos] = sl_px
                    take_profits[idx_pos] = tp_px

        out["signal"] = signals
        out["stop_loss"] = stop_losses
        out["take_profit"] = take_profits
        return out

    def on_bar(self, event: BarEvent) -> List[SignalEvent]:
        """Real-time event-driven bar handler for live/paper engine."""
        sym = event.symbol
        c_px = event.close
        h_px = event.high
        l_px = event.low
        t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)

        # 15:15 Square-off
        if t_str >= "15:15":
            if pos != 0.0:
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
            return []

        # Check holding exit
        if pos != 0.0:
            sl_px = self._entry_price[sym] * (1.0 - 0.0090) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0090)
            tp_px = self._entry_price[sym] * (1.0 + 0.0320) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0320)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 18)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 18)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []