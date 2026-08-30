"""
Ashva Quantitative Alpha Strategy — Alpha 6: Cross-Sectional Relative Strength VWAP Trend
Alpha ID: 6_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Institutional algorithmic execution in Indian equities predominantly benchmarks against Intraday VWAP.
Equities trading with strong relative strength above their session VWAP that experience minor pullbacks
into the VWAP anchor zone represent high-probability institutional accumulation points.
Entering at VWAP support with a 3.0:1 reward-to-risk ratio delivers high win-rates and positive net expectancy.

Economic Mechanism:
VWAP execution algorithms (institutional buy programs) actively defend the VWAP benchmark to avoid negative slippage.
Buying at VWAP support during macro uptrends (Price > EMA50) and shorting at VWAP resistance during macro downtrends
(Price < EMA50) aligns retail capital with institutional flow while strictly controlling downside risk (0.85% SL).

Contract Specification:
- Session VWAP: Dynamic intraday cumulative volume-weighted average price reset daily at 09:15 IST.
- Long Entry: 09:45 <= Time <= 13:00, Price > EMA50, Low touches VWAP (within +0.4%), Close bounces > VWAP, Volume >= 1.8x Vol SMA20.
- Short Entry: 09:45 <= Time <= 13:00, Price < EMA50, High touches VWAP (within -0.4%), Close rejects < VWAP, Volume >= 1.8x Vol SMA20.
- Stop Loss: 0.85% from entry price (strictly enforced).
- Profit Target: 2.55% from entry price (3.0 : 1 Reward-to-Risk).
- Breakeven Trail: Once unrealized gain >= 1.20%, stop moves to entry price + 0.15% buffer.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 16 bars holding).
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


class Alpha6RelativeStrengthVWAPTrend(BaseHypothesis, BaseStrategy):
    """
    Alpha 6: Cross-Sectional Relative Strength VWAP Trend.
    Captures institutional accumulation and distribution at intraday VWAP support/resistance.
    """

    strategy_id = "6_alpha"
    hypothesis_id = "6_alpha"
    name = "6_alpha — Relative Strength VWAP Trend"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "vwap_touch_tolerance": 0.004,    # 0.4% tolerance around VWAP for pullback touch
            "vol_surge_mult": 1.8,            # Volume confirmation threshold
            "trend_ema_span": 50,             # Macro trend alignment filter
            "stop_loss_pct": 0.0085,          # 0.85% stop loss
            "take_profit_pct": 0.0255,        # 2.55% take profit (3.0:1 RR)
            "trailing_trigger_pct": 0.0120,   # Breakeven trigger at +1.20%
            "max_holding_bars": 16,           # Max holding duration
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:45",      # Entry window opens
            "entry_end_time": "13:00",        # Entry window closes
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="6_alpha",
            name="6_alpha — Relative Strength VWAP Trend",
            category="VWAP_MOMENTUM",
            economic_rationale=(
                "Captures institutional accumulation/distribution at session VWAP defense points. "
                "Buying VWAP pullbacks in macro uptrends and shorting VWAP rallies in macro downtrends with "
                "a 3:1 RR provides positive post-friction expectancy across Indian equities."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="6_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "vwap_touch_tolerance": [0.003, 0.004, 0.005],
            "vol_surge_mult": [1.5, 1.8, 2.2],
            "stop_loss_pct": [0.0075, 0.0085, 0.0100],
            "take_profit_pct": [0.0225, 0.0255, 0.0300],
            "max_holding_bars": [12, 16, 20],
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

        vwap_tol = float(self.parameters.get("vwap_touch_tolerance", 0.004))
        vol_mult = float(self.parameters.get("vol_surge_mult", 1.8))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0085))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0255))
        trail_trigger = float(self.parameters.get("trailing_trigger_pct", 0.0120))
        max_bars = int(self.parameters.get("max_holding_bars", 16))
        entry_start = str(self.parameters.get("entry_start_time", "09:45"))
        entry_end = str(self.parameters.get("entry_end_time", "13:00"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time indicators (shift(1) to guarantee zero lookahead)
        open_prev = out["open"].shift(1)
        high_prev = out["high"].shift(1)
        low_prev = out["low"].shift(1)
        close_prev = out["close"].shift(1)
        vol_prev = out["volume"].shift(1)

        # 1. Macro Trend EMA50
        ema50 = close_prev.ewm(span=50, adjust=False).mean()

        # 2. Volume SMA20
        vol_sma20 = vol_prev.rolling(20).mean()

        # 3. Session VWAP computed point-in-time strictly on historical closed bars
        typical_price_prev = (high_prev + low_prev + close_prev) / 3.0
        pv_prev = typical_price_prev * vol_prev

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

            # Session VWAP cumulative tracking for this day
            cum_pv = 0.0
            cum_vol = 0.0

            for i in range(group_len):
                curr_bar = group.iloc[i]
                idx_pos = out.index.get_loc(curr_bar.name)
                t_str = curr_bar.name.strftime("%H:%M")
                c_px = float(curr_bar["close"])
                h_px = float(curr_bar["high"])
                l_px = float(curr_bar["low"])

                # Update cumulative session VWAP strictly from previous closed bar
                p_val = pv_prev.iloc[idx_pos]
                v_val = vol_prev.iloc[idx_pos]
                if pd.notna(p_val) and pd.notna(v_val) and v_val > 0:
                    cum_pv += p_val
                    cum_vol += v_val

                session_vwap = (cum_pv / cum_vol) if cum_vol > 0 else c_px

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

                    # Dynamic Trailing stop to breakeven + buffer once +1.2% profit reached
                    if pos > 0:
                        if h_px >= entry_px * (1.0 + trail_trigger):
                            trail_active = True
                            sl_px = max(sl_px, entry_px * 1.0015)

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
                            sl_px = min(sl_px, entry_px * 0.9985)

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
                    # New Entry check within window (Max 1 trade per stock per day)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        e50 = ema50.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]
                        c_p = close_prev.iloc[idx_pos]
                        o_p = open_prev.iloc[idx_pos]
                        l_p = low_prev.iloc[idx_pos]
                        h_p = high_prev.iloc[idx_pos]

                        if pd.notna(e50) and pd.notna(v_sma) and v_sma > 0 and session_vwap > 0:
                            is_vol_surge = (v_prev >= v_sma * vol_mult)

                            # Distances from VWAP
                            vwap_dist_low = (l_p - session_vwap) / session_vwap
                            vwap_dist_high = (h_p - session_vwap) / session_vwap

                            # Bullish VWAP Bounce: Macro trend up (c > ema50), pulled back near VWAP, bounced above VWAP
                            if (c_p > e50) and (c_p >= session_vwap) and (-vwap_tol <= vwap_dist_low <= vwap_tol * 1.5) and (c_p > o_p) and is_vol_surge:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True
                                trail_active = False

                            # Bearish VWAP Rejection: Macro trend down (c < ema50), rallied near VWAP, rejected below VWAP
                            elif (c_p < e50) and (c_p <= session_vwap) and (-vwap_tol * 1.5 <= vwap_dist_high <= vwap_tol) and (c_p < o_p) and is_vol_surge:
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
            sl_px = self._entry_price[sym] * (1.0 - 0.0085) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0085)
            tp_px = self._entry_price[sym] * (1.0 + 0.0255) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0255)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 16)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 16)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []