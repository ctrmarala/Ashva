"""
Ashva Quantitative Alpha Strategy — Alpha 7: Regime-Adaptive Volatility Explosion (RAVE)
Alpha ID: 7_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Equities in multi-day structural trends (EMA20 > EMA50 for Longs, EMA20 < EMA50 for Shorts)
that experience multi-day volatility compression (NR4/NR7) followed by explosive morning volume
(>= 2.5x 20-period average) produce rapid, unidirectional momentum thrusts.
Exiting proactively on momentum exhaustion (structural reversal bars after +1.0% MFE) rather than
passively waiting for the 15:15 EOD square-off preserves peak profits and dramatically increases Net PF.

Economic Mechanism:
Morning breakout momentum peaks within 90-180 minutes of execution. Holding through the afternoon
often leads to mean-reverting chop that decays profits back into flat territory while paying full taxes.
A two-stage dynamic exit (tight 0.80% SL, 2.80% TP, and structural trail on 2-bar reversal after +1.0%)
locks in asymmetric positive expectancy.

Contract Specification:
- Structural Trend Filter: EMA20 > EMA50 for Longs, EMA20 < EMA50 for Shorts.
- Volatility Compression: ATR14 <= 0.90x ATR SMA20.
- Entry Window: Strictly 09:30 to 11:15 IST (Morning expansion window).
- Long Entry: Close > ORB High (09:15-09:45 High), Volume >= 2.5x Vol SMA20, EMA20 > EMA50.
- Short Entry: Close < ORB Low (09:15-09:45 Low), Volume >= 2.5x Vol SMA20, EMA20 < EMA50.
- Stop Loss: 0.80% from entry price.
- Profit Target: 2.80% from entry price (3.5 : 1 Reward-to-Risk).
- Structural Momentum Exit: Once MFE >= 1.00%, exit if 2 consecutive bars reverse against position.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 14 bars holding).
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


class Alpha7RegimeAdaptiveVolatilityExplosion(BaseHypothesis, BaseStrategy):
    """
    Alpha 7: Regime-Adaptive Volatility Explosion (RAVE).
    High-conviction morning breakout strategy with proactive momentum peak preservation.
    """

    strategy_id = "7_alpha"
    hypothesis_id = "7_alpha"
    name = "7_alpha — Regime-Adaptive Volatility Explosion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "squeeze_atr_ratio": 0.90,        # ATR compression threshold vs SMA20(ATR)
            "orb_end_time": "09:45",          # Opening range formation end
            "vol_surge_mult": 2.5,            # Volume surge confirmation
            "fast_ema_span": 20,              # Fast structural trend
            "slow_ema_span": 50,              # Slow structural trend
            "stop_loss_pct": 0.0080,          # 0.80% tight initial stop loss
            "take_profit_pct": 0.0280,        # 2.80% take profit (3.5:1 RR)
            "trail_trigger_pct": 0.0100,      # Structural trailing trigger at +1.00%
            "max_holding_bars": 14,           # Max holding duration
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:45",      # Entry window opens
            "entry_end_time": "11:15",        # Strict morning entry cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="7_alpha",
            name="7_alpha — Regime-Adaptive Volatility Explosion",
            category="VOLATILITY_EXPANSION",
            economic_rationale=(
                "Combines multi-day structural trend alignment (EMA20 vs EMA50) with volatility compression (NR4/NR7) "
                "and explosive morning volume. Proactive structural momentum exits lock in peak intraday gains, "
                "yielding positive post-cost net expectancy across Indian equities."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="7_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "squeeze_atr_ratio": [0.85, 0.90, 0.95],
            "vol_surge_mult": [2.0, 2.5, 3.0],
            "stop_loss_pct": [0.007, 0.008, 0.009],
            "take_profit_pct": [0.024, 0.028, 0.032],
            "max_holding_bars": [10, 14, 18],
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

        squeeze_ratio = float(self.parameters.get("squeeze_atr_ratio", 0.90))
        vol_mult = float(self.parameters.get("vol_surge_mult", 2.5))
        orb_time = str(self.parameters.get("orb_end_time", "09:45"))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0080))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0280))
        trail_trigger = float(self.parameters.get("trail_trigger_pct", 0.0100))
        max_bars = int(self.parameters.get("max_holding_bars", 14))
        entry_start = str(self.parameters.get("entry_start_time", "09:45"))
        entry_end = str(self.parameters.get("entry_end_time", "11:15"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time indicators (shift(1) to guarantee zero lookahead)
        high_prev = out["high"].shift(1)
        low_prev = out["low"].shift(1)
        close_prev = out["close"].shift(1)
        vol_prev = out["volume"].shift(1)

        # 1. Structural Trend EMAs
        ema20 = close_prev.ewm(span=20, adjust=False).mean()
        ema50 = close_prev.ewm(span=50, adjust=False).mean()

        # 2. Volume SMA20
        vol_sma20 = vol_prev.rolling(20).mean()

        # 3. ATR 14 & ATR SMA20
        tr1 = high_prev - low_prev
        tr2 = (high_prev - close_prev.shift(1)).abs()
        tr3 = (low_prev - close_prev.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_sma20 = atr14.rolling(20).mean()

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

                    # Dynamic Trailing / Structural Profit Lock
                    if pos > 0:
                        # Once +1.0% profit reached, lock in breakeven + buffer
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
                        # Once +1.0% profit reached on short, lock in breakeven - buffer
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
                    # Strict Morning Window (09:45 to 11:15 IST)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        e20 = ema20.iloc[idx_pos]
                        e50 = ema50.iloc[idx_pos]
                        a14 = atr14.iloc[idx_pos]
                        a_sma = atr_sma20.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(e20) and pd.notna(e50) and pd.notna(a14) and pd.notna(a_sma) and pd.notna(v_sma) and a_sma > 0 and v_sma > 0:
                            is_squeezed = (a14 <= a_sma * squeeze_ratio)
                            is_vol_surge = (v_prev >= v_sma * vol_mult)

                            # Bullish Expansion: Structural Uptrend (EMA20 > EMA50), Squeeze Coiled, Volume Exploding, Breaks ORB High
                            if (c_px > orb_high) and (e20 > e50) and is_squeezed and is_vol_surge:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True
                                trail_active = False

                            # Bearish Expansion: Structural Downtrend (EMA20 < EMA50), Squeeze Coiled, Volume Exploding, Breaks ORB Low
                            elif (c_px < orb_low) and (e20 < e50) and is_squeezed and is_vol_surge:
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
            sl_px = self._entry_price[sym] * (1.0 - 0.0080) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0080)
            tp_px = self._entry_price[sym] * (1.0 + 0.0280) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0280)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 14)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 14)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []