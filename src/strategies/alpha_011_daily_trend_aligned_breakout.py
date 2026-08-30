"""
Ashva Quantitative Alpha Strategy — Alpha 11: Daily-Trend-Aligned Institutional Breakout (DAMIB)
Alpha ID: 11_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Equities breaking out of their prior-day extremes (Prior Day High for Longs, Prior Day Low for Shorts)
in alignment with the multi-day macro trend (75-bar 30m EMA) and backed by institutional opening volume
(>= 2.5x 20-period SMA) exhibit explosive, unidirectional multi-hour continuation.

Economic Mechanism:
Prior-day high/low levels represent major liquidity zones where resting institutional orders reside.
When high morning volume punches through these levels in the direction of the dominant multi-day trend,
it triggers cascading stop runs and institutional chase momentum, delivering clean 3:1+ reward-to-risk
runs with minimal adverse excursion.

Contract Specification:
- Multi-Day Macro Trend: 75-bar EMA on 30m (approx 6-day trend proxy).
- Prior Day High/Low Breakout: Point-in-time calculation from preceding trading session.
- Volume Confirmation: Volume >= 2.5x 20-period Volume SMA.
- Strict Morning Window: 09:45 to 11:15 IST.
- Stop Loss: 0.60% from entry price.
- Profit Target: 2.00% from entry price (3.33 : 1 Reward-to-Risk).
- Dynamic Trailing Lock: Once +0.90% profit reached, move stop loss to +0.25% in profit.
- Time Horizon: Intraday (15:15 IST mandatory square-off, max 10 holding bars).
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


class Alpha11DailyTrendAlignedInstitutionalBreakout(BaseHypothesis, BaseStrategy):
    """
    Alpha 11: Daily-Trend-Aligned Institutional Breakout (DAMIB).
    High-reward-to-risk prior-day breakout strategy aligned with multi-day macro trend.
    """

    strategy_id = "11_alpha"
    hypothesis_id = "11_alpha"
    name = "11_alpha — Daily-Trend-Aligned Institutional Breakout"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "macro_ema_span": 75,             # 75-bar EMA on 30m (~6 trading days)
            "vol_surge_mult": 2.5,            # 2.5x volume surge
            "stop_loss_pct": 0.0060,          # 0.60% tight stop loss
            "take_profit_pct": 0.0200,        # 2.00% profit target (3.33:1 RR)
            "trail_trigger_pct": 0.0090,      # Dynamic profit lock trigger at +0.90%
            "trail_lock_pct": 0.0025,         # Lock in +0.25% once trigger hit
            "max_holding_bars": 10,           # Max holding duration (5 hours on 30m)
            "timeframe": "30m",               # Default research timeframe
            "entry_start_time": "09:45",      # Morning window opens
            "entry_end_time": "11:15",        # Morning cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="11_alpha",
            name="11_alpha — Daily-Trend-Aligned Institutional Breakout",
            category="MOMENTUM_BREAKOUT",
            economic_rationale=(
                "Captures explosive morning liquidity breakouts through prior-day high/low extremes aligned with "
                "the 6-day macro trend. High 3.33:1 reward-to-risk and dynamic breakeven locks comfortably clear "
                "statutory turnover friction."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "30m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="11_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "macro_ema_span": [50, 75, 100],
            "vol_surge_mult": [2.0, 2.5, 3.0],
            "stop_loss_pct": [0.0050, 0.0060, 0.0070],
            "take_profit_pct": [0.0180, 0.0200, 0.0240],
            "max_holding_bars": [8, 10, 12],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates leak-free directional signals across historical OHLCV data.
        Returns DataFrame with 'signal', 'stop_loss', and 'take_profit' columns.
        """
        if df.empty or len(df) < 80:
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

        macro_span = int(self.parameters.get("macro_ema_span", 75))
        vol_mult = float(self.parameters.get("vol_surge_mult", 2.5))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0060))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0200))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0090))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0025))
        max_bars = int(self.parameters.get("max_holding_bars", 10))
        entry_start = str(self.parameters.get("entry_start_time", "09:45"))
        entry_end = str(self.parameters.get("entry_end_time", "11:15"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time indicators (shift(1) to guarantee zero lookahead)
        open_prev = out["open"].shift(1)
        high_prev = out["high"].shift(1)
        low_prev = out["low"].shift(1)
        close_prev = out["close"].shift(1)
        vol_prev = out["volume"].shift(1)

        # 1. Macro Trend EMA
        macro_ema = close_prev.ewm(span=macro_span, adjust=False).mean()

        # 2. Volume SMA20
        vol_sma20 = vol_prev.rolling(20).mean()

        # 3. Calculate Prior Day High and Prior Day Low strictly from completed prior days
        daily_highs = out["high"].groupby(out.index.date).max().shift(1)
        daily_lows = out["low"].groupby(out.index.date).min().shift(1)

        dates_series = pd.Series(out.index.date, index=out.index)
        prior_day_high_s = dates_series.map(daily_highs)
        prior_day_low_s = dates_series.map(daily_lows)

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        grouped = out.groupby(dates_series)

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

                    # Dynamic Trailing Profit Lock
                    if pos > 0:
                        if h_px >= entry_px * (1.0 + trail_trig):
                            trail_active = True
                            sl_px = max(sl_px, entry_px * (1.0 + trail_lock))

                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0
                            trail_active = False
                    elif pos < 0:
                        if l_px <= entry_px * (1.0 - trail_trig):
                            trail_active = True
                            sl_px = min(sl_px, entry_px * (1.0 - trail_lock))

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
                    # Morning Breakout Window Entry (09:45 to 11:15 IST)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        pd_high = prior_day_high_s.iloc[idx_pos]
                        pd_low = prior_day_low_s.iloc[idx_pos]
                        m_ema = macro_ema.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(pd_high) and pd.notna(pd_low) and pd.notna(m_ema) and pd.notna(v_sma) and v_sma > 0:
                            is_vol_surge = (v_prev >= v_sma * vol_mult)

                            # Bullish Prior Day Breakout: Macro trend up (Close > Macro EMA), breaks above Prior Day High, volume surging
                            if (c_px > pd_high) and (c_px > m_ema) and is_vol_surge:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True
                                trail_active = False

                            # Bearish Prior Day Breakdown: Macro trend down (Close < Macro EMA), breaks below Prior Day Low, volume surging
                            elif (c_px < pd_low) and (c_px < m_ema) and is_vol_surge:
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
            sl_px = self._entry_price[sym] * (1.0 - 0.0060) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0060)
            tp_px = self._entry_price[sym] * (1.0 + 0.0200) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0200)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 10)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 10)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []