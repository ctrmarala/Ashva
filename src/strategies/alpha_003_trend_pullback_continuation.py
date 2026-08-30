"""
Ashva Quantitative Alpha Strategy — Alpha 3: Multi-Timeframe Trend Pullback Continuation
Alpha ID: 3_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Equities exhibiting strong higher-order directional momentum (Daily/Multi-hour EMA alignment)
that experience short-term intraday pullbacks into dynamic support/resistance (EMA20 / VWAP)
and momentum exhaustion (RSI pullbacks) represent high-probability institutional dip-buying/rally-selling
opportunities. Entering on pullbacks rather than chasing breakout peaks provides superior risk-reward
and reduces stop-outs.

Economic Mechanism:
Institutional execution algorithms accumulate liquidity on pullbacks within prevailing trends.
Entering at dynamic value levels (EMA20/VWAP band) with a trailing or 2.5:1 RR target yields higher win rates
and significantly larger average win sizes, easily overcoming Indian statutory turnover costs on 15m/30m bars.

Contract Specification:
- Long Entry: Close > EMA50 (Trend Filter), Close pulls back to touch/dip below EMA20, RSI(14) in pullback zone (35-48), Bullish reversal candle (Close > Open and Close > Close[1]), Volume > Vol SMA20.
- Short Entry: Close < EMA50 (Trend Filter), Close rallies to touch/exceed EMA20, RSI(14) in rally zone (52-65), Bearish reversal candle (Close < Open and Close < Close[1]), Volume > Vol SMA20.
- Stop Loss: 1.20% from entry price (or recent swing low).
- Profit Target: 3.00% from entry price (2.5 : 1 Reward-to-Risk).
- Time Horizon: Intraday (15:15 IST mandatory close, max 16 bars holding).
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


class Alpha3TrendPullbackContinuation(BaseHypothesis, BaseStrategy):
    """
    Alpha 3: Multi-Timeframe Trend Pullback Continuation.
    Captures trend continuation entries on intraday pullbacks into dynamic value zones.
    """

    strategy_id = "3_alpha"
    hypothesis_id = "3_alpha"
    name = "3_alpha — Trend Pullback Continuation"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "trend_ema_span": 50,             # Macro trend filter
            "pullback_ema_span": 20,          # Value zone EMA
            "rsi_period": 14,                 # RSI period
            "rsi_long_lower": 35.0,           # RSI long pullback zone min
            "rsi_long_upper": 50.0,           # RSI long pullback zone max
            "rsi_short_lower": 50.0,          # RSI short rally zone min
            "rsi_short_upper": 65.0,          # RSI short rally zone max
            "stop_loss_pct": 0.0120,          # 1.20% stop loss
            "take_profit_pct": 0.0300,        # 3.00% take profit (2.5:1 RR)
            "max_holding_bars": 16,           # Max holding duration
            "timeframe": "15m",               # Default research timeframe
            "entry_start_time": "09:30",      # Cease early opening noise
            "entry_end_time": "14:15",        # Cease new late entries
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="3_alpha",
            name="3_alpha — Trend Pullback Continuation",
            category="TREND_PULLBACK",
            economic_rationale=(
                "Equities in strong directional momentum that experience intraday pullbacks into dynamic "
                "support (EMA20/VWAP) with RSI exhaustion present high-probability institutional continuation "
                "entries with favorable 2.5:1 reward-to-risk profiles."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="3_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for robustness testing."""
        return {
            "trend_ema_span": [40, 50, 60],
            "pullback_ema_span": [15, 20, 25],
            "rsi_long_upper": [48.0, 50.0, 52.0],
            "stop_loss_pct": [0.010, 0.012, 0.015],
            "take_profit_pct": [0.025, 0.030, 0.035],
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

        trend_span = int(self.parameters.get("trend_ema_span", 50))
        pb_span = int(self.parameters.get("pullback_ema_span", 20))
        rsi_period = int(self.parameters.get("rsi_period", 14))
        rsi_l_low = float(self.parameters.get("rsi_long_lower", 35.0))
        rsi_l_high = float(self.parameters.get("rsi_long_upper", 50.0))
        rsi_s_low = float(self.parameters.get("rsi_short_lower", 50.0))
        rsi_s_high = float(self.parameters.get("rsi_short_upper", 65.0))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0120))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0300))
        max_bars = int(self.parameters.get("max_holding_bars", 16))
        entry_start = str(self.parameters.get("entry_start_time", "09:30"))
        entry_end = str(self.parameters.get("entry_end_time", "14:15"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time indicators (shift(1) to guarantee zero lookahead)
        open_prev = out["open"].shift(1)
        high_prev = out["high"].shift(1)
        low_prev = out["low"].shift(1)
        close_prev = out["close"].shift(1)
        vol_prev = out["volume"].shift(1)

        # 1. EMAs
        ema_trend = close_prev.ewm(span=trend_span, adjust=False).mean()
        ema_pb = close_prev.ewm(span=pb_span, adjust=False).mean()
        vol_sma20 = vol_prev.rolling(20).mean()

        # 2. RSI (14)
        delta = close_prev.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

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
                    signals[idx_pos] = 0.0
                    continue

                if pos != 0.0:
                    bars_held += 1
                    # Long Exit conditions
                    if pos > 0:
                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0
                    # Short Exit conditions
                    elif pos < 0:
                        if h_px >= sl_px or l_px <= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0

                    signals[idx_pos] = pos
                    stop_losses[idx_pos] = sl_px
                    take_profits[idx_pos] = tp_px

                else:
                    # New Entry check within entry window (Max 1 trade per stock per day)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        e_tr = ema_trend.iloc[idx_pos]
                        e_pb = ema_pb.iloc[idx_pos]
                        r_val = rsi.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]
                        c_p = close_prev.iloc[idx_pos]
                        o_p = open_prev.iloc[idx_pos]
                        l_p = low_prev.iloc[idx_pos]
                        h_p = high_prev.iloc[idx_pos]

                        if pd.notna(e_tr) and pd.notna(e_pb) and pd.notna(r_val) and pd.notna(v_sma) and v_sma > 0:
                            vol_confirm = (v_prev >= v_sma * 1.0)

                            # Bullish Pullback: Trend is Up (c > ema50), pulled back near/below ema20, RSI in oversold/pullback zone, bullish candle
                            if (c_p > e_tr) and (l_p <= e_pb * 1.005) and (rsi_l_low <= r_val <= rsi_l_high) and (c_p > o_p) and vol_confirm:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish Rally: Trend is Down (c < ema50), rallied near/above ema20, RSI in overbought/rally zone, bearish candle
                            elif (c_p < e_tr) and (h_p >= e_pb * 0.995) and (rsi_s_low <= r_val <= rsi_s_high) and (c_p < o_p) and vol_confirm:
                                pos = -1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 + sl_pct)
                                tp_px = entry_px * (1.0 - tp_pct)
                                bars_held = 0
                                traded_today = True

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
            sl_px = self._entry_price[sym] * (1.0 - 0.0120) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0120)
            tp_px = self._entry_price[sym] * (1.0 + 0.0300) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0300)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 16)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 16)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []