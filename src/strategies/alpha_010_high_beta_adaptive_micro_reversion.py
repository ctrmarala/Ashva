"""
Ashva Quantitative Alpha Strategy — Alpha 10: High-Beta Adaptive Micro-Mean Reversion (HAMR)
Alpha ID: 10_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
In high-volatility Indian equities (ATR% >= 1.8% per day), statistical exhaustion at 2.5 sigma
Bollinger extremes with RSI capitulation (< 22 or > 78) mean-reverts 0.90% within 2 to 4 bars (60-120 minutes)
with high statistical regularity (>= 60% win rate).
By sizing the target (0.95%) within the stock's natural intrabar reach and cutting holding time to max 4 bars,
the strategy eliminates the massive friction drag caused by afternoon time-exit stalls.

Economic Mechanism:
Intraday liquidity exhaustion at statistical extremes is an instantaneous order-flow phenomenon.
When liquidity providers absorb retail capitulation, the snapback occurs rapidly.
Capturing the first 0.95% impulse with a tight 0.45% stop and 4-bar time stop locks in quick, high-win-rate
turnover with positive post-tax edge.

Contract Specification:
- High Volatility Universe Filter: 14-bar ATR / Close >= 0.0040 (0.40% per bar).
- Extreme Band Filter: Price breaches 2.5 sigma Bollinger Bands (20 period).
- RSI Exhaustion: RSI(14) <= 22.0 (Long) or RSI(14) >= 78.0 (Short).
- Volume Climax: Volume >= 2.0x 20-period Volume SMA.
- Strict Morning Window: 09:30 to 12:00 IST.
- Stop Loss: 0.45% from entry price.
- Profit Target: 0.95% from entry price (2.1 : 1 Reward-to-Risk).
- Fast Time Exit: Max 4 holding bars (2 hours on 30m).
- Time Horizon: Intraday (15:15 IST mandatory square-off).
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


class Alpha10HighBetaAdaptiveMicroReversion(BaseHypothesis, BaseStrategy):
    """
    Alpha 10: High-Beta Adaptive Micro-Mean Reversion (HAMR).
    High-win-rate rapid snapback strategy capturing high-beta statistical extremes.
    """

    strategy_id = "10_alpha"
    hypothesis_id = "10_alpha"
    name = "10_alpha — High-Beta Adaptive Micro-Mean Reversion"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "bb_period": 20,                  # Bollinger Band lookback
            "bb_std": 2.5,                    # 2.5 sigma deviation multiplier
            "rsi_period": 14,                 # RSI lookback
            "rsi_oversold": 22.0,             # Oversold threshold
            "rsi_overbought": 78.0,           # Overbought threshold
            "vol_climax_mult": 2.0,           # Volume climax multiplier
            "min_atr_pct": 0.0040,            # Min 0.40% ATR per bar (volatility filter)
            "stop_loss_pct": 0.0045,          # 0.45% tight stop loss
            "take_profit_pct": 0.0095,        # 0.95% take profit (2.1:1 RR)
            "max_holding_bars": 4,            # Fast 4-bar holding duration (2 hours)
            "timeframe": "30m",               # Default research timeframe
            "entry_start_time": "09:30",      # Morning window opens
            "entry_end_time": "12:00",        # Morning cutoff
            "square_off_time": "15:15",       # Intraday square-off
        }
        merged_params = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="10_alpha",
            name="10_alpha — High-Beta Adaptive Micro-Mean Reversion",
            category="STATISTICAL_MEAN_REVERSION",
            economic_rationale=(
                "Captures rapid (4-bar) 0.95% snapbacks in volatile equities following 2.5 sigma Bollinger "
                "exhaustion and RSI capitulation. Fast execution eliminates intraday time-decay stalls and "
                "maximizes post-cost net expectancy."
            ),
            target_instruments=merged_params.get("target_instruments", []),
            timeframe=merged_params.get("timeframe", "30m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged_params)
        BaseStrategy.__init__(self, strategy_id="10_alpha", parameters=merged_params)

        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """Returns the parameter search space for sensitivity and robustness testing."""
        return {
            "bb_std": [2.3, 2.5, 2.7],
            "rsi_oversold": [20.0, 22.0, 25.0],
            "vol_climax_mult": [1.8, 2.0, 2.4],
            "stop_loss_pct": [0.0040, 0.0045, 0.0050],
            "take_profit_pct": [0.0085, 0.0095, 0.0110],
            "max_holding_bars": [3, 4, 6],
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

        bb_period = int(self.parameters.get("bb_period", 20))
        bb_std = float(self.parameters.get("bb_std", 2.5))
        rsi_period = int(self.parameters.get("rsi_period", 14))
        rsi_os = float(self.parameters.get("rsi_oversold", 22.0))
        rsi_ob = float(self.parameters.get("rsi_overbought", 78.0))
        vol_mult = float(self.parameters.get("vol_climax_mult", 2.0))
        min_atr = float(self.parameters.get("min_atr_pct", 0.0040))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0095))
        max_bars = int(self.parameters.get("max_holding_bars", 4))
        entry_start = str(self.parameters.get("entry_start_time", "09:30"))
        entry_end = str(self.parameters.get("entry_end_time", "12:00"))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        # Point-in-time indicators (shift(1) to guarantee zero lookahead)
        open_prev = out["open"].shift(1)
        high_prev = out["high"].shift(1)
        low_prev = out["low"].shift(1)
        close_prev = out["close"].shift(1)
        vol_prev = out["volume"].shift(1)

        # 1. Bollinger Bands (2.5 sigma)
        bb_mid = close_prev.rolling(bb_period).mean()
        bb_sigma = close_prev.rolling(bb_period).std()
        bb_upper = bb_mid + (bb_sigma * bb_std)
        bb_lower = bb_mid - (bb_sigma * bb_std)

        # 2. RSI (14)
        delta = close_prev.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # 3. ATR 14 (Volatility filter)
        tr1 = high_prev - low_prev
        tr2 = (high_prev - close_prev.shift(1)).abs()
        tr3 = (low_prev - close_prev.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_pct = atr14 / (close_prev + 1e-10)

        # 4. Volume SMA20
        vol_sma20 = vol_prev.rolling(20).mean()

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
                    # Rapid exit conditions (SL, TP, or 4-bar time stop)
                    if pos > 0:
                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0
                            entry_px = 0.0
                            sl_px = 0.0
                            tp_px = 0.0
                            bars_held = 0
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
                    # Strict Morning Window Entry (09:30 to 12:00 IST)
                    if (not traded_today) and (entry_start <= t_str <= entry_end):
                        b_up = bb_upper.iloc[idx_pos]
                        b_low = bb_lower.iloc[idx_pos]
                        r_val = rsi.iloc[idx_pos]
                        a_pct = atr_pct.iloc[idx_pos]
                        v_prev = vol_prev.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]
                        c_p = close_prev.iloc[idx_pos]
                        o_p = open_prev.iloc[idx_pos]
                        l_p = low_prev.iloc[idx_pos]
                        h_p = high_prev.iloc[idx_pos]

                        if pd.notna(b_up) and pd.notna(b_low) and pd.notna(r_val) and pd.notna(a_pct) and pd.notna(v_sma) and v_sma > 0:
                            is_vol_climax = (v_prev >= v_sma * vol_mult)
                            is_high_beta = (a_pct >= min_atr)

                            # Bullish Micro-Reversion: High-Beta, Low breached lower band, RSI <= 22, Bullish reversal candle, Climax volume
                            if is_high_beta and (l_p <= b_low) and (r_val <= rsi_os) and (c_p > o_p) and is_vol_climax:
                                pos = 1.0
                                entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0
                                traded_today = True

                            # Bearish Micro-Reversion: High-Beta, High breached upper band, RSI >= 78, Bearish reversal candle, Climax volume
                            elif is_high_beta and (h_p >= b_up) and (r_val >= rsi_ob) and (c_p < o_p) and is_vol_climax:
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
            sl_px = self._entry_price[sym] * (1.0 - 0.0045) if pos > 0 else self._entry_price[sym] * (1.0 + 0.0045)
            tp_px = self._entry_price[sym] * (1.0 + 0.0095) if pos > 0 else self._entry_price[sym] * (1.0 - 0.0095)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            if (pos > 0 and (l_px <= sl_px or h_px >= tp_px or self._bars_held[sym] >= 4)) or \
               (pos < 0 and (h_px >= sl_px or l_px <= tp_px or self._bars_held[sym] >= 4)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]

        return []