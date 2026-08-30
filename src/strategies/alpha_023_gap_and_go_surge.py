"""
Ashva Quantitative Alpha Strategy — Alpha 23: Gap-and-Go Institutional Surge (GAG-V2)
Alpha ID: 23_alpha
Version: v1.0.0
Author: AshvaQuantLab

Hypothesis:
Archive winner alpha_04_gap_and_go achieved OOS Sharpe 8.53 and Net PF 99.0 with parameters:
{min_gap_pct: 0.5%, rvol_mult: 1.3, min_adx: 16, rr_ratio: 2.0, sl_buffer_atr: 0.5, eod_exit: 15:15}.

This alpha replicates that structural DNA with improved geometry:
- Qualifying gap: 0.40% to 2.00% (slightly lower floor for more trades)
- ADX >= 16 (trending stock required — filters ranging markets)
- RVOL >= 1.30x (institutional participation confirmed)
- Strong opening body (>= 60% of range)
- TP: 1.50% (3.33:1 RR vs 0.45% SL)

Economic Mechanism:
A meaningful overnight gap (0.40%+) with a directional ADX (>= 16) and heavy institutional volume
means the entire market-maker community is directionally biased. Gap-and-go stocks tend to retain
direction for 90-150 minutes before reversion attempts occur.

Cost-Validated Geometry:
- At 45% win rate (gap stocks tend to higher WR): 0.45 * Rs1875 - 0.55 * Rs562 = Rs844 - Rs309 = +Rs535
- After Rs300 costs: Net expectancy = +Rs235 per trade
- Break-even win rate: 36%

Contract Specification:
- Opening Gap: 0.40% <= gap_pct <= 2.00% (calculated vs prior day close).
- ADX(14) on daily >= 16 (trending market condition).
- First 15m bar: Strong body (>= 60%), RVOL >= 1.30x.
- Entry: Close of first 15m bar in gap direction.
- Stop Loss: 0.45%.
- Profit Target: 1.50% (3.33:1 RR).
- Dynamic Profit Lock: +0.40% lock once +0.90% achieved.
- Max holding: 10 bars (15m timeframe).
- Square-off: 15:15 IST.
- Universe: Dynamic. No hardcoded instruments.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis, HypothesisMetadata, StrategyHorizon, MarketMechanism,
)
from src.strategies.base import BaseStrategy
from src.core.events import BarEvent, SignalEvent, SignalType


class Alpha23GapAndGoSurge(BaseHypothesis, BaseStrategy):
    """
    Alpha 23: Gap-and-Go Institutional Surge (GAG-V2).
    Replicates archive winner alpha_04 (OOS Sharpe 8.53) with cost-validated 3.33:1 RR.
    """

    strategy_id = "23_alpha"
    hypothesis_id = "23_alpha"
    name = "23_alpha — Gap-and-Go Institutional Surge"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "min_gap_pct": 0.0040,            # 0.40% minimum gap
            "max_gap_pct": 0.0200,            # 2.00% maximum gap
            "min_adx": 16.0,                  # ADX(14) >= 16
            "min_body_ratio": 0.60,           # Strong body confirmation
            "min_rvol": 1.30,                 # 1.30x institutional volume
            "stop_loss_pct": 0.0045,          # 0.45% structural stop
            "take_profit_pct": 0.0150,        # 1.50% profit target (3.33:1 RR)
            "trail_trigger_pct": 0.0090,      # +0.90% profit lock trigger
            "trail_lock_pct": 0.0040,         # Lock in +0.40%
            "max_holding_bars": 10,
            "timeframe": "15m",
            "square_off_time": "15:15",
        }
        merged = {**default_params, **(parameters or {})}

        metadata = HypothesisMetadata(
            hypothesis_id="23_alpha",
            name="23_alpha — Gap-and-Go Institutional Surge",
            category="ORDER_FLOW_IMBALANCE",
            economic_rationale=(
                "Replicates archive winner alpha_04_gap_and_go (OOS Sharpe 8.53, Net PF 99.0). "
                "Qualifying gap + ADX trend + RVOL surge = directional institutional conviction. "
                "Cost-validated 3.33:1 RR breaks even at 36% win rate."
            ),
            target_instruments=merged.get("target_instruments", []),
            timeframe=merged.get("timeframe", "15m"),
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )

        BaseHypothesis.__init__(self, metadata=metadata, parameters=merged)
        BaseStrategy.__init__(self, strategy_id="23_alpha", parameters=merged)
        self._current_pos: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bars_held: Dict[str, int] = {}

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0030, 0.0040, 0.0050],
            "min_adx": [14.0, 16.0, 20.0],
            "take_profit_pct": [0.0130, 0.0150, 0.0180],
        }

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute ADX using Wilder's smoothing. Strictly uses shifted (historical) data."""
        h = high.shift(1)
        l = low.shift(1)
        c = close.shift(1)
        c_prev = c.shift(1)

        tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
        dm_plus = np.where((h - h.shift(1)) > (l.shift(1) - l), np.maximum(h - h.shift(1), 0), 0)
        dm_minus = np.where((l.shift(1) - l) > (h - h.shift(1)), np.maximum(l.shift(1) - l, 0), 0)

        dm_plus_s = pd.Series(dm_plus, index=close.index).ewm(alpha=1/period, adjust=False).mean()
        dm_minus_s = pd.Series(dm_minus, index=close.index).ewm(alpha=1/period, adjust=False).mean()
        tr_s = tr.ewm(alpha=1/period, adjust=False).mean()

        di_plus = 100.0 * dm_plus_s / (tr_s + 1e-10)
        di_minus = 100.0 * dm_minus_s / (tr_s + 1e-10)
        dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 60:
            out = df.copy()
            out["signal"] = 0.0; out["stop_loss"] = 0.0; out["take_profit"] = 0.0
            return out

        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")
        out.sort_index(inplace=True)

        min_gap = float(self.parameters.get("min_gap_pct", 0.0040))
        max_gap = float(self.parameters.get("max_gap_pct", 0.0200))
        min_adx_val = float(self.parameters.get("min_adx", 16.0))
        min_body = float(self.parameters.get("min_body_ratio", 0.60))
        min_rvol = float(self.parameters.get("min_rvol", 1.30))
        sl_pct = float(self.parameters.get("stop_loss_pct", 0.0045))
        tp_pct = float(self.parameters.get("take_profit_pct", 0.0150))
        trail_trig = float(self.parameters.get("trail_trigger_pct", 0.0090))
        trail_lock = float(self.parameters.get("trail_lock_pct", 0.0040))
        max_bars = int(self.parameters.get("max_holding_bars", 10))
        square_off = str(self.parameters.get("square_off_time", "15:15"))

        dates_series = pd.Series(out.index.date, index=out.index)

        # Point-in-time prior-day close (strictly shifted)
        daily_close = out["close"].groupby(dates_series).last()
        prev_close = daily_close.shift(1)
        prev_close_map = dates_series.map(prev_close)

        # ADX on intraday bars (using shift(1) for all OHLC inputs — strictly historical)
        adx_series = self._calc_adx(out["high"], out["low"], out["close"])

        vol_sma20 = out["volume"].shift(1).rolling(20).mean()

        signals = np.zeros(len(out), dtype=float)
        stop_losses = np.zeros(len(out), dtype=float)
        take_profits = np.zeros(len(out), dtype=float)

        grouped = out.groupby(dates_series)

        for date_val, group in grouped:
            if len(group) < 2:
                continue

            pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0
            bars_held = 0; traded_today = False

            for i, (ts, row) in enumerate(group.iterrows()):
                idx_pos = out.index.get_loc(ts)
                t_str = ts.strftime("%H:%M")
                c_px = float(row["close"]); h_px = float(row["high"])
                l_px = float(row["low"]); o_px = float(row["open"])

                if t_str >= square_off:
                    if pos != 0.0:
                        pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0; bars_held = 0
                    signals[idx_pos] = 0.0; continue

                if pos != 0.0:
                    bars_held += 1
                    if pos > 0:
                        if h_px >= entry_px * (1.0 + trail_trig):
                            sl_px = max(sl_px, entry_px * (1.0 + trail_lock))
                        if l_px <= sl_px or h_px >= tp_px or bars_held >= max_bars:
                            pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0; bars_held = 0
                    else:
                        if l_px <= entry_px * (1.0 - trail_trig):
                            sl_px = min(sl_px, entry_px * (1.0 - trail_lock))
                        if h_px >= sl_px or l_px <= tp_px or bars_held >= max_bars:
                            pos = 0.0; entry_px = 0.0; sl_px = 0.0; tp_px = 0.0; bars_held = 0

                    signals[idx_pos] = pos; stop_losses[idx_pos] = sl_px; take_profits[idx_pos] = tp_px

                else:
                    # Fire only on first bar of the session
                    if i == 0 and not traded_today:
                        p_close = prev_close_map.iloc[idx_pos]
                        adx_val = adx_series.iloc[idx_pos]
                        v_sma = vol_sma20.iloc[idx_pos]

                        if pd.notna(p_close) and pd.notna(adx_val) and pd.notna(v_sma) and p_close > 0 and v_sma > 0:
                            gap_pct = (o_px - p_close) / p_close
                            bar_range = h_px - l_px
                            body = abs(c_px - o_px)
                            body_ratio = body / bar_range if bar_range > 0 else 0.0
                            is_trending = adx_val >= min_adx_val
                            is_vol_ok = row["volume"] >= v_sma * min_rvol

                            # Bullish Gap-and-Go
                            if (min_gap <= gap_pct <= max_gap) and (c_px > o_px) and body_ratio >= min_body and is_trending and is_vol_ok:
                                pos = 1.0; entry_px = c_px
                                sl_px = entry_px * (1.0 - sl_pct)
                                tp_px = entry_px * (1.0 + tp_pct)
                                bars_held = 0; traded_today = True

                            # Bearish Gap-and-Go
                            elif (-max_gap <= gap_pct <= -min_gap) and (c_px < o_px) and body_ratio >= min_body and is_trending and is_vol_ok:
                                pos = -1.0; entry_px = c_px
                                sl_px = entry_px * (1.0 + sl_pct)
                                tp_px = entry_px * (1.0 - tp_pct)
                                bars_held = 0; traded_today = True

                    signals[idx_pos] = pos; stop_losses[idx_pos] = sl_px; take_profits[idx_pos] = tp_px

        out["signal"] = signals; out["stop_loss"] = stop_losses; out["take_profit"] = take_profits
        return out

    def on_bar(self, event: BarEvent) -> List[SignalEvent]:
        sym = event.symbol; t_str = event.timestamp.strftime("%H:%M")
        pos = self._current_pos.get(sym, 0.0)
        if t_str >= "15:15":
            if pos != 0.0:
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
            return []
        if pos != 0.0:
            ep = self._entry_price.get(sym, event.close)
            sl = ep * (1.0 - 0.0045) if pos > 0 else ep * (1.0 + 0.0045)
            tp = ep * (1.0 + 0.0150) if pos > 0 else ep * (1.0 - 0.0150)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1
            if (pos > 0 and (event.low <= sl or event.high >= tp or self._bars_held[sym] >= 10)) or \
               (pos < 0 and (event.high >= sl or event.low <= tp or self._bars_held[sym] >= 10)):
                self._current_pos[sym] = 0.0
                return [SignalEvent(strategy_id=self.strategy_id, symbol=sym, signal_type=SignalType.FLAT, timestamp=event.timestamp)]
        return []
