"""
Ashva 1-Minute Intrabar Execution Simulator (Microstructure Engine)
Provides tick-accurate trade path simulation by replaying 1-minute historical bars
to eliminate intrabar sequence ambiguity (determining whether SL, TP, or Ratchet triggered first).
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.data.data_lake import DataLake


@dataclass
class IntrabarTradeResult:
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str          # "TAKE_PROFIT", "STOP_LOSS", "BREAK_EVEN_SL", "STEP_RATCHET_SL", "TIME_EXIT"
    mfe_price: float          # Maximum Favorable Excursion price
    mae_price: float          # Maximum Adverse Excursion price
    mfe_pct: float            # MFE as % from entry
    mae_pct: float            # MAE as % from entry
    bars_held_1m: int
    trailing_level_reached: float  # In terms of R (e.g. 1.0, 1.5, 2.0)


class IntrabarSimulator:
    """
    High-performance 1-minute historical trade path replay engine.
    """

    def __init__(self, data_lake: Optional[DataLake] = None):
        self.lake = data_lake or DataLake(read_only=True)
        self._cache_1m: Dict[str, pd.DataFrame] = {}

    def get_1m_bars(self, symbol: str) -> pd.DataFrame:
        """Retrieves and caches 1-minute bars for a given symbol."""
        sym_clean = symbol.upper()
        if sym_clean not in self._cache_1m:
            df = self.lake.load_bars(sym_clean, "1m")
            if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp").sort_index()
            self._cache_1m[sym_clean] = df
        return self._cache_1m[sym_clean]

    def simulate_trade(
        self,
        symbol: str,
        entry_time: pd.Timestamp,
        entry_price: float,
        side: str,
        stop_loss: float,
        take_profit: float,
        trailing_mode: str = "NONE",  # "NONE", "BREAK_EVEN", "STEP_RATCHET"
        max_exit_time: Optional[pd.Timestamp] = None,
    ) -> IntrabarTradeResult:
        """
        Replays exact 1-minute path between entry_time and max_exit_time.
        """
        df_1m = self.get_1m_bars(symbol)
        if df_1m.empty:
            # Fallback when 1m data is missing
            return self._fallback_estimate(entry_time, entry_price, side, stop_loss, take_profit)

        # Slice 1m bars from entry_time
        trade_slice = df_1m.loc[entry_time:]
        if max_exit_time is not None:
            trade_slice = trade_slice.loc[:max_exit_time]

        if trade_slice.empty:
            return self._fallback_estimate(entry_time, entry_price, side, stop_loss, take_profit)

        is_buy = (side.upper() == "BUY" or side.upper() == "LONG")
        initial_risk = abs(entry_price - stop_loss)
        current_sl = stop_loss
        current_tp = take_profit
        highest_r = 0.0
        mfe_price = entry_price
        mae_price = entry_price

        bars_count = 0
        exit_time = trade_slice.index[-1]
        exit_price = trade_slice.iloc[-1]["close"]
        exit_reason = "TIME_EXIT"

        for idx, bar in trade_slice.iterrows():
            bars_count += 1
            bar_high = bar["high"]
            bar_low = bar["low"]
            bar_close = bar["close"]
            bar_open = bar["open"]

            if is_buy:
                # 1. Check Open Gap below existing SL
                if bar_open <= current_sl:
                    exit_time = idx
                    exit_price = min(bar_open, current_sl)  # Penalize with open gap slippage
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    break

                # 2. Check Low against EXISTING Stop Loss (BEFORE any MFE ratchet)
                if bar_low <= current_sl:
                    exit_time = idx
                    exit_price = current_sl
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    mae_price = min(mae_price, bar_low)
                    break

                # 3. Check High against Take Profit
                if current_tp > 0 and bar_high >= current_tp:
                    exit_time = idx
                    exit_price = current_tp
                    exit_reason = "TAKE_PROFIT"
                    mfe_price = max(mfe_price, current_tp)
                    break

                # 4. Mandatory 15:15 EOD Square-Off
                if idx.hour >= 15 and idx.minute >= 15:
                    exit_time = idx
                    exit_price = bar_close
                    exit_reason = "TIME_EXIT"
                    break

                # 5. Position Survived Bar: Update MFE and ratchet stop for SUBSEQUENT bars
                mfe_price = max(mfe_price, bar_high)
                mae_price = min(mae_price, bar_low)
                current_r = (bar_high - entry_price) / max(1e-4, initial_risk)
                highest_r = max(highest_r, current_r)

                if trailing_mode == "BREAK_EVEN" and highest_r >= 1.0:
                    current_sl = max(current_sl, entry_price)
                elif trailing_mode == "STEP_RATCHET":
                    if highest_r >= 2.0:
                        current_sl = max(current_sl, entry_price + 1.0 * initial_risk)
                    elif highest_r >= 1.5:
                        current_sl = max(current_sl, entry_price + 0.5 * initial_risk)
                    elif highest_r >= 1.0:
                        current_sl = max(current_sl, entry_price)

            else:
                # SHORT / SELL
                # 1. Check Open Gap above existing SL
                if bar_open >= current_sl:
                    exit_time = idx
                    exit_price = max(bar_open, current_sl)
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    break

                # 2. Check High against EXISTING Stop Loss
                if bar_high >= current_sl:
                    exit_time = idx
                    exit_price = current_sl
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    mae_price = max(mae_price, bar_high)
                    break

                # 3. Check Low against Take Profit
                if current_tp > 0 and bar_low <= current_tp:
                    exit_time = idx
                    exit_price = current_tp
                    exit_reason = "TAKE_PROFIT"
                    mfe_price = min(mfe_price, current_tp)
                    break

                # 4. Mandatory 15:15 EOD Square-Off
                if idx.hour >= 15 and idx.minute >= 15:
                    exit_time = idx
                    exit_price = bar_close
                    exit_reason = "TIME_EXIT"
                    break

                # 5. Position Survived Bar: Update MFE and ratchet stop for SUBSEQUENT bars
                mfe_price = min(mfe_price, bar_low)
                mae_price = max(mae_price, bar_high)
                current_r = (entry_price - bar_low) / max(1e-4, initial_risk)
                highest_r = max(highest_r, current_r)

                if trailing_mode == "BREAK_EVEN" and highest_r >= 1.0:
                    current_sl = min(current_sl, entry_price)
                elif trailing_mode == "STEP_RATCHET":
                    if highest_r >= 2.0:
                        current_sl = min(current_sl, entry_price - 1.0 * initial_risk)
                    elif highest_r >= 1.5:
                        current_sl = min(current_sl, entry_price - 0.5 * initial_risk)
                    elif highest_r >= 1.0:
                        current_sl = min(current_sl, entry_price)

        mfe_pct = ((mfe_price - entry_price) / entry_price * 100.0) if is_buy else ((entry_price - mfe_price) / entry_price * 100.0)
        mae_pct = ((mae_price - entry_price) / entry_price * 100.0) if is_buy else ((entry_price - mae_price) / entry_price * 100.0)

        return IntrabarTradeResult(
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            mfe_price=mfe_price,
            mae_price=mae_price,
            mfe_pct=round(mfe_pct, 2),
            mae_pct=round(mae_pct, 2),
            bars_held_1m=bars_count,
            trailing_level_reached=round(highest_r, 2),
        )

    def _fallback_estimate(
        self, entry_time: pd.Timestamp, entry_price: float, side: str, stop_loss: float, take_profit: float
    ) -> IntrabarTradeResult:
        return IntrabarTradeResult(
            exit_time=entry_time,
            exit_price=stop_loss,
            exit_reason="STOP_LOSS",
            mfe_price=entry_price,
            mae_price=stop_loss,
            mfe_pct=0.0,
            mae_pct=-abs((stop_loss - entry_price) / entry_price * 100.0),
            bars_held_1m=1,
            trailing_level_reached=0.0,
        )
