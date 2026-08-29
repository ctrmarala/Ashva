"""
Ashva 1-Minute Intrabar Execution Simulator (Microstructure Engine)
Performs discrete 1-minute historical bar path replay to model intrabar order flow.
NOTE: 1-minute OHLC bar replay is NOT tick data.
When High and Low in a single 1-minute bar create simultaneous SL and TP/MFE ambiguity,
the engine applies deterministic ambiguity handling (WORST_CASE / conservative SL-first by default).
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from src.data.data_lake import DataLake


class IntrabarAmbiguityMode(str, Enum):
    WORST_CASE = "WORST_CASE"  # Conservative: Adverse level (Stop Loss) hit first
    BEST_CASE = "BEST_CASE"    # Optimistic: Favorable level (Take Profit) hit first


@dataclass
class IntrabarTradeResult:
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str          # "TAKE_PROFIT", "STOP_LOSS", "BREAK_EVEN_SL", "STEP_RATCHET_SL", "TIME_EXIT", "INTRABAR_DATA_UNAVAILABLE"
    mfe_price: float          # Maximum Favorable Excursion price
    mae_price: float          # Maximum Adverse Excursion price
    mfe_pct: float            # MFE as % from entry
    mae_pct: float            # MAE as % from entry
    bars_held_1m: int
    trailing_level_reached: float  # In terms of R (e.g. 1.0, 1.5, 2.0)
    is_intrabar_qualified: bool = True  # False if 1m data was missing


class IntrabarSimulator:
    """
    High-performance 1-minute historical trade path replay engine with
    deterministic ambiguity resolution and explicit missing data handling.
    """

    def __init__(self, data_lake: Optional[DataLake] = None, default_mode: IntrabarAmbiguityMode = IntrabarAmbiguityMode.WORST_CASE):
        self.lake = data_lake or DataLake(read_only=True)
        self.default_mode = default_mode
        self._cache_1m: Dict[str, pd.DataFrame] = {}

    def get_1m_bars(self, symbol: str) -> pd.DataFrame:
        """Retrieves and caches 1-minute bars for a given symbol."""
        sym_clean = symbol.upper()
        if sym_clean not in self._cache_1m:
            df = self.lake.load_bars(sym_clean, "1m")
            if not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp").sort_index()
                # Precompute for fast numpy vectorization
                df['is_eod'] = (df.index.hour >= 15) & (df.index.minute >= 15)
                df['_timestamp_vals'] = df.index.values
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
        ambiguity_mode: Optional[IntrabarAmbiguityMode] = None,
    ) -> IntrabarTradeResult:
        """
        Replays exact 1-minute path between entry_time and max_exit_time.
        Applies deterministic ambiguity handling when SL and TP are both touched in the same 1m bar.
        """
        mode = ambiguity_mode or self.default_mode
        df_1m = self.get_1m_bars(symbol)
        
        # Explicit Missing Data Handling: DO NOT invent a synthetic stop loss
        if df_1m.empty:
            return self._missing_data_result(entry_time, entry_price, stop_loss)

        # Slice 1m bars from entry_time using fast numpy binary search
        timestamps = df_1m['_timestamp_vals'].values
        entry_np = entry_time.to_numpy()
        
        idx_start = np.searchsorted(timestamps, entry_np)
        if max_exit_time is not None:
            max_np = max_exit_time.to_numpy()
            idx_end = np.searchsorted(timestamps, max_np, side='right')
        else:
            idx_end = len(timestamps)
            
        if idx_start >= idx_end or idx_start >= len(timestamps):
            return self._missing_data_result(entry_time, entry_price, stop_loss)
            
        opens = df_1m['open'].values[idx_start:idx_end]
        highs = df_1m['high'].values[idx_start:idx_end]
        lows = df_1m['low'].values[idx_start:idx_end]
        closes = df_1m['close'].values[idx_start:idx_end]
        is_eod_arr = df_1m['is_eod'].values[idx_start:idx_end]
        ts_slice = timestamps[idx_start:idx_end]

        is_buy = (side.upper() in ("BUY", "LONG"))
        initial_risk = max(1e-4, abs(entry_price - stop_loss))
        current_sl = stop_loss
        current_tp = take_profit
        highest_r = 0.0
        mfe_price = entry_price
        mae_price = entry_price

        bars_count = 0
        exit_time = pd.Timestamp(ts_slice[-1])
        exit_price = closes[-1]
        exit_reason = "TIME_EXIT"

        for i in range(len(ts_slice)):
            bars_count += 1
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bar_is_eod = is_eod_arr[i]
            idx = pd.Timestamp(ts_slice[i])

            if is_buy:
                # 1. Check Open Gap below existing SL
                if current_sl > 0 and bar_open <= current_sl:
                    exit_time = idx
                    exit_price = min(bar_open, current_sl)  # Penalize with open gap slippage
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    mae_price = min(mae_price, bar_low)
                    break

                # 2. Check Ambiguity: Both SL and TP touched in same 1-minute bar
                sl_hit = (current_sl > 0 and bar_low <= current_sl)
                tp_hit = (current_tp > 0 and bar_high >= current_tp)

                if sl_hit and tp_hit:
                    exit_time = idx
                    if mode == IntrabarAmbiguityMode.WORST_CASE:
                        exit_price = current_sl
                        exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                        mae_price = min(mae_price, bar_low)
                    else:
                        exit_price = current_tp
                        exit_reason = "TAKE_PROFIT"
                        mfe_price = max(mfe_price, current_tp)
                    break

                # 3. Single-barrier hits
                if sl_hit:
                    exit_time = idx
                    exit_price = current_sl
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    mae_price = min(mae_price, bar_low)
                    break

                if tp_hit:
                    exit_time = idx
                    exit_price = current_tp
                    exit_reason = "TAKE_PROFIT"
                    mfe_price = max(mfe_price, current_tp)
                    break

                # 4. Mandatory 15:15 EOD Square-Off
                if bar_is_eod:
                    exit_time = idx
                    exit_price = bar_close
                    exit_reason = "TIME_EXIT"
                    mfe_price = max(mfe_price, bar_high)
                    mae_price = min(mae_price, bar_low)
                    break

                # 5. Position Survived Bar: Update MFE and ratchet stop for SUBSEQUENT bars
                mfe_price = max(mfe_price, bar_high)
                mae_price = min(mae_price, bar_low)
                current_r = (bar_high - entry_price) / initial_risk
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
                if current_sl > 0 and bar_open >= current_sl:
                    exit_time = idx
                    exit_price = max(bar_open, current_sl)
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    mae_price = max(mae_price, bar_high)
                    break

                # 2. Check Ambiguity: Both SL and TP touched in same 1-minute bar
                sl_hit = (current_sl > 0 and bar_high >= current_sl)
                tp_hit = (current_tp > 0 and bar_low <= current_tp)

                if sl_hit and tp_hit:
                    exit_time = idx
                    if mode == IntrabarAmbiguityMode.WORST_CASE:
                        exit_price = current_sl
                        exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                        mae_price = max(mae_price, bar_high)
                    else:
                        exit_price = current_tp
                        exit_reason = "TAKE_PROFIT"
                        mfe_price = min(mfe_price, current_tp)
                    break

                # 3. Single-barrier hits
                if sl_hit:
                    exit_time = idx
                    exit_price = current_sl
                    exit_reason = "STOP_LOSS" if current_sl == stop_loss else "STEP_RATCHET_SL"
                    mae_price = max(mae_price, bar_high)
                    break

                if tp_hit:
                    exit_time = idx
                    exit_price = current_tp
                    exit_reason = "TAKE_PROFIT"
                    mfe_price = min(mfe_price, current_tp)
                    break

                # 4. Mandatory 15:15 EOD Square-Off
                if bar_is_eod:
                    exit_time = idx
                    exit_price = bar_close
                    exit_reason = "TIME_EXIT"
                    mfe_price = min(mfe_price, bar_low)
                    mae_price = max(mae_price, bar_high)
                    break

                # 5. Position Survived Bar: Update MFE and ratchet stop for SUBSEQUENT bars
                mfe_price = min(mfe_price, bar_low)
                mae_price = max(mae_price, bar_high)
                current_r = (entry_price - bar_low) / initial_risk
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
            is_intrabar_qualified=True,
        )

    def _missing_data_result(
        self, entry_time: pd.Timestamp, entry_price: float, stop_loss: float
    ) -> IntrabarTradeResult:
        """Explicitly handles missing 1m data without inventing fake stop losses."""
        return IntrabarTradeResult(
            exit_time=entry_time,
            exit_price=entry_price,
            exit_reason="INTRABAR_DATA_UNAVAILABLE",
            mfe_price=entry_price,
            mae_price=entry_price,
            mfe_pct=0.0,
            mae_pct=0.0,
            bars_held_1m=0,
            trailing_level_reached=0.0,
            is_intrabar_qualified=False,
        )
