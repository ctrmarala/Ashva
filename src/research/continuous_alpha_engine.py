"""
Ashva Continuous Autonomous Alpha Discovery & Portfolio Engineering Engine (v3 Chronological)
Ultra-fast, fully vectorized quantitative search and fine-tuning engine with exact
bar-by-bar chronological intrabar SL/TP matching BacktestEngine and IndianCostModel.
"""

import sys
import os
import time
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.research.experiment_ledger import ResearchExperimentLedger, get_current_git_sha


class FastDayBarArray:
    """Pre-extracted vectorized daily multi-bar structure with 2D intraday bar arrays for 1 symbol."""
    __slots__ = (
        'symbol', 'dates', 'n_days', 'b0_open', 'b0_high', 'b0_low', 'b0_close', 'b0_vol',
        'b1_open', 'intraday_highs', 'intraday_lows', 'intraday_closes',
        'prev_closes', 'prev_highs', 'prev_lows',
        'daily_atrs', 'tod_vols', 'bar1_10d_max_vols', 'is_nr', 'is_in', 'is_din', 'is_2g', 'is_2r',
        'is_oos'
    )

    def __init__(self, symbol: str, df: pd.DataFrame, oos_days: int = 120):
        self.symbol = symbol
        timestamps = pd.to_datetime(df.index)
        dates = timestamps.date
        times = timestamps.time
        df["time_str"] = [t.strftime("%H:%M") for t in times]
        df["bar_date"] = dates
        df["bar_time"] = times

        daily_summary = df.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
            day_open=("open", "first"),
            day_vol=("volume", "sum")
        )

        daily_range = daily_summary["day_high"] - daily_summary["day_low"]
        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)

        tr1 = daily_range
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        # NR3..NR7 series
        nr_series = {}
        for k in range(3, 8):
            is_nr_s = pd.Series(False, index=daily_summary.index)
            for d_i in range(k, len(daily_summary)):
                if daily_range.iloc[d_i - 1] < daily_range.iloc[d_i - k:d_i - 1].min():
                    is_nr_s.iloc[d_i] = True
            nr_series[k] = is_nr_s

        h = daily_summary["day_high"]
        l = daily_summary["day_low"]
        is_in_s = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
        is_din_s = is_in_s & (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))
        is_2g_s = (daily_summary["day_close"].shift(1) > daily_summary["day_open"].shift(1)) & (daily_summary["day_close"].shift(2) > daily_summary["day_open"].shift(2))
        is_2r_s = (daily_summary["day_close"].shift(1) < daily_summary["day_open"].shift(1)) & (daily_summary["day_close"].shift(2) < daily_summary["day_open"].shift(2))

        # Time of day rolling mean volume
        tod_vol_s = df.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(df["volume"])
        df["tod_mean_vol"] = tod_vol_s

        # Bar 0 (09:15) and Bar 1 (09:30)
        b0_df = df[df["time_str"] == "09:15"]
        b1_df = df[df["time_str"] == "09:30"]

        # Bar 1 10d max volume
        b1_series = pd.Series(b0_df["volume"].values, index=b0_df["bar_date"])
        max_b1_10d_s = b1_series.rolling(10, min_periods=5).max().shift(1)

        # Common valid day index
        valid_dates = b0_df["bar_date"].values
        b1_dates_map = {d: o for d, o in zip(b1_df["bar_date"].values, b1_df["open"].values)}

        # Filter days where both bar 0 and bar 1 exist
        final_dates = [d for d in valid_dates if d in b1_dates_map and d in daily_summary.index]
        self.n_days = len(final_dates)
        self.dates = np.array(final_dates)

        b0_sub = b0_df.set_index("bar_date").loc[final_dates]
        self.b0_open = b0_sub["open"].values.astype(np.float64)
        self.b0_high = b0_sub["high"].values.astype(np.float64)
        self.b0_low = b0_sub["low"].values.astype(np.float64)
        self.b0_close = b0_sub["close"].values.astype(np.float64)
        self.b0_vol = b0_sub["volume"].values.astype(np.float64)
        self.tod_vols = b0_sub["tod_mean_vol"].values.astype(np.float64)
        self.b1_open = np.array([b1_dates_map[d] for d in final_dates], dtype=np.float64)

        # Build 2D intraday arrays for bars 1..24 (09:30 to 15:15)
        # 24 slots per day
        self.intraday_highs = np.zeros((self.n_days, 25), dtype=np.float64)
        self.intraday_lows = np.zeros((self.n_days, 25), dtype=np.float64)
        self.intraday_closes = np.zeros((self.n_days, 25), dtype=np.float64)

        time_order = sorted(df["time_str"].unique())
        time_to_idx = {t: i for i, t in enumerate(time_order)}

        df_indexed = df.set_index(["bar_date", "time_str"])
        for day_i, d in enumerate(final_dates):
            for t_str, slot_i in time_to_idx.items():
                if (d, t_str) in df_indexed.index:
                    row = df_indexed.loc[(d, t_str)]
                    self.intraday_highs[day_i, slot_i] = row["high"]
                    self.intraday_lows[day_i, slot_i] = row["low"]
                    self.intraday_closes[day_i, slot_i] = row["close"]
                else:
                    # fill with previous close or day close
                    if slot_i > 0:
                        self.intraday_highs[day_i, slot_i] = self.intraday_highs[day_i, slot_i - 1]
                        self.intraday_lows[day_i, slot_i] = self.intraday_lows[day_i, slot_i - 1]
                        self.intraday_closes[day_i, slot_i] = self.intraday_closes[day_i, slot_i - 1]

        self.prev_closes = prev_close.loc[final_dates].values.astype(np.float64)
        self.prev_highs = prev_high.loc[final_dates].values.astype(np.float64)
        self.prev_lows = prev_low.loc[final_dates].values.astype(np.float64)
        self.daily_atrs = daily_atr14.loc[final_dates].values.astype(np.float64)
        self.bar1_10d_max_vols = max_b1_10d_s.loc[final_dates].values.astype(np.float64)

        self.is_nr = {k: nr_series[k].loc[final_dates].values.astype(bool) for k in nr_series}
        self.is_in = is_in_s.loc[final_dates].values.astype(bool)
        self.is_din = is_din_s.loc[final_dates].values.astype(bool)
        self.is_2g = is_2g_s.loc[final_dates].values.astype(bool)
        self.is_2r = is_2r_s.loc[final_dates].values.astype(bool)

        # OOS flags
        max_dt = pd.to_datetime(final_dates[-1])
        oos_cutoff = (max_dt - pd.Timedelta(days=oos_days)).date()
        self.is_oos = np.array([d >= oos_cutoff for d in final_dates], dtype=bool)


class ContinuousAlphaEngine:
    """
    Vectorized high-throughput discovery and fine-tuning engine.
    """

    def __init__(self, symbols: Optional[List[str]] = None, lookback_days: int = 540, oos_days: int = 120):
        self.lake = DataLake(read_only=True)
        self.cost_model = IndianCostModel(default_slippage_bps=3.0)
        self.symbols = symbols or [
            "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
            "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
            "BAJFINANCE", "MARUTI", "SUNPHARMA"
        ]
        self.lookback_days = lookback_days
        self.oos_days = oos_days
        self.cache: Dict[str, FastDayBarArray] = {}
        self._build_fast_cache()

    def _build_fast_cache(self):
        print(f"[*] Building high-speed vectorized bar arrays for {len(self.symbols)} assets...", end="", flush=True)
        t0 = time.time()
        for sym in self.symbols:
            df = self.lake.load_bars(sym, "15m", max_lookback_days=self.lookback_days)
            if df.empty or len(df) < 50:
                continue
            self.cache[sym] = FastDayBarArray(sym, df, oos_days=self.oos_days)
        print(f" [DONE in {time.time() - t0:.2f}s, cached {len(self.cache)} assets]")

    def evaluate_hypothesis_fast(
        self,
        mechanism_type: str,
        params: Dict[str, Any],
        capital_per_trade: float = 125000.0,
    ) -> Dict[str, Any]:
        """
        Evaluates a hypothesis configuration with exact chronological intrabar bar 1..24 evaluation.
        """
        min_gap = params.get("min_gap", 0.0030)
        max_gap = params.get("max_gap", 0.0150)
        min_rvol = params.get("min_rvol", 1.50)
        min_body = params.get("min_body", 0.60)
        target_rr = params.get("target_rr", 1.50)
        nr_k = params.get("nr_k", 4)
        vol_mult = params.get("vol_mult", 1.20)

        all_net_pnls = []
        all_oos_pnls = []
        pos_assets = 0
        total_trades = 0
        total_wins = 0
        total_oos_trades = 0
        total_oos_wins = 0

        for sym, bar in self.cache.items():
            sym_trades = 0
            sym_net = 0.0

            gap_pct = (bar.b0_open - bar.prev_closes) / np.maximum(1e-4, bar.prev_closes)
            abs_gap = np.abs(gap_pct)
            rvol = bar.b0_vol / np.maximum(1.0, bar.tod_vols)
            bar_range = bar.b0_high - bar.b0_low
            body_size = np.abs(bar.b0_close - bar.b0_open)
            body_ratio = np.where(bar_range > 0, body_size / bar_range, 0.0)

            # Signal conditions
            if mechanism_type == "NR_GAP_BREAKOUT":
                nr_flags = bar.is_nr.get(nr_k, np.zeros(bar.n_days, dtype=bool))
                base_cond = nr_flags & (abs_gap >= min_gap) & (abs_gap <= max_gap) & (rvol >= min_rvol) & (body_ratio >= min_body)
                long_cond = base_cond & (gap_pct > 0) & (bar.b0_close > bar.prev_highs) & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (gap_pct < 0) & (bar.b0_close < bar.prev_lows) & (bar.b0_close < bar.b0_open)

            elif mechanism_type == "INSIDE_DAY_GAP":
                base_cond = bar.is_in & (abs_gap >= min_gap) & (abs_gap <= max_gap) & (rvol >= min_rvol) & (body_ratio >= min_body)
                long_cond = base_cond & (gap_pct > 0) & (bar.b0_close > bar.prev_highs) & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (gap_pct < 0) & (bar.b0_close < bar.prev_lows) & (bar.b0_close < bar.b0_open)

            elif mechanism_type == "TWO_DAY_TREND_GAP":
                base_cond = (abs_gap >= min_gap) & (abs_gap <= max_gap) & (rvol >= min_rvol) & (body_ratio >= min_body)
                long_cond = base_cond & (gap_pct > 0) & bar.is_2g & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (gap_pct < 0) & bar.is_2r & (bar.b0_close < bar.b0_open)

            elif mechanism_type == "OUTLIER_VOLUME_DRIVE":
                vol_cond = ~np.isnan(bar.bar1_10d_max_vols) & (bar.b0_vol >= (vol_mult * bar.bar1_10d_max_vols))
                base_cond = vol_cond & (abs_gap >= min_gap) & (abs_gap <= max_gap) & (body_ratio >= min_body)
                long_cond = base_cond & (gap_pct > 0) & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (gap_pct < 0) & (bar.b0_close < bar.b0_open)

            elif mechanism_type == "GAP_MARUBOZU":
                base_cond = (abs_gap >= min_gap) & (abs_gap <= max_gap) & (rvol >= min_rvol) & (body_ratio >= 0.70)
                long_cond = base_cond & (gap_pct > 0) & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (gap_pct < 0) & (bar.b0_close < bar.b0_open)

            elif mechanism_type == "DOUBLE_INSIDE_EXPANSION":
                base_cond = bar.is_din & (rvol >= min_rvol)
                long_cond = base_cond & (bar.b0_close > bar.prev_highs) & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (bar.b0_close < bar.prev_lows) & (bar.b0_close < bar.b0_open)

            elif mechanism_type == "GAP_VOLUME_SHOCK_DRIFT":
                # Alpha 37 mechanism generalized
                base_cond = (abs_gap >= min_gap) & (abs_gap <= max_gap) & (rvol >= min_rvol) & (body_ratio >= min_body)
                long_cond = base_cond & (gap_pct > 0) & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (gap_pct < 0) & (bar.b0_close < bar.b0_open)

            else:
                continue

            long_indices = np.where(long_cond)[0]
            short_indices = np.where(short_cond)[0]

            # Process Longs with exact chronological intraday evaluation (bars 1..24)
            for idx in long_indices:
                entry_p = bar.b1_open[idx]
                if entry_p <= 0: continue
                sl_p = bar.b0_low[idx]
                risk = max(entry_p * 0.002, entry_p - sl_p)
                tp_p = entry_p + (target_rr * risk)

                exit_p = bar.intraday_closes[idx, -1] # Default EOD 15:15
                # Check bars 1 to 24 chronologically
                for bar_k in range(1, 25):
                    h_k = bar.intraday_highs[idx, bar_k]
                    l_k = bar.intraday_lows[idx, bar_k]
                    if l_k <= sl_p:
                        exit_p = sl_p
                        break
                    elif h_k >= tp_p:
                        exit_p = tp_p
                        break

                qty = int(capital_per_trade / entry_p)
                if qty <= 0: continue

                costs = self.cost_model.calculate_trade_costs(
                    buy_price=entry_p,
                    sell_price=exit_p,
                    quantity=qty,
                    segment=Segment.EQUITY_INTRADAY,
                    is_stop_loss=(exit_p <= sl_p),
                )
                net = costs.net_pnl

                all_net_pnls.append(net)
                sym_trades += 1
                sym_net += net
                total_trades += 1
                if net > 0: total_wins += 1

                if bar.is_oos[idx]:
                    all_oos_pnls.append(net)
                    total_oos_trades += 1
                    if net > 0: total_oos_wins += 1

            # Process Shorts with exact chronological intraday evaluation (bars 1..24)
            for idx in short_indices:
                entry_p = bar.b1_open[idx]
                if entry_p <= 0: continue
                sl_p = bar.b0_high[idx]
                risk = max(entry_p * 0.002, sl_p - entry_p)
                tp_p = entry_p - (target_rr * risk)

                exit_p = bar.intraday_closes[idx, -1]
                for bar_k in range(1, 25):
                    h_k = bar.intraday_highs[idx, bar_k]
                    l_k = bar.intraday_lows[idx, bar_k]
                    if h_k >= sl_p:
                        exit_p = sl_p
                        break
                    elif l_k <= tp_p:
                        exit_p = tp_p
                        break

                qty = int(capital_per_trade / entry_p)
                if qty <= 0: continue

                costs = self.cost_model.calculate_trade_costs(
                    buy_price=exit_p,
                    sell_price=entry_p,
                    quantity=qty,
                    segment=Segment.EQUITY_INTRADAY,
                    is_stop_loss=(exit_p >= sl_p),
                )
                net = costs.net_pnl

                all_net_pnls.append(net)
                sym_trades += 1
                sym_net += net
                total_trades += 1
                if net > 0: total_wins += 1

                if bar.is_oos[idx]:
                    all_oos_pnls.append(net)
                    total_oos_trades += 1
                    if net > 0: total_oos_wins += 1

            if sym_trades > 0 and sym_net > 0:
                pos_assets += 1

        if total_trades == 0:
            return {
                "total_trades": 0, "net_pnl": 0.0, "net_pf": 0.0, "win_rate": 0.0,
                "sharpe": 0.0, "oos_trades": 0, "oos_net_pnl": 0.0, "oos_win_rate": 0.0,
                "oos_sharpe": 0.0, "positive_assets": 0, "avg_net_trade_pct": 0.0
            }

        total_net_pnl = sum(all_net_pnls)
        wins = [p for p in all_net_pnls if p > 0]
        losses = [p for p in all_net_pnls if p <= 0]
        win_rate = (len(wins) / total_trades) * 100.0
        net_wins = sum(wins)
        net_losses = abs(sum(losses))
        net_pf = (net_wins / net_losses) if net_losses > 0 else (99.0 if net_wins > 0 else 0.0)

        rets = [p / capital_per_trade for p in all_net_pnls]
        std = np.std(rets)
        sharpe = (np.mean(rets) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0

        # OOS
        if total_oos_trades > 0:
            oos_net_pnl = sum(all_oos_pnls)
            oos_wr = (total_oos_wins / total_oos_trades) * 100.0
            oos_rets = [p / capital_per_trade for p in all_oos_pnls]
            s_oos = np.std(oos_rets)
            oos_sharpe = (np.mean(oos_rets) / s_oos * np.sqrt(252 * 6.25 / 9.6)) if s_oos > 0 else 0.0
        else:
            oos_net_pnl = 0.0
            oos_wr = 0.0
            oos_sharpe = 0.0

        avg_net_trade_pct = (total_net_pnl / (total_trades * capital_per_trade)) * 100.0

        return {
            "mechanism_type": mechanism_type,
            "params": params,
            "total_trades": total_trades,
            "net_pnl": round(total_net_pnl, 2),
            "net_pf": round(net_pf, 2),
            "win_rate": round(win_rate, 1),
            "sharpe": round(sharpe, 2),
            "oos_trades": total_oos_trades,
            "oos_net_pnl": round(oos_net_pnl, 2),
            "oos_win_rate": round(oos_wr, 1),
            "oos_sharpe": round(oos_sharpe, 2),
            "positive_assets": pos_assets,
            "avg_net_trade_pct": round(avg_net_trade_pct, 3),
        }
