"""
Ashva Quantitative Alpha 14 Signal & Trade Feature Decomposition Engine
Analyzes winning vs losing trades and favorable (W1) vs choppy (W3/W4) regimes
using the existing signal features computed by Alpha 14 at the 09:30 AM decision point.

Zero new indicators, zero parameter optimization, zero curve fitting.
Answers:
1. Which existing signal components (gap size, body ratio, RVOL, gap/ATR, normalized ATR, wick ratio, side)
   statistically separate winning continuations from failed fades?
2. What happened microstructurally in Windows 3 & 4 that caused trades to fail?
3. Did failed trades reverse immediately (immediate gap fade) or suffer EOD decay?
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.alpha_14_gap_momentum_drift import Alpha14GapMomentumDrift
from scripts.run_hypothesis_lab import DEFAULT_UNIVERSE
from scripts.scan_nifty_50_universe import NIFTY_50_UNIVERSE


def run_feature_decomposition(
    universe_type: str = "nifty14",
    lookback_days: int = 540,
    n_windows: int = 5,
    capital_per_asset: float = 500000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deconstructs every executed trade by Alpha 14 and correlates its outcome with opening bar features.
    """
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    strat = Alpha14GapMomentumDrift()

    symbols = DEFAULT_UNIVERSE if universe_type == "nifty14" else NIFTY_50_UNIVERSE
    window_days = lookback_days // n_windows

    print("=" * 120)
    print(f"[*] ASHVA ALPHA 14 FEATURE DECOMPOSITION & SIGNAL ATTRIBUTION")
    print(f"[*] Universe: {universe_type.upper()} ({len(symbols)} Assets) | Lookback: {lookback_days} Days")
    print(f"[*] Analyzing Decision-Point Features: Gap%, Gap/ATR, Body%, RVOL, NormATR%, GapHeld, Wicks, Long/Short")
    print("=" * 120)

    # 1. Load bars and precompute all technical components
    raw_bars = {}
    all_timestamps = []
    for sym in symbols:
        df = lake.load_bars(sym, "15m", max_lookback_days=lookback_days)
        if not df.empty and len(df) > 50:
            raw_bars[sym] = df
            all_timestamps.extend(df.index.tolist())

    max_global_ts = max(all_timestamps)
    min_global_ts = min(all_timestamps)

    # 2. Window partitions
    window_boundaries = []
    for w_idx in range(n_windows):
        start_day_offset = lookback_days - (w_idx * window_days)
        end_day_offset = lookback_days - ((w_idx + 1) * window_days)
        w_start_ts = max_global_ts - timedelta(days=start_day_offset)
        w_end_ts = max_global_ts - timedelta(days=end_day_offset)
        if w_idx == 0:
            w_start_ts = min_global_ts
        if w_idx == n_windows - 1:
            w_end_ts = max_global_ts + timedelta(hours=1)
        window_boundaries.append((f"W{w_idx+1}", w_start_ts, w_end_ts))

    def get_window_label(ts: pd.Timestamp) -> str:
        for w_label, w_start, w_end in window_boundaries:
            if w_start <= ts < w_end:
                return w_label
        return "Unknown"

    trade_records = []

    for sym, df in raw_bars.items():
        sig_df = strat.generate_signals(df)
        if sig_df.empty:
            continue

        # Extract features for all bars
        dates = pd.Series(df.index.date, index=df.index)
        daily_closes = df["close"].resample("1D").last().dropna()
        daily_highs = df["high"].resample("1D").max().dropna()
        daily_lows = df["low"].resample("1D").min().dropna()
        daily_tr = pd.concat([
            daily_highs - daily_lows,
            (daily_highs - daily_closes.shift(1)).abs(),
            (daily_lows - daily_closes.shift(1)).abs()
        ], axis=1).max(axis=1)
        daily_atrs = daily_tr.rolling(window=14).mean()

        prior_close_map = daily_closes.shift(1).to_dict()
        daily_atr_map = daily_atrs.to_dict()

        df_copy = df.copy()
        df_copy["time_only"] = df_copy.index.time
        tod_0915_vol = df_copy[df_copy["time_only"] == pd.to_datetime("09:15:00").time()]["volume"].rolling(20).mean()
        tod_vol_map = tod_0915_vol.to_dict()

        eng = BacktestEngine(cost_model=cost_model, initial_capital=capital_per_asset, segment=Segment.EQUITY_INTRADAY)
        res = eng.run(sig_df, symbol=sym, strategy_id="ALPHA_14", risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)

        for trade in res.trade_list:
            entry_ts = trade.entry_time
            # Find the signal bar (prior bar, 09:15)
            sig_idx = df.index.get_indexer([entry_ts], method="pad")[0] - 1
            if sig_idx < 0:
                continue

            sig_bar = df.iloc[sig_idx]
            sig_ts = df.index[sig_idx]
            sig_date = sig_ts.date()
            d_key = pd.to_datetime(sig_date)

            p_close = prior_close_map.get(d_key, np.nan)
            c_atr = daily_atr_map.get(d_key, np.nan)
            c_open = sig_bar["open"]
            c_high = sig_bar["high"]
            c_low = sig_bar["low"]
            c_close = sig_bar["close"]
            c_vol = sig_bar["volume"]
            c_tod = tod_vol_map.get(sig_ts, c_vol)

            if pd.isna(p_close) or p_close <= 0 or pd.isna(c_atr) or c_atr <= 0:
                continue

            gap_pct = ((c_open - p_close) / p_close) * 100.0
            abs_gap_pct = abs(gap_pct)
            gap_to_atr = abs(c_open - p_close) / c_atr
            candle_range = max(c_high - c_low, 0.01)
            body_ratio = (abs(c_close - c_open) / candle_range) * 100.0
            rvol = c_vol / max(1.0, c_tod)
            norm_atr_pct = (c_atr / p_close) * 100.0

            # Gap held check
            if trade.side == "LONG":
                gap_held = (c_low >= p_close)
                rejection_wick = ((c_high - c_close) / candle_range) * 100.0
            else:
                gap_held = (c_high <= p_close)
                rejection_wick = ((c_close - c_low) / candle_range) * 100.0

            w_label = get_window_label(entry_ts)
            is_win = (trade.net_pnl > 0)

            trade_records.append({
                "Trade_ID": trade.trade_id,
                "Symbol": sym,
                "Entry_Time": entry_ts,
                "Window": w_label,
                "Side": trade.side,
                "Net_PnL": trade.net_pnl,
                "Gross_PnL": trade.gross_pnl,
                "Outcome": "WIN" if is_win else "LOSS",
                "Is_Win": 1 if is_win else 0,
                "Exit_Reason": trade.exit_reason,
                "Duration_Bars": trade.duration_bars,
                "MFE_Pct": trade.mfe_pct,
                "MAE_Pct": trade.mae_pct,
                "Abs_Gap_Pct": abs_gap_pct,
                "Gap_Pct": gap_pct,
                "Gap_To_ATR": gap_to_atr,
                "Body_Ratio_Pct": body_ratio,
                "RVOL": rvol,
                "Norm_ATR_Pct": norm_atr_pct,
                "Rejection_Wick_Pct": rejection_wick,
                "Gap_Held": 1 if gap_held else 0,
            })

    trades_df = pd.DataFrame(trade_records)
    print(f"[*] Total Trades Deconstructed: {len(trades_df)} Trades across {len(symbols)} Symbols\n")

    if trades_df.empty:
        raise ValueError("No trades recorded to decompose.")

    # 1. Feature Comparison: Winners vs Losers across Full Dataset
    winners_df = trades_df[trades_df["Is_Win"] == 1]
    losers_df = trades_df[trades_df["Is_Win"] == 0]

    feature_cols = [
        ("Abs_Gap_Pct", "Overnight Gap (%)"),
        ("Gap_To_ATR", "Gap / Daily ATR Ratio"),
        ("Body_Ratio_Pct", "Bar 1 Body/Range (%)"),
        ("RVOL", "09:15 RVOL (x TOD)"),
        ("Norm_ATR_Pct", "Normalized ATR (%)"),
        ("Rejection_Wick_Pct", "Adverse Rejection Wick (%)"),
        ("Gap_Held", "Gap Held Entire Bar 1 (%)"),
        ("Duration_Bars", "Trade Duration (Bars)"),
        ("MFE_Pct", "Max Favorable Excursion (%)"),
        ("MAE_Pct", "Max Adverse Excursion (%)"),
    ]

    win_loss_comparison = []
    for col, name in feature_cols:
        w_vals = winners_df[col].dropna()
        l_vals = losers_df[col].dropna()

        w_mean = w_vals.mean()
        w_med = w_vals.median()
        l_mean = l_vals.mean()
        l_med = l_vals.median()

        # Two-sample t-test / Mann-Whitney U test
        t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False) if len(w_vals) > 1 and len(l_vals) > 1 else (0.0, 1.0)
        sig_marker = "⭐⭐ (p < 0.01)" if p_val < 0.01 else ("⭐ (p < 0.05)" if p_val < 0.05 else "ns (p >= 0.05)")

        win_loss_comparison.append({
            "Feature": name,
            "Winners_Mean": round(w_mean, 2),
            "Winners_Median": round(w_med, 2),
            "Losers_Mean": round(l_mean, 2),
            "Losers_Median": round(l_med, 2),
            "Difference": round(w_mean - l_mean, 2),
            "T_Statistic": round(t_stat, 2),
            "P_Value": round(p_val, 4),
            "Significance": sig_marker,
        })

    comp_df = pd.DataFrame(win_loss_comparison)

    # 2. Regime-by-Regime Feature Breakdown (W1 vs W3/W4 vs W5)
    regime_summary = []
    for w_label, grp in trades_df.groupby("Window"):
        n_t = len(grp)
        n_w = grp["Is_Win"].sum()
        wr = (n_w / n_t) * 100.0 if n_t > 0 else 0.0
        tot_pnl = grp["Net_PnL"].sum()

        regime_summary.append({
            "Window": w_label,
            "Trades": n_t,
            "Win_Rate_Pct": round(wr, 1),
            "Net_PnL_INR": round(tot_pnl, 2),
            "Avg_Gap_Pct": round(grp["Abs_Gap_Pct"].mean(), 2),
            "Avg_Gap_To_ATR": round(grp["Gap_To_ATR"].mean(), 2),
            "Avg_Body_Ratio_Pct": round(grp["Body_Ratio_Pct"].mean(), 1),
            "Avg_RVOL": round(grp["RVOL"].mean(), 2),
            "Avg_Norm_ATR": round(grp["Norm_ATR_Pct"].mean(), 2),
            "Gap_Held_Pct": round(grp["Gap_Held"].mean() * 100.0, 1),
            "SL_Exits_Pct": round((grp["Exit_Reason"] == "STOP_LOSS").mean() * 100.0, 1),
            "TP_Exits_Pct": round((grp["Exit_Reason"] == "TAKE_PROFIT").mean() * 100.0, 1),
            "EOD_Exits_Pct": round((grp["Exit_Reason"] == "EOD").mean() * 100.0, 1),
        })

    regime_df = pd.DataFrame(regime_summary).sort_values(by="Window")

    # 3. Long vs Short Asymmetry Breakdown
    side_summary = []
    for side, grp in trades_df.groupby("Side"):
        n_t = len(grp)
        n_w = grp["Is_Win"].sum()
        wr = (n_w / n_t) * 100.0 if n_t > 0 else 0.0
        tot_pnl = grp["Net_PnL"].sum()
        gp = grp[grp["Net_PnL"] > 0]["Net_PnL"].sum()
        gl = abs(grp[grp["Net_PnL"] < 0]["Net_PnL"].sum())
        pf = gp / max(1.0, gl)

        side_summary.append({
            "Side": side,
            "Trades": n_t,
            "Trades_Pct": round((n_t / len(trades_df)) * 100.0, 1),
            "Win_Rate_Pct": round(wr, 1),
            "Net_PnL_INR": round(tot_pnl, 2),
            "Net_PF": round(min(pf, 99.0), 2),
            "Avg_Trade_INR": round(tot_pnl / n_t, 1),
        })

    side_df = pd.DataFrame(side_summary)

    # 4. Exit Reason & Path Analysis
    exit_summary = []
    for reason, grp in trades_df.groupby("Exit_Reason"):
        n_t = len(grp)
        tot_pnl = grp["Net_PnL"].sum()
        exit_summary.append({
            "Exit_Reason": reason,
            "Trades": n_t,
            "Pct_Of_Total": round((n_t / len(trades_df)) * 100.0, 1),
            "Net_PnL_INR": round(tot_pnl, 2),
            "Avg_Duration_Bars": round(grp["Duration_Bars"].mean(), 1),
            "Avg_MFE_Pct": round(grp["MFE_Pct"].mean(), 2),
            "Avg_MAE_Pct": round(grp["MAE_Pct"].mean(), 2),
        })

    exit_df = pd.DataFrame(exit_summary)

    # Write Markdown Report
    out_md = f"# ASHVA ALPHA 14 SIGNAL DECOMPOSITION & ATTRIBUTION REPORT\n\n"
    out_md += f"> **Strategy**: `ALPHA_14_GAP_MOMENTUM_DRIFT` (Frozen Parameters)\n"
    out_md += f"> **Total Trades Analyzed**: `{len(trades_df)} Trades` across `{len(symbols)} Stocks` over `540 Days`\n"
    out_md += f"> **Cost Engine**: Indian Statutory Taxes + 3.0 bps Slippage\n\n"

    out_md += "## 1. Feature Comparison: Winning Trades vs Losing Trades\n\n"
    out_md += comp_df.to_markdown(index=False) + "\n\n"

    out_md += "## 2. Regime-by-Regime Signal Microstructure (Favorable W1 vs Choppy W3/W4 vs OOS W5)\n\n"
    out_md += regime_df.to_markdown(index=False) + "\n\n"

    out_md += "## 3. Directional Asymmetry: Long vs Short Breakdown\n\n"
    out_md += side_df.to_markdown(index=False) + "\n\n"

    out_md += "## 4. Execution Path & Exit Reason Distribution\n\n"
    out_md += exit_df.to_markdown(index=False) + "\n\n"

    out_file = Path("alpha_14_feature_decomposition_report.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(out_md)

    print("=" * 120)
    print(f"[*] DECOMPOSITION COMPLETE: Report saved to {out_file.resolve()}")
    print("=" * 120)

    return comp_df, regime_df, side_df, exit_df


def main():
    parser = argparse.ArgumentParser(description="Alpha 14 Signal Feature Decomposition")
    parser.add_argument("--universe", type=str, choices=["nifty14", "nifty50"], default="nifty14",
                        help="Universe to analyze (default: nifty14)")
    parser.add_argument("--lookback", type=int, default=540,
                        help="Lookback in days (default: 540)")
    parser.add_argument("--windows", type=int, default=5,
                        help="Number of chronological windows (default: 5)")

    args = parser.parse_args()
    run_feature_decomposition(
        universe_type=args.universe,
        lookback_days=args.lookback,
        n_windows=args.windows
    )


if __name__ == "__main__":
    main()
