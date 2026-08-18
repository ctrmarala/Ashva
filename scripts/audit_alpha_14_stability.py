"""
Ashva Quantitative Alpha 14 (Overnight Gap Momentum Drift) Chronological Stability Audit
Evaluates Alpha 14 across multiple consecutive, non-overlapping chronological time windows
using the frozen strategy parameters, DataLake, BacktestEngine, and Indian statutory cost model.

Objective:
Test whether Alpha 14's positive performance is consistent across multiple independent market regimes,
or isolated to a single fortunate time window.
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.alpha_14_gap_momentum_drift import Alpha14GapMomentumDrift
from scripts.run_hypothesis_lab import DEFAULT_UNIVERSE
from scripts.scan_nifty_50_universe import NIFTY_50_UNIVERSE


def run_alpha_14_stability_audit(
    universe_type: str = "nifty14",
    total_lookback_days: int = 540,
    n_windows: int = 5,
    capital_per_asset: float = 500000.0,
    risk_per_trade_pct: float = 0.005,
    capital_per_trade_pct: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes a multi-window chronological stability audit for Alpha 14 across independent time slices.
    """
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    strat = Alpha14GapMomentumDrift()

    symbols = DEFAULT_UNIVERSE if universe_type == "nifty14" else NIFTY_50_UNIVERSE
    total_basket_capital = capital_per_asset * len(symbols)
    window_days = total_lookback_days // n_windows

    print("=" * 120)
    print(f"[*] ASHVA ALPHA 14 CHRONOLOGICAL WINDOW STABILITY AUDIT")
    print(f"[*] Universe: {universe_type.upper()} ({len(symbols)} Assets) | Total Lookback: {total_lookback_days} Days")
    print(f"[*] Partition: {n_windows} Consecutive Non-Overlapping Windows (~{window_days} Days Each)")
    print(f"[*] Capital: Rs {capital_per_asset:,.0f}/Asset (Total Basket Capital = Rs {total_basket_capital:,.0f})")
    print(f"[*] Costs: Indian Statutory Taxes + 3.0 bps Slippage")
    print("=" * 120)

    # 1. Load bars and generate signals for all assets
    symbol_signals = {}
    all_timestamps = []

    for sym in symbols:
        df = lake.load_bars(sym, "15m", max_lookback_days=total_lookback_days)
        if not df.empty and len(df) > 50:
            sig_df = strat.generate_signals(df)
            symbol_signals[sym] = sig_df
            all_timestamps.extend(sig_df.index.tolist())

    if not all_timestamps:
        raise ValueError("No price/signal data loaded from DataLake.")

    min_global_ts = min(all_timestamps)
    max_global_ts = max(all_timestamps)
    total_span_days = (max_global_ts - min_global_ts).days

    print(f"[*] Dataset Global Span: {min_global_ts.strftime('%Y-%m-%d')} to {max_global_ts.strftime('%Y-%m-%d')} ({total_span_days} days)")

    # 2. Build time window boundaries
    window_boundaries = []
    for w_idx in range(n_windows):
        # Window 0 is the oldest, Window (n_windows-1) is the most recent
        start_day_offset = total_lookback_days - (w_idx * window_days)
        end_day_offset = total_lookback_days - ((w_idx + 1) * window_days)
        
        w_start_ts = max_global_ts - timedelta(days=start_day_offset)
        w_end_ts = max_global_ts - timedelta(days=end_day_offset)
        
        if w_idx == 0:
            w_start_ts = min_global_ts
        if w_idx == n_windows - 1:
            w_end_ts = max_global_ts + timedelta(hours=1)

        window_boundaries.append((f"Window_{w_idx+1}", w_start_ts, w_end_ts))

    # 3. Evaluate each window
    window_summary_records = []
    window_symbol_details = []

    for w_label, w_start, w_end in window_boundaries:
        w_trades = 0
        w_wins = 0
        w_net_pnl = 0.0
        w_gross_profit = 0.0
        w_gross_loss = 0.0
        w_taxes = 0.0
        all_symbol_daily = []

        print(f"\n[>] Evaluating {w_label} ({w_start.strftime('%Y-%m-%d')} -> {w_end.strftime('%Y-%m-%d')})...", end="", flush=True)

        for sym, sig_df in symbol_signals.items():
            w_sig = sig_df[(sig_df.index >= w_start) & (sig_df.index < w_end)]
            if w_sig.empty or len(w_sig) < 10:
                continue

            eng = BacktestEngine(
                cost_model=cost_model,
                initial_capital=capital_per_asset,
                segment=Segment.EQUITY_INTRADAY
            )
            res = eng.run(
                w_sig,
                symbol=sym,
                strategy_id="ALPHA_14_GAP_MOMENTUM_DRIFT",
                risk_per_trade_pct=risk_per_trade_pct,
                capital_per_trade_pct=capital_per_trade_pct
            )

            if res.total_trades > 0:
                w_trades += res.total_trades
                w_wins += res.winning_trades
                w_net_pnl += res.total_net_pnl
                w_gross_profit += sum(t.gross_pnl for t in res.trade_list if t.gross_pnl > 0)
                w_gross_loss += sum(abs(t.gross_pnl) for t in res.trade_list if t.gross_pnl < 0)
                w_taxes += res.total_taxes_paid

                if len(res.equity_curve) > 1:
                    d_pnl = res.equity_curve.diff().resample("1D").sum().fillna(0.0)
                    all_symbol_daily.append(d_pnl)

                if res.total_net_pnl > 0:
                    window_symbol_details.append({
                        "Window": w_label,
                        "Date_Range": f"{w_start.strftime('%Y-%m-%d')} to {w_end.strftime('%Y-%m-%d')}",
                        "Symbol": sym,
                        "Net_PnL_INR": round(res.total_net_pnl, 2),
                        "Trades": res.total_trades,
                        "Win_Rate_Pct": round(res.win_rate_pct, 1),
                        "PF": round(min(res.net_profit_factor, 99.0), 2),
                        "Sharpe": round(res.sharpe_ratio, 2),
                    })

        # Calculate portfolio metrics for this window
        win_rate = (w_wins / max(1, w_trades)) * 100.0 if w_trades > 0 else 0.0
        basket_roi_pct = (w_net_pnl / total_basket_capital) * 100.0
        net_pf = (w_gross_profit / max(1.0, w_gross_loss)) if w_gross_loss > 0 else (99.0 if w_gross_profit > 0 else 0.0)

        if all_symbol_daily:
            comb_pnl = pd.concat(all_symbol_daily, axis=1).sum(axis=1).fillna(0.0)
            ret_series = (comb_pnl / total_basket_capital).dropna()
            m_r = ret_series.mean()
            s_r = ret_series.std()
            sharpe = float((m_r / s_r * np.sqrt(252))) if s_r > 1e-7 else 0.0
        else:
            sharpe = 0.0

        status = "🟢 Profitable" if w_net_pnl > 0 else "🔴 Loss"
        print(f" [DONE: Trades={w_trades}, WinRate={win_rate:.1f}%, Net PnL=Rs {w_net_pnl:+,.0f}, Basket ROI={basket_roi_pct:+.2f}%, Sharpe={sharpe:.2f}]")

        window_summary_records.append({
            "Window": w_label,
            "Date_Range": f"{w_start.strftime('%Y-%m-%d')} to {w_end.strftime('%Y-%m-%d')}",
            "Trades": w_trades,
            "Win_Rate_Pct": round(win_rate, 1),
            "Net_PnL_INR": round(w_net_pnl, 2),
            "Basket_ROI_Pct": round(basket_roi_pct, 2),
            "Net_PF": round(min(net_pf, 99.0), 2),
            "Sharpe": round(sharpe, 2),
            "Taxes_Paid_INR": round(w_taxes, 2),
            "Status": status,
        })

    summary_df = pd.DataFrame(window_summary_records)
    details_df = pd.DataFrame(window_symbol_details)

    # Markdown Report Generation
    out_md = f"# ASHVA ALPHA 14 CHRONOLOGICAL STABILITY REPORT\n\n"
    out_md += f"> **Strategy**: `ALPHA_14_GAP_MOMENTUM_DRIFT` (Frozen Parameters)\n"
    out_md += f"> **Universe**: `{universe_type.upper()}` ({len(symbols)} Assets) | **Partition**: `{n_windows} Non-Overlapping Windows`\n"
    out_md += f"> **Cost Engine**: Indian Statutory Taxes + 3.0 bps Slippage\n\n"

    out_md += "## 1. Chronological Window Stability Matrix\n\n"
    out_md += summary_df.to_markdown(index=False) + "\n\n"

    out_md += "## 2. Window-by-Window Positive Asset Contributions\n\n"
    if not details_df.empty:
        out_md += details_df.to_markdown(index=False) + "\n\n"
    else:
        out_md += "No positive asset contributions in any window.\n\n"

    output_path = Path("alpha_14_stability_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out_md)

    print("\n" + "=" * 120)
    print(f"[*] ALPHA 14 STABILITY AUDIT COMPLETE: Report saved to {output_path.resolve()}")
    print("=" * 120)

    return summary_df, details_df


def main():
    parser = argparse.ArgumentParser(description="Alpha 14 Chronological Window Stability Audit")
    parser.add_argument("--universe", type=str, choices=["nifty14", "nifty50"], default="nifty14",
                        help="Universe to audit (default: nifty14)")
    parser.add_argument("--windows", type=int, default=5,
                        help="Number of non-overlapping chronological windows (default: 5)")
    parser.add_argument("--lookback", type=int, default=540,
                        help="Total historical lookback in days (default: 540)")

    args = parser.parse_args()
    run_alpha_14_stability_audit(
        universe_type=args.universe,
        total_lookback_days=args.lookback,
        n_windows=args.windows
    )


if __name__ == "__main__":
    main()
