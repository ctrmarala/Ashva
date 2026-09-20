"""
Ashva Configurable Unified Shared-Capital Multi-Alpha Portfolio Replay Runner

Executes full multi-alpha replay over a shared cash pool with configurable:
- Start and End Date
- Initial Portfolio Capital
- Max Concurrent Positions (Slots) & Capital Allocation %
- Max Consecutive Losses / Daily Trade Guardrails
- Automatic 5-Month Monthly Breakdown or Custom Period Replay
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.registry import get_strategy_by_name
from src.ui.data_access import UIDataAccess
from src.analytics.unified_portfolio_engine import UnifiedPortfolioEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ashva Configurable Unified Multi-Alpha Shared-Capital Portfolio Replay"
    )
    parser.add_argument("--start-date", "-s", type=str, default=None, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", type=str, default=None, help="End Date (YYYY-MM-DD)")
    parser.add_argument("--capital", "-c", type=float, default=500000.0, help="Initial Portfolio Capital (default: 500000.0)")
    parser.add_argument("--max-concurrent", "-m", type=int, default=4, help="Max Concurrent Positions (default: 4)")
    parser.add_argument("--slot-pct", type=float, default=None, help="Capital Allocation percentage per Slot (default: 1/max_concurrent)")
    parser.add_argument("--compound", action="store_true", help="Enable portfolio equity compounding")
    parser.add_argument("--max-consecutive-losses", type=int, default=None, help="Max Consecutive Losses before Strategy Lockout")
    parser.add_argument("--max-daily-trades-per-strat", type=int, default=None, help="Max Trades per Strategy per Day")
    parser.add_argument("--max-daily-trades-total", type=int, default=None, help="Max Total Portfolio Trades per Day")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols list (default: full 77 universe)")
    parser.add_argument("--alphas", type=str, default=None, help="Comma-separated alpha IDs (default: all PROVEN alphas)")
    parser.add_argument("--export-csv", type=str, default=None, help="File path to export executed trades CSV")
    parser.add_argument("--monthly-breakdown", action="store_true", default=True, help="Display month-by-month performance breakdown")
    return parser.parse_args()


def extract_candidate_trades(lake: DataLake, symbols: list, alpha_ids: list, dal: UIDataAccess) -> list:
    """Extracts raw candidate trade signals across selected alphas and symbols."""
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    engine = BacktestEngine(
        cost_model=cost_model,
        initial_capital=500000.0,
        segment=Segment.EQUITY_INTRADAY,
        use_1m_intrabar=True,
        data_lake=lake,
    )

    all_candidates = []
    print(f"[*] Extracting candidate signals for {len(alpha_ids)} alphas across {len(symbols)} symbols...")

    for idx, alpha_id in enumerate(alpha_ids, 1):
        strat_cls = get_strategy_by_name(alpha_id)
        if not strat_cls:
            continue

        detail = dal.get_alpha_detail(alpha_id)
        optimal_tf = detail.get("timeframe") or "15m"
        strat = strat_cls({"timeframe": optimal_tf})

        alpha_trade_count = 0
        for sym in symbols:
            df = lake.load_bars(sym, optimal_tf, max_lookback_days=540)
            if df.empty or len(df) < 50:
                continue

            sig_df = strat.generate_signals(df)
            res = engine.run(sig_df, symbol=sym, strategy_id=alpha_id, capital_per_trade_pct=0.25)
            
            for t in res.trade_list:
                all_candidates.append({
                    "alpha_id": alpha_id,
                    "symbol": sym,
                    "side": t.side,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                })
                alpha_trade_count += 1

        print(f"  [{idx:02d}/{len(alpha_ids):02d}] {alpha_id:<12} ({optimal_tf:>3}): {alpha_trade_count:4d} signals")

    print(f"[+] Total raw candidate signals extracted: {len(all_candidates):,}\n")
    return all_candidates


def main():
    args = parse_args()
    lake = DataLake(read_only=True)
    dal = UIDataAccess()

    symbols = args.symbols.split(",") if args.symbols else get_universe_symbols()
    
    if args.alphas:
        alpha_ids = [a.strip() for a in args.alphas.split(",")]
    else:
        all_alphas_df = dal.get_alpha_registry_table()
        proven_df = all_alphas_df[all_alphas_df["status"] == "PROVEN"]
        alpha_ids = proven_df["alpha_id"].tolist()

    slot_pct = args.slot_pct if args.slot_pct is not None else (1.0 / args.max_concurrent)

    print("=" * 125)
    print("ASHVA CONFIGURABLE UNIFIED SHARED-CAPITAL PORTFOLIO REPLAY")
    print(f"  • Portfolio Capital:          Rs {args.capital:,.0f}")
    print(f"  • Concurrency Slots:          {args.max_concurrent} Max Simultaneous Positions ({slot_pct*100:.1f}% per slot)")
    print(f"  • Equity Compounding:         {'ENABLED' if args.compound else 'DISABLED (Fixed Sizing)'}")
    print(f"  • Consecutive Loss Lockout:   {args.max_consecutive_losses or 'None'}")
    print(f"  • Max Daily Trades / Strat:   {args.max_daily_trades_per_strat or 'Unlimited'}")
    print(f"  • Max Daily Trades / Total:   {args.max_daily_trades_total or 'Unlimited'}")
    print(f"  • Alphas Selected:            {len(alpha_ids)} Strategies")
    print(f"  • Symbols Selected:           {len(symbols)} Equities")
    print("=" * 125)

    candidate_trades = extract_candidate_trades(lake, symbols, alpha_ids, dal)

    engine = UnifiedPortfolioEngine(
        initial_capital=args.capital,
        max_concurrent_positions=args.max_concurrent,
        capital_per_trade_pct=slot_pct,
        compound_capital=args.compound,
        max_consecutive_losses=args.max_consecutive_losses,
        max_daily_trades_per_strategy=args.max_daily_trades_per_strat,
        max_daily_trades_total=args.max_daily_trades_total,
    )

    # Determine replay schedule
    if args.start_date and args.end_date:
        periods = [(f"Custom Period ({args.start_date} to {args.end_date})", args.start_date, args.end_date)]
    else:
        periods = [
            ("Month 1 (Apr 19 - May 18, 2026)", "2026-04-19", "2026-05-18"),
            ("Month 2 (May 19 - Jun 18, 2026)", "2026-05-19", "2026-06-18"),
            ("Month 3 (Jun 19 - Jul 18, 2026)", "2026-06-19", "2026-07-18"),
            ("Month 4 (Jul 19 - Aug 18, 2026)", "2026-07-19", "2026-08-18"),
            ("Month 5 (Aug 19 - Sep 18, 2026)", "2026-08-19", "2026-09-18"),
        ]

    period_results = []
    for label, s_date, e_date in periods:
        res = engine.run_portfolio_simulation(candidate_trades, start_date=s_date, end_date=e_date)
        res["label"] = label
        period_results.append(res)

    print("\n" + "=" * 125)
    print(f"{'PERIOD':<32} | {'CANDIDATES':<10} | {'EXECUTED':<8} | {'REJECTED':<8} | {'WIN %':<6} | {'GROSS PNL':<12} | {'TAXES/FEES':<11} | {'NET PNL':<12} | {'ROI %':<8} | {'MAX DD':<7}")
    print("-" * 125)

    tot_cands = sum(m["total_candidates"] for m in period_results)
    tot_exec = sum(m["total_executed_trades"] for m in period_results)
    tot_rej = sum(m["total_rejected_signals"] for m in period_results)
    tot_gross = sum(m["gross_pnl"] for m in period_results)
    tot_taxes = sum(m["total_taxes_and_costs"] for m in period_results)
    tot_net = sum(m["net_pnl"] for m in period_results)
    all_exec_trades = [t for m in period_results for t in m["executed_trades"]]
    overall_wr = (len([t for t in all_exec_trades if t.net_pnl > 0]) / max(1, len(all_exec_trades))) * 100.0
    overall_roi = (tot_net / args.capital) * 100.0

    for m in period_results:
        print(f"{m['label']:<32} | {m['total_candidates']:>10d} | {m['total_executed_trades']:>8d} | {m['total_rejected_signals']:>8d} | {m['win_rate_pct']:>5.1f}% | Rs {m['gross_pnl']:>9,.0f} | Rs {m['total_taxes_and_costs']:>8,.0f} | Rs {m['net_pnl']:>9,.0f} | {m['roi_pct']:>+6.2f}% | {m['max_drawdown_pct']:>5.2f}%")

    print("-" * 125)
    print(f"{'CUMULATIVE TOTAL':<32} | {tot_cands:>10d} | {tot_exec:>8d} | {tot_rej:>8d} | {overall_wr:>5.1f}% | Rs {tot_gross:>9,.0f} | Rs {tot_taxes:>8,.0f} | Rs {tot_net:>9,.0f} | {overall_roi:>+6.2f}% |")
    print("=" * 125)

    # Rejection summary
    all_rejections = [r for m in period_results for r in m["rejected_signals"]]
    if all_rejections:
        print("\n[*] SIGNAL REJECTION ANALYSIS (Capacity & Risk Guardrails):")
        rej_counts = {}
        for r in all_rejections:
            rej_counts[r.reason] = rej_counts.get(r.reason, 0) + 1
        for r_reason, count in sorted(rej_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(all_rejections)) * 100.0
            print(f"  • {r_reason:<30}: {count:5d} signals ({pct:5.1f}%)")

    # Optional export
    if args.export_csv:
        export_records = []
        for t in all_exec_trades:
            export_records.append({
                "trade_id": t.trade_id,
                "alpha_id": t.alpha_id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "allocated_capital": t.allocated_capital,
                "gross_pnl": t.gross_pnl,
                "taxes_and_charges": t.cost_breakdown.total_tax_and_charges,
                "net_pnl": t.net_pnl,
                "roi_pct": t.roi_pct,
                "exit_reason": t.exit_reason,
            })
        df_export = pd.DataFrame(export_records)
        df_export.to_csv(args.export_csv, index=False)
        print(f"\n[+] Exported {len(df_export)} executed trades to: {args.export_csv}")


if __name__ == "__main__":
    main()

