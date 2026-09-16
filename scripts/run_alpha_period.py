"""
Ashva Precision Period Backtest Runner
Allows executing and inspecting any Alpha strategy over a custom date range (e.g. Aug 15 to Sept 15).

Usage:
  python scripts/run_alpha_period.py --alpha-id 33_alpha --start 2026-08-15 --end 2026-09-15
  python scripts/run_alpha_period.py --alpha-id 33_alpha --timeframe 15m --start 2026-08-15 --end 2026-09-15
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine, BacktestTrade
from src.strategies.registry import get_strategy_by_name
from src.analytics.metrics import calculate_profit_factor


def parse_args():
    parser = argparse.ArgumentParser(description="Run Alpha over custom date range")
    parser.add_argument("--alpha-id", type=str, default="33_alpha", help="Strategy ID (e.g., 33_alpha, 45_alpha)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (15m, 5m, 30m, 1m)")
    parser.add_argument("--start", type=str, default="2026-08-15", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-09-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=500000.0, help="Initial capital in INR")
    parser.add_argument("--allocation", type=float, default=0.25, help="Capital allocation per trade")
    return parser.parse_args()


def run_period_backtest():
    args = parse_args()
    lake = DataLake(read_only=True)
    symbols = get_universe_symbols()
    cost_model = IndianCostModel()
    
    strat_cls = get_strategy_by_name(args.alpha_id)
    if not strat_cls:
        print(f"[!] Error: Strategy '{args.alpha_id}' not found in registry.")
        sys.exit(1)

    strat = strat_cls({"timeframe": args.timeframe})
    engine = BacktestEngine(
        cost_model=cost_model,
        initial_capital=args.capital,
        segment=Segment.EQUITY_INTRADAY,
        use_1m_intrabar=True,
        data_lake=lake,
    )

    s_ts = pd.to_datetime(f"{args.start} 00:00:00")
    e_ts = pd.to_datetime(f"{args.end} 23:59:59")

    print("=" * 105)
    print(f"[*] ASHVA PRECISION PERIOD BACKTEST: {args.alpha_id.upper()} ({args.timeframe.upper()})")
    print(f"[*] Period: {args.start} to {args.end} | Universe: {len(symbols)} Equities | Capital: Rs {args.capital:,.0f}")
    print("=" * 105)

    all_trades = []
    symbol_trade_count = 0

    for sym in symbols:
        df = lake.load_bars(sym, args.timeframe, max_lookback_days=540)
        if df.empty or len(df) < 50:
            continue

        # Generate signals on full history to avoid warm-up boundary distortion
        sig_df = strat.generate_signals(df)
        
        # Run engine
        res = engine.run(sig_df, symbol=sym, strategy_id=args.alpha_id, capital_per_trade_pct=args.allocation)
        
        # Filter trades strictly within the target date window
        period_trades = [
            t for t in res.trade_list
            if s_ts <= pd.to_datetime(t.entry_time) <= e_ts
        ]
        
        if period_trades:
            all_trades.extend(period_trades)
            symbol_trade_count += 1

    # Sort trades chronologically
    all_trades.sort(key=lambda t: t.entry_time)

    print(f"\n[+] Total Trades Found in Period ({args.start} to {args.end}): {len(all_trades)}")
    print("-" * 105)
    print(f"{'#':<3} {'Symbol':<12} {'Side':<6} {'Entry Time':<17} {'Exit Time':<17} {'Entry':<9} {'Exit':<9} {'Gross':<10} {'Taxes':<9} {'Net PnL':<10} {'Reason':<11}")
    print("-" * 105)

    total_gross = 0.0
    total_costs = 0.0
    total_net = 0.0
    wins = 0

    for idx, t in enumerate(all_trades, 1):
        gross = t.gross_pnl
        costs = t.cost_breakdown.total_tax_and_charges
        net = t.net_pnl
        total_gross += gross
        total_costs += costs
        total_net += net
        if net > 0:
            wins += 1

        print(
            f"{idx:<3} {t.symbol:<12} {t.side:<6} {str(t.entry_time)[:16]:<17} {str(t.exit_time)[:16]:<17} "
            f"{t.entry_price:<9.2f} {t.exit_price:<9.2f} {gross:+9.2f} {costs:8.2f} {net:+9.2f} {t.exit_reason:<11}"
        )

    print("-" * 105)
    win_rate = (wins / max(1, len(all_trades))) * 100.0
    net_pnls = [t.net_pnl for t in all_trades]
    net_pf = calculate_profit_factor(net_pnls)

    print("\n" + "=" * 50)
    print("PERIOD PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Period Trades:      {len(all_trades)}")
    print(f"Winning Trades:           {wins} ({win_rate:.1f}%)")
    print(f"Losing Trades:            {len(all_trades) - wins}")
    print(f"Gross Trading P&L:        Rs {total_gross:+,.2f}")
    print(f"Statutory Taxes & Costs:  Rs {total_costs:,.2f}")
    print(f"Net Realized P&L:         Rs {total_net:+,.2f}")
    print(f"Net Profit Factor:        {net_pf:.2f}")
    print(f"Period Net ROI:           {(total_net / args.capital) * 100:+.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_period_backtest()