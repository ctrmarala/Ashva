"""
Ashva Proven Alphas Replay Portfolio Runner
Executes all PROVEN positive alphas across the full liquid universe for any custom period.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine, BacktestTrade
from src.strategies.registry import get_all_strategies, get_strategy_by_name
from src.ui.data_access import UIDataAccess
from src.analytics.metrics import calculate_profit_factor


def main():
    parser = argparse.ArgumentParser(description="Replay Proven Alphas")
    parser.add_argument("--start", type=str, default="2026-08-15", help="Start Date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-09-16", help="End Date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=500000.0, help="Portfolio capital in INR")
    args = parser.parse_args()

    lake = DataLake(read_only=True)
    symbols = get_universe_symbols()
    cost_model = IndianCostModel()
    dal = UIDataAccess()

    # Identify all currently PROVEN alphas
    all_alphas_df = dal.get_alpha_registry_table()
    proven_df = all_alphas_df[all_alphas_df["status"] == "PROVEN"]
    proven_ids = proven_df["alpha_id"].tolist()

    print("=" * 115)
    print(f"[*] ASHVA PORTFOLIO REPLAY: {len(proven_ids)} PROVEN ALPHAS ({args.start} to {args.end})")
    print(f"[*] Universe: {len(symbols)} Equities | Portfolio Capital: Rs {args.capital:,.0f}")
    print(f"[*] Proven Alphas: {', '.join(proven_ids)}")
    print("=" * 115)

    engine = BacktestEngine(
        cost_model=cost_model,
        initial_capital=args.capital,
        segment=Segment.EQUITY_INTRADAY,
        use_1m_intrabar=True,
        data_lake=lake,
    )

    s_ts = pd.to_datetime(f"{args.start} 00:00:00")
    e_ts = pd.to_datetime(f"{args.end} 23:59:59")

    all_portfolio_trades = []
    alpha_breakdown = {}

    for alpha_id in proven_ids:
        strat_cls = get_strategy_by_name(alpha_id)
        if not strat_cls:
            continue

        # Get optimal timeframe from UIDataAccess
        detail = dal.get_alpha_detail(alpha_id)
        optimal_tf = detail.get("timeframe") or "15m"
        
        strat = strat_cls({"timeframe": optimal_tf})
        alpha_trades = []

        for sym in symbols:
            df = lake.load_bars(sym, optimal_tf, max_lookback_days=540)
            if df.empty or len(df) < 50:
                continue

            sig_df = strat.generate_signals(df)
            res = engine.run(sig_df, symbol=sym, strategy_id=alpha_id, capital_per_trade_pct=0.25)
            
            period_trades = [
                t for t in res.trade_list
                if s_ts <= pd.to_datetime(t.entry_time) <= e_ts
            ]
            for pt in period_trades:
                # Attach alpha_id
                pt_dict = {
                    "alpha_id": alpha_id,
                    "symbol": pt.symbol,
                    "side": pt.side,
                    "entry_time": pt.entry_time,
                    "exit_time": pt.exit_time,
                    "entry_price": pt.entry_price,
                    "exit_price": pt.exit_price,
                    "quantity": pt.quantity,
                    "gross_pnl": pt.gross_pnl,
                    "costs": pt.cost_breakdown.total_tax_and_charges,
                    "net_pnl": pt.net_pnl,
                    "exit_reason": pt.exit_reason,
                    "timeframe": optimal_tf,
                }
                alpha_trades.append(pt_dict)
                all_portfolio_trades.append(pt_dict)

        a_gross = sum(t["gross_pnl"] for t in alpha_trades)
        a_costs = sum(t["costs"] for t in alpha_trades)
        a_net = sum(t["net_pnl"] for t in alpha_trades)
        a_wins = sum(1 for t in alpha_trades if t["net_pnl"] > 0)
        a_n = len(alpha_trades)
        a_wr = (a_wins / a_n * 100.0) if a_n > 0 else 0.0
        a_pf = calculate_profit_factor([t["net_pnl"] for t in alpha_trades]) if a_n > 0 else 0.0

        alpha_breakdown[alpha_id] = {
            "timeframe": optimal_tf,
            "trades": a_n,
            "wins": a_wins,
            "win_rate": a_wr,
            "gross_pnl": a_gross,
            "costs": a_costs,
            "net_pnl": a_net,
            "profit_factor": a_pf,
        }

    # Sort all trades chronologically
    all_portfolio_trades.sort(key=lambda t: t["entry_time"])

    print("\n" + "-" * 115)
    print("DETAILED CHRONOLOGICAL EXECUTED TRADE LOG:")
    print("-" * 115)
    print(f"{'#':<3} {'Alpha':<10} {'Symbol':<12} {'TF':<4} {'Side':<6} {'Entry Time':<17} {'Exit Time':<17} {'Gross':<10} {'Costs':<9} {'Net PnL':<10} {'Reason':<11}")
    print("-" * 115)

    tot_gross = 0.0
    tot_costs = 0.0
    tot_net = 0.0
    tot_wins = 0

    for idx, t in enumerate(all_portfolio_trades, 1):
        tot_gross += t["gross_pnl"]
        tot_costs += t["costs"]
        tot_net += t["net_pnl"]
        if t["net_pnl"] > 0:
            tot_wins += 1

        print(
            f"{idx:<3} {t['alpha_id']:<10} {t['symbol']:<12} {t['timeframe']:<4} {t['side']:<6} "
            f"{str(t['entry_time'])[:16]:<17} {str(t['exit_time'])[:16]:<17} "
            f"{t['gross_pnl']:+9.2f} {t['costs']:8.2f} {t['net_pnl']:+9.2f} {t['exit_reason']:<11}"
        )

    tot_trades = len(all_portfolio_trades)
    tot_wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    tot_pf = calculate_profit_factor([t["net_pnl"] for t in all_portfolio_trades]) if tot_trades > 0 else 0.0
    period_roi = (tot_net / args.capital) * 100.0

    print("-" * 115)
    print("\n" + "=" * 80)
    print(f"PER-ALPHA PERFORMANCE IN PERIOD ({args.start} to {args.end})")
    print("=" * 80)
    print(f"{'Alpha ID':<12} {'TF':<5} {'Trades':<8} {'Win Rate':<10} {'Gross PnL':<14} {'Costs':<12} {'Net Realized PnL':<18} {'Net PF':<8}")
    print("-" * 80)
    for a_id, stats in alpha_breakdown.items():
        print(
            f"{a_id:<12} {stats['timeframe']:<5} {stats['trades']:<8} {stats['win_rate']:5.1f}%     "
            f"Rs {stats['gross_pnl']:+10.2f}   Rs {stats['costs']:8.2f}   Rs {stats['net_pnl']:+12.2f}   {stats['profit_factor']:.2f}"
        )
    print("-" * 80)

    print("\n" + "=" * 60)
    print("COMBINED PORTFOLIO REPLAY SUMMARY")
    print("=" * 60)
    print(f"Total Portfolio Trades Executed:  {tot_trades}")
    print(f"Winning Trades:                   {tot_wins} ({tot_wr:.1f}%)")
    print(f"Losing Trades:                    {tot_trades - tot_wins}")
    print(f"Total Gross Realized P&L:         Rs {tot_gross:+,.2f}")
    print(f"Total Statutory Costs Paid:       Rs {tot_costs:,.2f}")
    print(f"Total Net Realized P&L:           Rs {tot_net:+,.2f}")
    print(f"Portfolio Net Profit Factor:      {tot_pf:.2f}")
    print(f"Initial Portfolio Capital:        Rs {args.capital:,.2f}")
    print(f"Ending Portfolio Equity:          Rs {args.capital + tot_net:,.2f}")
    print(f"Period Net Return on Investment:  {period_roi:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()