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


def evaluate_alpha_period(
    alpha_id: str,
    timeframe: str = "15m",
    start_date: str = "2026-08-15",
    end_date: str = "2026-09-15",
    capital: float = 500000.0,
    allocation: float = 0.25,
    lake: Optional[DataLake] = None,
    symbols: Optional[list] = None,
    cost_model: Optional[IndianCostModel] = None,
) -> dict:
    """
    Core reusable period backtest engine.
    Calculates exact trade list, win rate, gross, costs, and net PnL for any alpha over an arbitrary date range.
    """
    lake = lake or DataLake(read_only=True)
    symbols = symbols or get_universe_symbols()
    cost_model = cost_model or IndianCostModel()

    strat_cls = get_strategy_by_name(alpha_id)
    if not strat_cls:
        return {"error": f"Strategy '{alpha_id}' not found in registry.", "trades": 0, "net_pnl": 0.0}

    strat = strat_cls({"timeframe": timeframe})
    engine = BacktestEngine(
        cost_model=cost_model,
        initial_capital=capital,
        segment=Segment.EQUITY_INTRADAY,
        use_1m_intrabar=True,
        data_lake=lake,
    )

    s_ts = pd.to_datetime(f"{start_date} 00:00:00")
    e_ts = pd.to_datetime(f"{end_date} 23:59:59")

    all_trades = []
    for sym in symbols:
        df = lake.load_bars(sym, timeframe, max_lookback_days=540)
        if df.empty or len(df) < 50:
            continue

        sig_df = strat.generate_signals(df)
        res = engine.run(sig_df, symbol=sym, strategy_id=alpha_id, capital_per_trade_pct=allocation)

        period_trades = [
            t for t in res.trade_list
            if s_ts <= pd.to_datetime(t.entry_time) <= e_ts
        ]
        if period_trades:
            all_trades.extend(period_trades)

    all_trades.sort(key=lambda t: t.entry_time)

    total_gross = sum(t.gross_pnl for t in all_trades)
    total_costs = sum(t.cost_breakdown.total_tax_and_charges for t in all_trades)
    total_net = sum(t.net_pnl for t in all_trades)
    wins = sum(1 for t in all_trades if t.net_pnl > 0)
    win_rate = (wins / max(1, len(all_trades))) * 100.0
    net_pnls = [t.net_pnl for t in all_trades]
    net_pf = calculate_profit_factor(net_pnls)

    return {
        "alpha_id": alpha_id,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "trades": len(all_trades),
        "wins": wins,
        "losses": len(all_trades) - wins,
        "win_rate_pct": round(win_rate, 1),
        "gross_pnl": round(total_gross, 2),
        "total_costs": round(total_costs, 2),
        "net_pnl": round(total_net, 2),
        "net_profit_factor": round(net_pf, 2),
        "net_roi_pct": round((total_net / capital) * 100.0, 2),
        "trade_list": all_trades,
    }


def run_period_backtest():
    args = parse_args()
    print("=" * 105)
    print(f"[*] ASHVA PRECISION PERIOD BACKTEST: {args.alpha_id.upper()} ({args.timeframe.upper()})")
    print(f"[*] Period: {args.start} to {args.end} | Capital: Rs {args.capital:,.0f}")
    print("=" * 105)

    res = evaluate_alpha_period(
        alpha_id=args.alpha_id,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        capital=args.capital,
        allocation=args.allocation,
    )

    all_trades = res.get("trade_list", [])
    print(f"\n[+] Total Trades Found in Period ({args.start} to {args.end}): {len(all_trades)}")
    print("-" * 105)
    print(f"{'#':<3} {'Symbol':<12} {'Side':<6} {'Entry Time':<17} {'Exit Time':<17} {'Entry':<9} {'Exit':<9} {'Gross':<10} {'Taxes':<9} {'Net PnL':<10} {'Reason':<11}")
    print("-" * 105)

    for idx, t in enumerate(all_trades, 1):
        gross = t.gross_pnl
        costs = t.cost_breakdown.total_tax_and_charges
        net = t.net_pnl
        print(
            f"{idx:<3} {t.symbol:<12} {t.side:<6} {str(t.entry_time)[:16]:<17} {str(t.exit_time)[:16]:<17} "
            f"{t.entry_price:<9.2f} {t.exit_price:<9.2f} {gross:+9.2f} {costs:8.2f} {net:+9.2f} {t.exit_reason:<11}"
        )

    print("-" * 105)
    print("\n" + "=" * 50)
    print("PERIOD PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Period Trades:      {res['trades']}")
    print(f"Winning Trades:           {res['wins']} ({res['win_rate_pct']}%)")
    print(f"Losing Trades:            {res['losses']}")
    print(f"Gross Trading P&L:        Rs {res['gross_pnl']:+,.2f}")
    print(f"Statutory Taxes & Costs:  Rs {res['total_costs']:,.2f}")
    print(f"Net Realized P&L:         Rs {res['net_pnl']:+,.2f}")
    print(f"Net Profit Factor:        {res['net_profit_factor']:.2f}")
    print(f"Period Net ROI:           {res['net_roi_pct']:+.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_period_backtest()