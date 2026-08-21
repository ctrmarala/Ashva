"""
Ashva Zero-Code Monthly & Custom Period Performance Inspector
Allows instant querying of portfolio and alpha performance for any month, quarter,
year, or custom date range with ZERO code modification.

Usage Examples:
  python scripts/inspect_period.py --month 2026-07
  python scripts/inspect_period.py --month 2025-07
  python scripts/inspect_period.py --start 2026-01-01 --end 2026-06-30
  python scripts/inspect_period.py --year 2026
  python scripts/inspect_period.py --monthly-table
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from src.portfolio.master_portfolio_backtester import MasterPortfolioBacktester
from scripts.ingest_all_nifty50_timeframes import NIFTY_50_UNIVERSE
import importlib


def load_production_model_pack() -> Dict[str, Any]:
    pack_file = Path("config/production_model_pack.json")
    if not pack_file.exists():
        raise FileNotFoundError("config/production_model_pack.json not found. Run production sweep first.")
    with open(pack_file, "r") as f:
        return json.load(f)


def get_strategy_instances(champion_alphas: List[Dict[str, Any]]):
    strategies = []
    tf_map = {}
    trail_map = {}

    strat_dir = Path("src/strategies")
    for champ in champion_alphas:
        name = champ["strategy_name"]
        tf = champ["optimal_timeframe"]
        mode = champ["optimal_trailing_mode"]
        tf_map[name] = tf
        trail_map[name] = mode

        # Find file
        for p in strat_dir.glob("alpha_*.py"):
            mod_name = f"src.strategies.{p.stem}"
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, name):
                    cls = getattr(mod, name)
                    strategies.append(cls)
                    break
            except Exception:
                pass
    return strategies, tf_map, trail_map


def run_inspection(
    month: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
    show_monthly_table: bool = False,
    initial_capital: float = 500000.0,
):
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel()
    model_pack = load_production_model_pack()
    champions = model_pack.get("champion_alphas", [])

    strategies, tf_map, trail_map = get_strategy_instances(champions)

    tester = MasterPortfolioBacktester(
        data_lake=lake,
        initial_capital=initial_capital,
        risk_per_trade_inr=2500.0,
        max_concurrent_positions=5,
        max_positions_per_sector=2,
        cost_model=cost_model,
    )

    res = tester.run_portfolio_backtest(
        strategies=strategies,
        symbols=NIFTY_50_UNIVERSE,
        strategy_timeframe_map=tf_map,
        strategy_trailing_map=trail_map,
        use_regime_filter=True,
    )

    trades = res["trade_list"]
    df_trades = pd.DataFrame([{
        "trade_id": t.trade_id,
        "strategy": t.strategy_id,
        "symbol": t.symbol,
        "sector": t.sector,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "side": t.side,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "qty": t.quantity,
        "gross_pnl": t.gross_pnl,
        "net_pnl": t.net_pnl,
        "costs": t.total_costs,
        "exit_reason": t.exit_reason,
    } for t in trades])

    if df_trades.empty:
        print("[!] No trades generated in backtest.")
        return

    df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"])
    df_trades["month_str"] = df_trades["entry_time"].dt.strftime("%Y-%m")
    df_trades["year"] = df_trades["entry_time"].dt.year

    # 1. Show Complete Month-by-Month Calendar Table if requested
    if show_monthly_table:
        print("\n" + "=" * 90)
        print("                  ASHVA MASTER PORTFOLIO: MONTH-BY-MONTH CALENDAR TRACK RECORD")
        print("=" * 90)
        monthly_summary = []
        for m_str, grp in df_trades.groupby("month_str"):
            n_t = len(grp)
            n_w = (grp["net_pnl"] > 0).sum()
            wr = (n_w / n_t) * 100.0 if n_t > 0 else 0.0
            pnl = grp["net_pnl"].sum()
            gross_win = grp.loc[grp["net_pnl"] > 0, "net_pnl"].sum()
            gross_loss = abs(grp.loc[grp["net_pnl"] < 0, "net_pnl"].sum())
            pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
            roi = (pnl / initial_capital) * 100.0

            monthly_summary.append({
                "Month": m_str,
                "Trades": n_t,
                "Win Rate": f"{wr:.1f}%",
                "Profit Factor": f"{pf:.2f}",
                "Net PnL (Rs)": f"{pnl:+10,.2f}",
                "Net ROI": f"{roi:+.2f}%",
            })
        print(pd.DataFrame(monthly_summary).to_string(index=False))
        print("=" * 90)
        return

    # 2. Filter by specific period
    filtered_df = df_trades.copy()
    period_title = "Full Backtest Horizon"

    if month:
        filtered_df = filtered_df[filtered_df["month_str"] == month]
        period_title = f"Month of {month}"
    elif year:
        filtered_df = filtered_df[filtered_df["year"] == year]
        period_title = f"Year {year}"
    elif start_date or end_date:
        if start_date:
            filtered_df = filtered_df[filtered_df["entry_time"] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df["entry_time"] <= end_date]
        period_title = f"Period: {start_date or 'Start'} to {end_date or 'End'}"

    print("\n" + "=" * 90)
    print(f"               ASHVA PERFORMANCE REPORT: {period_title.upper()}")
    print("=" * 90)

    if filtered_df.empty:
        print(f"[!] No trades occurred during {period_title}.")
        print("=" * 90)
        return

    n_trades = len(filtered_df)
    n_wins = (filtered_df["net_pnl"] > 0).sum()
    n_losses = (filtered_df["net_pnl"] < 0).sum()
    win_rate = (n_wins / n_trades) * 100.0
    total_net_pnl = filtered_df["net_pnl"].sum()
    total_costs = filtered_df["costs"].sum()
    gross_pnl = filtered_df["gross_pnl"].sum()
    roi = (total_net_pnl / initial_capital) * 100.0

    gross_win = filtered_df.loc[filtered_df["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = abs(filtered_df.loc[filtered_df["net_pnl"] < 0, "net_pnl"].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    print(f"Period:                  {period_title}")
    print(f"Initial Capital Base:    Rs {initial_capital:,.2f}")
    print(f"Total Net PnL:           Rs {total_net_pnl:+,.2f} ({roi:+.2f}% ROI on Capital)")
    print(f"Gross PnL / Total Taxes: Rs {gross_pnl:+,.2f} / Rs {total_costs:,.2f}")
    print(f"Total Executed Trades:   {n_trades} ({n_wins} Wins / {n_losses} Losses)")
    print(f"Win Rate:                {win_rate:.1f}%")
    print(f"Profit Factor:           {pf:.2f}")
    print("-" * 90)

    print("Alpha Breakdown:")
    for strat_name, grp in filtered_df.groupby("strategy"):
        s_pnl = grp["net_pnl"].sum()
        s_wins = (grp["net_pnl"] > 0).sum()
        s_wr = (s_wins / len(grp)) * 100.0
        print(f" - {strat_name:32s}: Rs {s_pnl:+9,.2f} | Win Rate: {s_wr:5.1f}% ({s_wins}/{len(grp)} trades)")

    print("-" * 90)
    print("Top Asset Contributors:")
    for sym, grp in sorted(filtered_df.groupby("symbol"), key=lambda x: x[1]["net_pnl"].sum(), reverse=True)[:6]:
        print(f" - {sym:12s}: Rs {grp['net_pnl'].sum():+9,.2f} ({len(grp)} trades)")

    print("\nDetailed Trade Log:")
    display_cols = ["entry_time", "strategy", "symbol", "side", "qty", "net_pnl", "exit_reason"]
    print(filtered_df[display_cols].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashva Zero-Code Monthly & Custom Period Inspector")
    parser.add_argument("--month", type=str, help="Target month in YYYY-MM format (e.g. 2026-07)")
    parser.add_argument("--start", type=str, help="Start date in YYYY-MM-DD format (e.g. 2026-07-01)")
    parser.add_argument("--end", type=str, help="End date in YYYY-MM-DD format (e.g. 2026-07-31)")
    parser.add_argument("--year", type=int, help="Target year (e.g. 2026)")
    parser.add_argument("--monthly-table", action="store_true", help="Print full month-by-month calendar table")
    parser.add_argument("--capital", type=float, default=500000.0, help="Initial Capital Base (default: 500000)")

    args = parser.parse_args()

    # Default to July 2026 if no arguments passed
    if not any([args.month, args.start, args.end, args.year, args.monthly_table]):
        args.month = "2026-07"

    run_inspection(
        month=args.month,
        start_date=args.start,
        end_date=args.end,
        year=args.year,
        show_monthly_table=args.monthly_table,
        initial_capital=args.capital,
    )
