"""
Ashva Production Replay Trading Engine: Replay Execution for Target Sessions
Executes all qualified positive alphas across the liquid NIFTY universe using the production TradingEngine.
"""

import sys
import argparse
from pathlib import Path
from datetime import time
from typing import Dict, List, Any
import pandas as pd
import numpy as np

sys.path.append(str(Path.cwd()))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.trading.contract import QualifiedAlphaContract
from src.trading.engine import TradingEngine
from src.market_data.replay_provider import ReplayMarketDataProvider
from src.execution.replay_adapter import ReplayExecutionAdapter

from src.strategies.registry import get_all_strategies, get_strategy_by_name
from src.ui.data_access import UIDataAccess

from src.core.universe_manager import get_universe_symbols

parser = argparse.ArgumentParser(description="Ashva Replay Engine Runner")
parser.add_argument("--start-date", type=str, default="2026-08-15", help="Replay start date (YYYY-MM-DD)")
parser.add_argument("--end-date", type=str, default="2026-09-16", help="Replay end date (YYYY-MM-DD)")
parser.add_argument("--universe", type=str, default="ALL_77", help="ALL_77, ALL, or NIFTY_14")
args = parser.parse_args()

lake = DataLake(read_only=True)
cost_model = IndianCostModel(default_slippage_bps=3.0)
dal = UIDataAccess()

if args.universe in ("ALL_77", "ALL", "ALL_50"):
    universe = get_universe_symbols()
else:
    universe = [
        "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
        "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
        "BAJFINANCE", "MARUTI", "SUNPHARMA"
    ]

# Format exact timestamp boundaries to capture all intraday bars
s_date = f"{args.start_date} 00:00:00" if len(args.start_date) == 10 else args.start_date
e_date = f"{args.end_date} 23:59:59" if len(args.end_date) == 10 else args.end_date

# Dynamically discover all PROVEN alphas from system registry
all_alphas_df = dal.get_alpha_registry_table()
proven_df = all_alphas_df[all_alphas_df["status"] == "PROVEN"]
proven_ids = proven_df["alpha_id"].tolist()

print("=" * 115)
print(f"[*] ASHVA PRODUCTION REPLAY ENGINE: EXECUTING {len(proven_ids)} QUALIFIED PROVEN ALPHAS ({s_date} to {e_date})")
print(f"[*] Universe: {len(universe)} NIFTY Equities | Segment: Cash Intraday (15:15 IST Square-off)")
print(f"[*] Qualified Alphas: {', '.join(proven_ids)}")
print("=" * 115)

contracts = []
for strat_id in proven_ids:
    cls_ref = get_strategy_by_name(strat_id)
    if not cls_ref:
        continue
    detail = dal.get_alpha_detail(strat_id)
    strat_tf = detail.get("timeframe", "15m")
    
    c = QualifiedAlphaContract(
        alpha_id=strat_id,
        strategy_class=cls_ref,
        universe=universe,
        timeframe=strat_tf,
        entry_start_time=time(9, 15),
        entry_end_time=time(15, 0),
        trailing_mode="NONE",
        risk_per_trade_pct=0.0050,
        max_capital_allocation_pct=0.20,
    )
    contracts.append(c)

print(f"[+] Configured {len(contracts)} Qualified Alpha Contracts in TradingEngine.")

replay_provider = ReplayMarketDataProvider(data_lake=lake, start_date=s_date, end_date=e_date)
replay_provider.subscribe(universe, "15m")

replay_adapter = ReplayExecutionAdapter(cost_model=cost_model, segment=Segment.EQUITY_INTRADAY)

trading_engine = TradingEngine(
    market_data_provider=replay_provider,
    execution_adapter=replay_adapter,
    alpha_contracts=contracts,
    initial_capital=500000.0,
    cost_model=cost_model,
)

print(f"[+] Starting event replay stream from {s_date} to {e_date}...")
summary = trading_engine.run()

total_trades = summary.get("total_trades", 0)
gross_pnl = summary.get("realized_gross_pnl", summary.get("gross_pnl", 0.0))
net_pnl = summary.get("realized_pnl", summary.get("net_pnl", 0.0))
ending_eq = summary.get("current_capital", 500000.0)
win_rate = summary.get("win_rate_pct", 0.0)
closed_trades = trading_engine.position_manager.closed_trades

print("\n" + "=" * 115)
print(f"[*] REPLAY ENGINE EXECUTION SUMMARY ({args.start_date} to {args.end_date})")
print("=" * 115)
print(f"Total Portfolio Trades Executed: {total_trades}")
print(f"Total Gross Realized PnL:        Rs {gross_pnl:+10,.2f}")
print(f"Total Net Realized PnL:          Rs {net_pnl:+10,.2f}")
print(f"Ending Portfolio Equity:         Rs {ending_eq:10,.2f}")
print(f"Win Rate:                        {win_rate:5.1f}%")

if closed_trades:
    print("\n" + "-" * 115)
    print("DETAILED EXECUTED TRADE LOG:")
    print("-" * 115)
    t_df = pd.DataFrame([{
        "Trade ID": t.get("trade_id"),
        "Alpha ID": t.get("alpha_id"),
        "Symbol": t.get("symbol"),
        "Side": t.get("side"),
        "Entry Time": str(t.get("entry_time")),
        "Exit Time": str(t.get("exit_time")),
        "Entry Price": f"Rs {t.get('entry_price', 0.0):.2f}",
        "Exit Price": f"Rs {t.get('exit_price', 0.0):.2f}",
        "Qty": t.get("quantity"),
        "Gross PnL": f"Rs {t.get('gross_pnl', 0.0):+.2f}",
        "Net PnL": f"Rs {t.get('net_pnl', 0.0):+.2f}",
        "Exit Reason": t.get("exit_reason")
    } for t in closed_trades])
    print(t_df.to_string(index=False))
else:
    print("\n[-] No triggers were generated across these dates.")

print("=" * 115)
