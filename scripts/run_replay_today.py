"""
Ashva Production Replay Trading Engine: Replay Execution for Recent Sessions
Executes qualified positive alphas across the liquid NIFTY universe using the production TradingEngine.
"""

import sys
import argparse
from pathlib import Path
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

# Strategy Registry
from src.strategies.alpha_78_double_inside_momentum import Alpha78DoubleInsideMomentum
from src.strategies.alpha_81_double_inside_2r_expansion import Alpha81DoubleInside2RExpansion
from src.strategies.alpha_82_double_inside_volume_shock import Alpha82DoubleInsideVolumeShock
from src.strategies.alpha_85_double_inside_225r_expansion import Alpha85DoubleInside225RExpansion
from src.strategies.alpha_70_double_inside_target_expansion import Alpha70DoubleInsideTargetExpansion
from src.strategies.alpha_73_inside_day_expansion import Alpha73InsideDayExpansion
from src.strategies.alpha_56_nr4_moderate_gap_shock import Alpha56NR4ModerateGapShock
from src.strategies.alpha_68_nr5_high_conviction_gap import Alpha68NR5HighConvictionGap
from src.strategies.alpha_67_ten_day_max_vol_gap import Alpha67TenDayMaxVolGap
from src.strategies.alpha_54_gap_marubozu_momentum import Alpha54GapMarubozuMomentum
from src.strategies.alpha_04_gap_and_go import Alpha04GapAndGo

parser = argparse.ArgumentParser(description="Ashva Replay Engine Runner")
parser.add_argument("--start-date", type=str, default="2026-08-01", help="Replay start date (YYYY-MM-DD)")
parser.add_argument("--end-date", type=str, default="2026-08-20", help="Replay end date (YYYY-MM-DD)")
args = parser.parse_args()

lake = DataLake(read_only=True)
cost_model = IndianCostModel(default_slippage_bps=3.0)

universe = [
    "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
    "BAJFINANCE", "MARUTI", "SUNPHARMA"
]

print("=" * 115)
print(f"[*] ASHVA PRODUCTION REPLAY ENGINE: EXECUTING QUALIFIED ALPHAS ({args.start_date} to {args.end_date})")
print(f"[*] Universe: {len(universe)} Liquid NIFTY Equities | Segment: Cash Intraday (15:15 IST Square-off)")
print("=" * 115)

qualified_models = [
    ("ALPHA_78_DOUBLE_INSIDE_MOMENTUM", Alpha78DoubleInsideMomentum, "STEP_RATCHET"),
    ("ALPHA_81_DOUBLE_INSIDE_2R_EXPANSION", Alpha81DoubleInside2RExpansion, "STEP_RATCHET"),
    ("ALPHA_82_DOUBLE_INSIDE_VOLUME_SHOCK", Alpha82DoubleInsideVolumeShock, "STEP_RATCHET"),
    ("ALPHA_85_DOUBLE_INSIDE_225R_EXPANSION", Alpha85DoubleInside225RExpansion, "STEP_RATCHET"),
    ("ALPHA_70_DOUBLE_INSIDE_TARGET_EXPANSION", Alpha70DoubleInsideTargetExpansion, "STEP_RATCHET"),
    ("ALPHA_73_INSIDE_DAY_EXPANSION", Alpha73InsideDayExpansion, "BREAK_EVEN"),
    ("ALPHA_56_NR4_MODERATE_GAP_SHOCK", Alpha56NR4ModerateGapShock, "STEP_RATCHET"),
    ("ALPHA_68_NR5_HIGH_CONVICTION_GAP", Alpha68NR5HighConvictionGap, "STEP_RATCHET"),
    ("ALPHA_67_TEN_DAY_MAX_VOL_GAP", Alpha67TenDayMaxVolGap, "BREAK_EVEN"),
    ("ALPHA_54_GAP_MARUBOZU_MOMENTUM", Alpha54GapMarubozuMomentum, "STEP_RATCHET"),
    ("ALPHA_04_GAP_AND_GO", Alpha04GapAndGo, "BREAK_EVEN"),
]

contracts = []
for name, cls_ref, trailing in qualified_models:
    c = QualifiedAlphaContract(
        alpha_id=name,
        strategy_class=cls_ref,
        universe=universe,
        timeframe="15m",
        trailing_mode=trailing,
        risk_per_trade_pct=0.0050,
        max_capital_allocation_pct=0.20,
    )
    contracts.append(c)

print(f"[+] Configured {len(contracts)} Qualified Alpha Contracts in TradingEngine.")

replay_provider = ReplayMarketDataProvider(data_lake=lake, start_date=args.start_date, end_date=args.end_date)
replay_provider.subscribe(universe, "15m")

replay_adapter = ReplayExecutionAdapter(cost_model=cost_model, segment=Segment.EQUITY_INTRADAY)

trading_engine = TradingEngine(
    market_data_provider=replay_provider,
    execution_adapter=replay_adapter,
    alpha_contracts=contracts,
    initial_capital=500000.0,
    cost_model=cost_model,
)

print(f"[+] Starting event replay stream from {args.start_date} to {args.end_date}...")
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
