"""
Ashva Continuous Autonomous Alpha Discovery & Portfolio Factory
Multi-hour iterative quantitative discovery, validation, and performance tracking
for Indian Intraday Cash Equities using the Dynamic Strategy Registry.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.registry import get_all_strategies
from src.research.cpcv_engine import CPCVEngine
from src.research.regime_profiler import MarketRegimeProfiler
from scripts.ingest_all_nifty50_timeframes import NIFTY_50_UNIVERSE


def run_continuous_factory(
    timeframe: str = "15m",
    trailing_mode: str = "BREAK_EVEN",
    initial_capital: float = 500000.0,
    symbols: List[str] = NIFTY_50_UNIVERSE,
):
    print("=" * 110)
    print("      ASHVA QUANTITATIVE RESEARCH FACTORY: AUTONOMOUS ALPHA DISCOVERY & VALIDATION")
    print(f"[*] Universe: {len(symbols)} Equities | Timeframe: {timeframe} | Trailing: {trailing_mode}")
    print("=" * 110)

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel()
    cpcv_engine = CPCVEngine(n_partitions=6, k_test_partitions=2)

    # 1. Dynamic Auto-Discovery (Plug & Play)
    strategies = get_all_strategies(reload=True)
    print(f"[+] Dynamically discovered {len(strategies)} Alpha Strategies from src/strategies/\n")

    champion_roster = []

    for idx, (strat_name, strat_cls) in enumerate(strategies.items(), 1):
        strat = strat_cls()
        total_trades = 0
        gross_pnl = 0.0
        net_pnl = 0.0
        wins = 0
        gross_win = 0.0
        gross_loss = 0.0
        all_trade_records = []

        print(f"[{idx:02d}/{len(strategies)}] Evaluating {strat_name:35s}...", end="", flush=True)

        for sym in symbols:
            df = lake.load_bars(sym.upper(), timeframe)
            if df.empty or len(df) < 500:
                continue

            if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            # Polymorphic Signal Generation (Zero hardcoding)
            try:
                df_signals = strat.generate_signals(df)
            except Exception as e:
                continue

            if "signal" not in df_signals.columns:
                continue

            engine = BacktestEngine(cost_model=cost_model, initial_capital=initial_capital, segment=Segment.EQUITY_INTRADAY)
            res = engine.run(df_signals, symbol=sym.upper(), strategy_id=strat_name, trailing_mode=trailing_mode)

            total_trades += res.total_trades
            net_pnl += res.total_net_pnl
            wins += res.winning_trades
            gross_win += sum([t.net_pnl for t in res.trade_list if t.net_pnl > 0])
            gross_loss += sum([abs(t.net_pnl) for t in res.trade_list if t.net_pnl < 0])

            for t in res.trade_list:
                all_trade_records.append({
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "net_pnl": t.net_pnl,
                })

        if total_trades < 15 or net_pnl <= 0:
            pnl_str = f"Rs {net_pnl:,.2f}" if total_trades > 0 else "No Trades"
            print(f" [REJECT: Net PnL={pnl_str}]")
            continue

        win_rate = (wins / total_trades) * 100.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 99.0
        df_trades_all = pd.DataFrame(all_trade_records)

        # CPCV & PBO Verification
        cpcv_res = cpcv_engine.evaluate_trades(df_trades_all)
        pbo = cpcv_res.get("pbo", 1.0)
        oos_sharpe = cpcv_res.get("mean_oos_sharpe", 0.0)

        if not cpcv_res.get("is_overfitted", True) and oos_sharpe >= 0.50 and pbo <= 0.30:
            print(f" [CHAMPION: Net PnL=+Rs {net_pnl:,.2f}, OOS Sharpe={oos_sharpe:.2f}, PBO={cpcv_res['pbo_pct']}]")
            champion_roster.append({
                "name": strat_name,
                "net_pnl": round(net_pnl, 2),
                "win_rate": round(win_rate, 1),
                "pf": round(profit_factor, 2),
                "oos_sharpe": oos_sharpe,
                "pbo": cpcv_res["pbo_pct"],
                "trades": total_trades,
            })
        else:
            print(f" [OVERFIT: Net PnL=+Rs {net_pnl:,.2f}, PBO={cpcv_res.get('pbo_pct', '100%')}, OOS Sharpe={oos_sharpe:.2f}]")

    print("\n" + "=" * 110)
    print(f"               FACTORY SUMMARY: {len(champion_roster)} CERTIFIED CHAMPIONS DISCOVERED")
    print("=" * 110)
    if champion_roster:
        df_champs = pd.DataFrame(champion_roster)
        print(df_champs.to_string(index=False))
    print("=" * 110 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashva Continuous Autonomous Alpha Factory")
    parser.add_argument("--timeframe", type=str, default="15m", help="Target timeframe (default: 15m)")
    parser.add_argument("--trailing", type=str, default="BREAK_EVEN", help="Trailing mode (default: BREAK_EVEN)")
    parser.add_argument("--capital", type=float, default=500000.0, help="Initial capital base (default: 500000)")
    args = parser.parse_args()

    run_continuous_factory(
        timeframe=args.timeframe,
        trailing_mode=args.trailing,
        initial_capital=args.capital,
    )
