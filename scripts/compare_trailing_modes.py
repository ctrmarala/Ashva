"""
Ashva Quantitative Strategy Benchmark: Static SL/TP vs Break-Even vs Tiered Step-Ratchet
Simulates all 15 champion alphas across 14 NIFTY blue chips over 540 days (420d IS / 120d OOS)
under full statutory Indian taxes and 3.0 bps slippage.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine, BacktestTrade
from scripts.run_hypothesis_lab import STRATEGY_MAP, DEFAULT_UNIVERSE

lake = DataLake(read_only=True)
cost_model = IndianCostModel(default_slippage_bps=3.0)

CHAMPION_ALPHAS = [
    "alpha_37",  # Gap Volume Shock Drift (RVOL >= 2.0x, RR 1.5)
    "alpha_74",  # Gap Marubozu Momentum (Body >= 70%, RVOL >= 1.25x, RR 1.5)
    "alpha_71",  # Outlier Volume Drive (Vol >= 1.2x 10d Max, Gap 0.35-1.2%, RR 1.5)
    "alpha_79",  # High RR Opening Drive (RVOL >= 1.5x, RR 2.0)
    "alpha_46",  # High Conviction Gap Drift (RVOL >= 1.75x, RR 1.5)
    "alpha_49",  # Moderate Gap Volume Shock (RVOL >= 1.5x, RR 1.5)
    "alpha_67",  # Ten Day Max Vol Gap (10d Volume Record, RR 1.5)
    "alpha_72",  # NR3 Range Contraction Volatility Expansion (RVOL >= 1.5x, RR 2.0)
    "alpha_73",  # Inside Day Range Expansion (RVOL >= 1.2x, RR 1.5)
    "alpha_75",  # Morning Drive Momentum Continuation (Gap >= 0.35%, RVOL >= 1.2x, Body >= 60%)
    "alpha_14",  # Gap Momentum Drift (Baseline Champion, RR 1.5)
    "alpha_59",  # NR6 Gap Breakout (NR6 + Breakout RR 1.5)
    "alpha_66",  # Two-Day Trend High Vol Gap (2G/2R + High Vol Gap)
    "alpha_56",  # NR4 Moderate Gap Shock (NR4 + Volume Shock RR 1.5)
    "alpha_52",  # NR5 Gap Breakout (NR5 + Breakout RR 1.5)
]

symbols = DEFAULT_UNIVERSE
lookback_days = 540
capital_per_asset = 500000.0

print("=" * 130)
print("[*] RUNNING COMPREHENSIVE BENCHMARK: STATIC vs BREAK-EVEN vs STEP-RATCHET TRAILING")
print(f"[*] Champion Alphas: {len(CHAMPION_ALPHAS)} Alphas | Universe: {len(symbols)} Blue Chips | Lookback: {lookback_days}d")
print("=" * 130)

modes = ["NONE", "BREAK_EVEN", "STEP_RATCHET"]
mode_summaries = {}

for mode in modes:
    print(f"\n[>] Simulating Mode: {mode} across {len(CHAMPION_ALPHAS)} alphas...", end="", flush=True)
    all_trades: List[BacktestTrade] = []
    total_net = 0.0
    total_gross = 0.0
    total_taxes = 0.0
    exit_reasons = {}

    for strat_id in CHAMPION_ALPHAS:
        strat_name, strat_cls = STRATEGY_MAP[strat_id]
        strat_obj = strat_cls()

        for sym in symbols:
            df = lake.load_bars(sym, "15m", max_lookback_days=lookback_days)
            if df.empty or len(df) < 50:
                continue

            sig_df = strat_obj.generate_signals(df)
            eng = BacktestEngine(cost_model=cost_model, initial_capital=capital_per_asset, segment=Segment.EQUITY_INTRADAY)
            res = eng.run(
                sig_df,
                symbol=sym,
                strategy_id=strat_id,
                risk_per_trade_pct=0.005,
                capital_per_trade_pct=0.25,
                trailing_mode=mode,
            )

            for t in res.trade_list:
                all_trades.append(t)
                total_gross += t.gross_pnl
                total_net += t.net_pnl
                total_taxes += (t.gross_pnl - t.net_pnl)
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    n_trades = len(all_trades)
    wins = [t for t in all_trades if t.net_pnl > 0]
    losses = [t for t in all_trades if t.net_pnl <= 0]
    wr = (len(wins) / max(1, n_trades)) * 100.0
    net_w = sum(t.net_pnl for t in wins)
    net_l = abs(sum(t.net_pnl for t in losses))
    pf = (net_w / net_l) if net_l > 0 else (99.0 if net_w > 0 else 0.0)

    rets = [t.net_pnl / 125000.0 for t in all_trades]
    std = np.std(rets) if rets else 0.0
    sharpe = (np.mean(rets) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0

    # Max Drawdown
    equity = 7000000.0
    peak = equity
    max_dd = 0.0
    # Sort trades chronologically
    sorted_trades = sorted(all_trades, key=lambda t: t.exit_time)
    for t in sorted_trades:
        equity += t.net_pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd

    mode_summaries[mode] = {
        "Mode": mode,
        "Total_Trades": n_trades,
        "Gross_PnL_INR": round(total_gross, 2),
        "Taxes_Paid_INR": round(total_taxes, 2),
        "Net_PnL_INR": round(total_net, 2),
        "Win_Rate_Pct": round(wr, 1),
        "Profit_Factor": round(pf, 2),
        "Sharpe_Ratio": round(sharpe, 2),
        "Max_Drawdown_Pct": round(max_dd, 2),
        "Exit_Reasons": exit_reasons,
    }
    print(f" [DONE: Net PnL = Rs {total_net:+,.0f}, WR = {wr:.1f}%, PF = {pf:.2f}, Sharpe = {sharpe:+.2f}, MaxDD = {max_dd:.2f}%]")

df_comp = pd.DataFrame([
    {
        "Execution_Mode": s["Mode"],
        "Total_Trades": s["Total_Trades"],
        "Net_PnL_INR": f"Rs {s['Net_PnL_INR']:+12,.2f}",
        "Win_Rate_Pct": f"{s['Win_Rate_Pct']:.1f}%",
        "Profit_Factor": f"{s['Profit_Factor']:.2f}",
        "Sharpe": f"{s['Sharpe_Ratio']:+.2f}",
        "Max_Drawdown_Pct": f"{s['Max_Drawdown_Pct']:.2f}%",
        "Exit_Reasons_Breakdown": str(s["Exit_Reasons"]),
    }
    for s in mode_summaries.values()
])

print("\n" + "=" * 130)
print("[*] COMPARATIVE PERFORMANCE BENCHMARK (15 CHAMPION ALPHAS ACROSS 540 DAYS):")
print("=" * 130)
print(df_comp.to_string(index=False))

df_comp.to_csv("trailing_mode_comparison_benchmark.csv", index=False)
print("\n[+] Saved comparative benchmark to trailing_mode_comparison_benchmark.csv")
