"""
Ashva Master Multi-Alpha Ensemble Portfolio Simulation & Monthly ROI Tearsheet
Executes full 540-day BacktestEngine evaluation for our champion alphas,
merges trade streams into MultiAlphaPortfolioEngine, and computes monthly ROI distributions.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine, BacktestTrade
from src.analytics.portfolio_engine import MultiAlphaPortfolioEngine
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha
from scripts.run_hypothesis_lab import STRATEGY_MAP, DEFAULT_UNIVERSE

lake = DataLake(read_only=True)
cost_model = IndianCostModel(default_slippage_bps=3.0)
ledger = ResearchExperimentLedger()
git_sha = get_current_git_sha()

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
oos_days = 120
capital_per_asset = 500000.0
total_basket_capital = capital_per_asset * len(symbols)

print("=" * 130)
print(f"[*] RUNNING MASTER MULTI-ALPHA ENSEMBLE PORTFOLIO SIMULATION")
print(f"[*] Champion Alphas: {', '.join(CHAMPION_ALPHAS)}")
print(f"[*] Universe: {len(symbols)} Liquid Blue Chips | Lookback: {lookback_days}d (IS: {lookback_days - oos_days}d, OOS: {oos_days}d)")
print(f"[*] Total Portfolio Capital: Rs {total_basket_capital:,.0f} | Execution: Strict Next-Open Fill + Indian Statutory Costs")
print("=" * 130)

strategy_all_trades = {}
individual_summaries = []

for strat_id in CHAMPION_ALPHAS:
    if strat_id not in STRATEGY_MAP:
        continue
    strat_name, strat_cls = STRATEGY_MAP[strat_id]
    strat_obj = strat_cls()

    trades_list: List[BacktestTrade] = []
    total_net = 0.0
    total_gross = 0.0
    total_taxes = 0.0
    oos_net = 0.0
    oos_trades_cnt = 0
    oos_wins_cnt = 0
    pos_assets = 0

    max_dt = None

    for sym in symbols:
        df = lake.load_bars(sym, "15m", max_lookback_days=lookback_days)
        if df.empty or len(df) < 50:
            continue

        sig_df = strat_obj.generate_signals(df)
        eng = BacktestEngine(cost_model=cost_model, initial_capital=capital_per_asset, segment=Segment.EQUITY_INTRADAY)
        res = eng.run(sig_df, symbol=sym, strategy_id=strat_id, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)

        if max_dt is None and len(df) > 0:
            max_dt = pd.to_datetime(df.index[-1])

        if res.total_trades > 0:
            if res.total_net_pnl > 0:
                pos_assets += 1
            for t in res.trade_list:
                trades_list.append(t)
                total_gross += t.gross_pnl
                total_net += t.net_pnl
                total_taxes += (t.gross_pnl - t.net_pnl)

    strategy_all_trades[strat_name] = trades_list

    # Separate OOS
    if max_dt:
        oos_cutoff = (max_dt - pd.Timedelta(days=oos_days)).date()
        for t in trades_list:
            if pd.to_datetime(t.entry_time).date() >= oos_cutoff:
                oos_trades_cnt += 1
                oos_net += t.net_pnl
                if t.net_pnl > 0:
                    oos_wins_cnt += 1

    n_t = len(trades_list)
    wins = [t for t in trades_list if t.net_pnl > 0]
    losses = [t for t in trades_list if t.net_pnl <= 0]
    wr = (len(wins) / max(1, n_t)) * 100.0
    net_w = sum(t.net_pnl for t in wins)
    net_l = abs(sum(t.net_pnl for t in losses))
    pf = (net_w / net_l) if net_l > 0 else (99.0 if net_w > 0 else 0.0)

    rets = [t.net_pnl / 125000.0 for t in trades_list]
    std = np.std(rets) if rets else 0.0
    sharpe = (np.mean(rets) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0

    oos_wr = (oos_wins_cnt / max(1, oos_trades_cnt)) * 100.0 if oos_trades_cnt > 0 else 0.0
    avg_net_trade_pct = (total_net / (max(1, n_t) * 125000.0)) * 100.0

    individual_summaries.append({
        "Alpha_ID": strat_id,
        "Strategy_Name": strat_name,
        "540d_Net_PnL": round(total_net, 2),
        "540d_Trades": n_t,
        "Win_Rate_Pct": round(wr, 1),
        "Profit_Factor": round(pf, 2),
        "Sharpe": round(sharpe, 2),
        "120d_OOS_PnL": round(oos_net, 2),
        "120d_OOS_Trades": oos_trades_cnt,
        "120d_OOS_WR": round(oos_wr, 1),
        "Positive_Assets": f"{pos_assets}/{len(symbols)}",
        "Avg_Net_Trade_Pct": round(avg_net_trade_pct, 2),
    })

df_ind = pd.DataFrame(individual_summaries)
print("\n" + "=" * 130)
print("[*] INDIVIDUAL CHAMPION ALPHA SCORECARDS (540d / 120d Untouched OOS):")
print("=" * 130)
print(df_ind.to_string(index=False))

# Run Multi-Alpha Portfolio Engine
port_engine = MultiAlphaPortfolioEngine(initial_capital=total_basket_capital)
port_results = port_engine.evaluate_portfolio(strategy_all_trades, capital_per_trade=125000.0)

print("\n" + "=" * 130)
print("[*] MASTER MULTI-ALPHA ENSEMBLE PORTFOLIO PERFORMANCE SUMMARY:")
print("=" * 130)
print(f"  • Total Portfolio Trades Executed: {port_results['total_trades']:,} Trades across 14 Assets")
print(f"  • Total Gross PnL:                 Rs {port_results['total_gross_pnl']:+12,.2f}")
print(f"  • Total Indian Statutory Taxes:    Rs {port_results['total_taxes_paid']:12,.2f}")
print(f"  • TOTAL NET PROFIT (After Taxes):  Rs {port_results['total_net_pnl']:+12,.2f}")
print(f"  • Overall Net Win Rate:            {port_results['win_rate']:.1f}%")
print(f"  • Overall Net Profit Factor:       {port_results['profit_factor']:.2f}")
print(f"  • Annualized Portfolio Sharpe:     {port_results['annualized_sharpe']:.2f}")
print(f"  • Maximum Portfolio Drawdown:      Rs {port_results['max_drawdown_inr']:,.2f} ({port_results['max_drawdown_pct']:.2f}%)")
print(f"  • Average Monthly Net PnL:         Rs {port_results['avg_monthly_net_pnl']:+12,.2f}")
print(f"  • Average Monthly ROI (Active Cap):{port_results['avg_monthly_roi_active_pct']:+.2f}% / Month")
print("=" * 130)

print("\n[*] MONTH-BY-MONTH REALIZED ROI BREAKDOWN:")
print("=" * 130)
df_month = port_results["monthly_table"]
print(df_month.to_string(index=False))

df_ind.to_csv("master_champion_alphas_summary.csv", index=False)
df_month.to_csv("master_portfolio_monthly_roi.csv", index=False)
print("\n[+] Saved results to master_champion_alphas_summary.csv & master_portfolio_monthly_roi.csv")
