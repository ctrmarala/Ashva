"""
Ashva Nifty 50 Universe Downloader & Multi-Alpha Quantitative Scanner
Downloads and evaluates all 50 liquid Nifty F&O equities against Ashva's verified alpha engines.
"""

import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.backtest.engine import BacktestEngine
from src.analytics.indian_costs import IndianCostModel, Segment
from src.strategies.alpha_orb_pro import AlphaAuctionORBPro
from src.strategies.alpha_04_gap_and_go import Alpha04GapAndGo
from src.strategies.alpha_14_gap_momentum_drift import Alpha14GapMomentumDrift
from src.strategies.alpha_18_three_day_trend_orb import Alpha18ThreeDayTrendORB
from src.core.universe_manager import get_universe_symbols, get_universe_name

NIFTY_50_UNIVERSE = get_universe_symbols()
TARGET_UNIVERSE = NIFTY_50_UNIVERSE


def sync_nifty_50(data_lake: DataLake):
    print("=" * 80)
    print(f"[*] VERIFYING UNIVERSE DATA ({len(TARGET_UNIVERSE)} ASSETS) IN DATA LAKE")
    print("=" * 80)

    for i, sym in enumerate(TARGET_UNIVERSE, 1):
        existing = data_lake.load_bars(sym, "15m")
        if not existing.empty:
            print(f"[{i:02d}/{len(TARGET_UNIVERSE)}] {sym:12s} -> Ready ({len(existing)} bars)")
        else:
            print(f"[{i:02d}/{len(TARGET_UNIVERSE)}] {sym:12s} -> [!] Missing in DataLake (Sync via Angel One SmartAPI)")


def scan_universe_alphas():
    data_lake = DataLake(read_only=False)
    sync_nifty_50(data_lake)

    print("\n" + "=" * 100)
    print("[*] EXECUTING MULTI-ALPHA SCANNER ACROSS NIFTY 50 UNIVERSE")
    print("=" * 100)

    cost_model = IndianCostModel(slippage_bps=3.0)
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)

    strategies = {
        "ALPHA_14 (Gap Drift)": Alpha14GapMomentumDrift(),
        "ALPHA_02 (ORB Pro)": AlphaAuctionORBPro(),
        "ALPHA_18 (3D Trend)": Alpha18ThreeDayTrendORB(),
        "ALPHA_04 (Gap & Go)": Alpha04GapAndGo(),
    }

    all_results = []

    for strat_name, strat_obj in strategies.items():
        print(f"\n[>] Scanning {strat_name} across all 50 stocks...")
        for sym in NIFTY_50_UNIVERSE:
            df = data_lake.load_bars(sym, "15m")
            if df.empty or len(df) < 100:
                continue

            try:
                signals_df = strat_obj.generate_signals(df)
                res = engine.run(signals_df, symbol=sym, strategy_id=strat_name, risk_per_trade_pct=0.015, capital_per_trade_pct=0.25)

                if res.total_trades > 0:
                    all_results.append({
                        "Strategy": strat_name,
                        "Symbol": sym,
                        "Net_PnL": res.total_net_pnl,
                        "Net_ROI_Pct": res.net_roi_pct,
                        "Trades": res.total_trades,
                        "Win_Rate": res.win_rate_pct,
                        "Net_PF": res.net_profit_factor if res.net_profit_factor < 90 else 99.0,
                        "Sharpe": res.sharpe_ratio,
                        "Max_DD_Pct": res.max_drawdown_pct,
                        "Taxes_Paid": res.total_taxes_paid,
                    })
            except Exception as e:
                pass

    res_df = pd.DataFrame(all_results)
    if res_df.empty:
        print("[!] No trades generated.")
        return

    # Filter for Positive Alpha Champions (Net PnL > 0)
    winners = res_df[res_df["Net_PnL"] > 0].sort_values("Net_PnL", ascending=False)

    print("\n" + "=" * 120)
    print(f"[*] NIFTY 50 UNIVERSE SCAN RESULTS: {len(winners)} POSITIVE ALPHA LEADERS DISCOVERED")
    print("=" * 120)
    print(f"{'Strategy':25s} {'Symbol':12s} {'Net PnL (INR)':15s} {'ROI %':10s} {'Trades':8s} {'Win Rate':10s} {'Net PF':8s} {'Sharpe':8s} {'MaxDD %':8s}")
    print("-" * 120)

    for _, row in winners.iterrows():
        print(
            f"{row['Strategy']:25s} {row['Symbol']:12s} "
            f"Rs {row['Net_PnL']:+10.2f}    "
            f"{row['Net_ROI_Pct']:+6.2f}%    "
            f"{int(row['Trades']):<8d} "
            f"{row['Win_Rate']:6.1f}%    "
            f"{row['Net_PF']:6.2f}   "
            f"{row['Sharpe']:+6.2f}   "
            f"{row['Max_DD_Pct']:6.2f}%"
        )

    print("-" * 120)
    total_winner_pnl = winners["Net_PnL"].sum()
    total_winner_trades = winners["Trades"].sum()
    print(f"[*] COMBINED POSITIVE BASKET: Total Net P&L = Rs +{total_winner_pnl:,.2f} | Total Trades = {int(total_winner_trades)} trades")
    print("=" * 120)


if __name__ == "__main__":
    scan_universe_alphas()
