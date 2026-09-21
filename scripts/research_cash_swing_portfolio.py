"""
Ashva Cash Swing Portfolio Simulation: 17 Complete Months on Rs 5L Base Capital
================================================================================

Simulates Cash Swing (CNC/Overnight Long-Only holding) for 1-Day (Exit Next Open)
and 2-Day (Exit Day 2 Close) on Rs 5,00,000 base capital (4 slots @ Rs 1,25,000/slot).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from scripts.research_pcde_phase2_3 import load_candidates, evaluate_candidate_economics, compute_metrics


def run_cash_swing_portfolio():
    lake = DataLake(read_only=True)
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=5.0)  # 5 bps for swing
    slot_cap = 125000.0

    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]
    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")

    # Complete 17 months: 2025-04 to 2026-08
    complete_months = [m for m in sorted(df["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df = df[df["month_year"].isin(complete_months)].copy().reset_index(drop=True)

    # Filter to Long signals only (SEBI cash delivery holding)
    df_long = df[df["side"] == "LONG"].copy().reset_index(drop=True)

    # Pre-load daily bars
    symbols = sorted(df_long["symbol"].unique())
    daily_bars = {}
    for sym in symbols:
        d_df = lake.load_bars(sym, "1d", max_lookback_days=540)
        if not d_df.empty:
            daily_bars[sym] = d_df

    # Extract Swing Trade Outcomes
    swing_trades = []
    for idx, row in df_long.iterrows():
        sym = row["symbol"]
        entry_time = pd.to_datetime(row["entry_time"])
        entry_date = entry_time.date()
        entry_p = row["entry_price"]

        if sym not in daily_bars:
            continue
        d_df = daily_bars[sym]
        d_df_future = d_df[d_df.index.date > entry_date]

        if len(d_df_future) >= 2:
            d1_open = float(d_df_future.iloc[0]["open"])
            d1_date = d_df_future.index[0].date()
            d2_close = float(d_df_future.iloc[1]["close"])
            d2_date = d_df_future.index[1].date()

            shares = int(slot_cap / entry_p) if entry_p > 0 else 0
            if shares <= 0:
                continue

            # 1-Day Swing (Exit Next Open)
            # Delivery STT = 0.1% buy + 0.1% sell = 0.2%, DP charges Rs 15.93, Angel brokerage Rs 0 delivery / Rs 20
            pnl_1d = (d1_open - entry_p) * shares - (0.0022 * slot_cap + 35.0)
            exit_time_1d = pd.to_datetime(f"{d1_date} 09:15:00")

            # 2-Day Swing (Exit Day 2 Close)
            pnl_2d = (d2_close - entry_p) * shares - (0.0022 * slot_cap + 35.0)
            exit_time_2d = pd.to_datetime(f"{d2_date} 15:15:00")

            swing_trades.append({
                "alpha_id": row["alpha_id"],
                "symbol": sym,
                "family": row["family"],
                "entry_time": entry_time,
                "entry_price": entry_p,
                "shares": shares,
                "exit_time_1d": exit_time_1d,
                "net_pnl_1d": pnl_1d,
                "exit_time_2d": exit_time_2d,
                "net_pnl_2d": pnl_2d,
                "month_year": row["month_year"]
            })

    st_df = pd.DataFrame(swing_trades).sort_values("entry_time").reset_index(drop=True)
    print(f"[*] Total Valid Long Swing Candidate Signals: {len(st_df):,}")

    # Simulate 1-Day Swing Portfolio (4 Slots @ Rs 1.25L, Symbol Diversity, Zero Lookahead)
    # Simulate 2-Day Swing Portfolio
    monthly_1d = []
    monthly_2d = []

    for ym in complete_months:
        m_cands = st_df[st_df["month_year"] == ym].copy().reset_index(drop=True)
        m_groups = list(m_cands.groupby("entry_time"))

        # 1-Day Swing Dispatch
        active_1d = []
        exec_1d = []
        for entry_time, group in m_groups:
            active_1d = [p for p in active_1d if p["exit_time_1d"] > entry_time]
            for c in group.to_dict("records"):
                if len(active_1d) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_1d}:
                    continue
                active_1d.append(c)
                exec_1d.append(c)

        pnl_1d_sum = sum(t["net_pnl_1d"] for t in exec_1d)
        roi_1d = (pnl_1d_sum / 500000.0) * 100.0
        monthly_1d.append({
            "month": str(ym), "trades": len(exec_1d), "net_pnl": pnl_1d_sum, "roi_pct": roi_1d
        })

        # 2-Day Swing Dispatch
        active_2d = []
        exec_2d = []
        for entry_time, group in m_groups:
            active_2d = [p for p in active_2d if p["exit_time_2d"] > entry_time]
            for c in group.to_dict("records"):
                if len(active_2d) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_2d}:
                    continue
                active_2d.append(c)
                exec_2d.append(c)

        pnl_2d_sum = sum(t["net_pnl_2d"] for t in exec_2d)
        roi_2d = (pnl_2d_sum / 500000.0) * 100.0
        monthly_2d.append({
            "month": str(ym), "trades": len(exec_2d), "net_pnl": pnl_2d_sum, "roi_pct": roi_2d
        })

    df_1d_res = pd.DataFrame(monthly_1d)
    df_2d_res = pd.DataFrame(monthly_2d)

    print("\n" + "=" * 120)
    print("ASHVA CASH SWING CAPABILITY (1-DAY & 2-DAY OVERNIGHT HOLDING ON RS 5L CAPITAL, 4 SLOTS)")
    print("=" * 120)
    print(f"{'METRIC / STATISTIC':<32} | {'INTRADAY MODEL 4 (15:15)':<26} | {'CASH SWING 1-DAY (NEXT OPEN)':<28} | {'CASH SWING 2-DAY (DAY 2 CLOSE)':<28}")
    print("-" * 120)
    print(f"{'Average Monthly ROI':<32} | {'+0.55%':<26} | {np.mean(df_1d_res['roi_pct']):>+7.2f}%                     | {np.mean(df_2d_res['roi_pct']):>+7.2f}%")
    print(f"{'Median Monthly ROI':<32} | {'+0.15%':<26} | {np.median(df_1d_res['roi_pct']):>+7.2f}%                     | {np.median(df_2d_res['roi_pct']):>+7.2f}%")
    pos_1d = (df_1d_res['roi_pct'] > 0).sum()
    pos_2d = (df_2d_res['roi_pct'] > 0).sum()
    print(f"{'Profitable Months (Win-Rate)':<32} | {'9/17 (52.9%)':<26} | {pos_1d}/17 ({pos_1d/17*100:4.1f}%)                  | {pos_2d}/17 ({pos_2d/17*100:4.1f}%)")
    print(f"{'Cumulative 17-Month Net P&L':<32} | {'Rs +47,093':<26} | Rs {df_1d_res['net_pnl'].sum():>+10,.0f}               | Rs {df_2d_res['net_pnl'].sum():>+10,.0f}")
    print(f"{'Average Trades per Month':<32} | {'4.2 trades':<26} | {np.mean(df_1d_res['trades']):>5.1f} trades                 | {np.mean(df_2d_res['trades']):>5.1f} trades")
    print(f"{'Worst Monthly Return':<32} | {'-0.65%':<26} | {np.min(df_1d_res['roi_pct']):>+7.2f}%                     | {np.min(df_2d_res['roi_pct']):>+7.2f}%")
    print(f"{'Best Monthly Return':<32} | {'+3.16%':<26} | {np.max(df_1d_res['roi_pct']):>+7.2f}%                     | {np.max(df_2d_res['roi_pct']):>+7.2f}%")
    print("=" * 120)

    print("\n--- MONTH-BY-MONTH REALIZED SWING PERFORMANCE ---")
    print(f"{'MONTH':<10} | {'INTRADAY MODEL 4':<24} | {'CASH SWING 1-DAY (NEXT OPEN)':<28} | {'CASH SWING 2-DAY (DAY 2 CLOSE)':<28}")
    print("-" * 100)
    for i in range(len(df_1d_res)):
        m_str = df_1d_res.iloc[i]["month"]
        p1 = df_1d_res.iloc[i]["net_pnl"]
        r1 = df_1d_res.iloc[i]["roi_pct"]
        p2 = df_2d_res.iloc[i]["net_pnl"]
        r2 = df_2d_res.iloc[i]["roi_pct"]
        print(f"{m_str:<10} | {'+0.55% Baseline':<24} | Rs {p1:>+7,.0f} ({r1:>+5.2f}%)               | Rs {p2:>+7,.0f} ({r2:>+5.2f}%)")
    print("=" * 100)


if __name__ == "__main__":
    run_cash_swing_portfolio()
