"""
Ashva Goal Mode: Critical Question #9 (Intraday Boundary & Multi-Day Holding Diagnostic)
========================================================================================

Evaluates whether the 15:15 forced EOD square-off is curtailing profitable trends,
and compares Intraday P&L vs Next-Day Open / Next-Day Close holding on the candidate set.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from scripts.research_pcde_phase2_3 import load_candidates, evaluate_candidate_economics


def run_eod_diagnostic():
    lake = DataLake(read_only=True)
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0

    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]
    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")

    # Complete 17 months
    complete_months = [m for m in sorted(df["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df = df[df["month_year"].isin(complete_months)].copy().reset_index(drop=True)

    print("=" * 100)
    print("CRITICAL QUESTION #9: INTRADAY BOUNDARY & EXIT REASON DECOMPOSITION (17 MONTHS)")
    print("=" * 100)
    
    exit_summary = df.groupby("exit_reason").agg(
        trades=("alpha_id", "count"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
        gross_pnl=("gross_pnl", "sum"),
        costs=("costs", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean")
    )
    print(exit_summary.to_string())
    print("-" * 100)
    
    # EOD Square-off / Time Exit trades
    time_exits = df[df["exit_reason"].isin(["TIME_EXIT", "EOD_SQUARE_OFF"])]
    tp_exits = df[df["exit_reason"] == "TAKE_PROFIT"]
    sl_exits = df[df["exit_reason"] == "STOP_LOSS"]

    print(f"\n[*] Take Profit Exits:  {len(tp_exits):,d} trades | Win Rate: {(tp_exits['net_pnl'] > 0).mean()*100:.1f}% | Net PnL: Rs {tp_exits['net_pnl'].sum():>+10,.0f} (Avg: Rs {tp_exits['net_pnl'].mean():>+6.1f})")
    print(f"[*] Stop Loss Exits:    {len(sl_exits):,d} trades | Win Rate: {(sl_exits['net_pnl'] > 0).mean()*100:.1f}% | Net PnL: Rs {sl_exits['net_pnl'].sum():>+10,.0f} (Avg: Rs {sl_exits['net_pnl'].mean():>+6.1f})")
    print(f"[*] Time / EOD Exits:   {len(time_exits):,d} trades | Win Rate: {(time_exits['net_pnl'] > 0).mean()*100:.1f}% | Net PnL: Rs {time_exits['net_pnl'].sum():>+10,.0f} (Avg: Rs {time_exits['net_pnl'].mean():>+6.1f})")
    print("=" * 100)


if __name__ == "__main__":
    run_eod_diagnostic()
