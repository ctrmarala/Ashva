"""
Audit of Candidate Pool & J1 Clairvoyant Upper Bound
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.ui.data_access import UIDataAccess
from scripts.research_pcde_phase1_5 import harvest_raw_candidates, evaluate_candidate_economics, solve_milp_clairvoyant

lake = DataLake(read_only=True)
symbols = get_universe_symbols()
dal = UIDataAccess()
cost_model = IndianCostModel(default_slippage_bps=3.0)

all_alphas_df = dal.get_alpha_registry_table()
proven_df = all_alphas_df[all_alphas_df["status"] == "PROVEN"]
alpha_ids = proven_df["alpha_id"].tolist()

raw_candidates = harvest_raw_candidates(lake, symbols, alpha_ids, dal)
s_ts = pd.to_datetime("2026-04-19 00:00:00")
e_ts = pd.to_datetime("2026-09-18 23:59:59")
window_raw = [c for c in raw_candidates if s_ts <= c["entry_time"] <= e_ts]

slot_cap = 125000.0
cands_eval = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in window_raw]

df = pd.DataFrame(cands_eval)
total_cands = len(df)
gross_pos = len(df[df["gross_pnl"] > 0])
net_pos = len(df[df["net_pnl"] > 0])
tax_killed = len(df[(df["gross_pnl"] > 0) & (df["net_pnl"] <= 0)])
gross_neg = len(df[df["gross_pnl"] <= 0])

print("=" * 90)
print("AUDIT OF 5-MONTH CANDIDATE STREAM (2,394 CANDIDATES @ Rs 125,000 SLOTS)")
print("=" * 90)
print(f"Total Candidate Signals:           {total_cands:,}")
print(f"Gross Profitable Signals:          {gross_pos:,} ({gross_pos/total_cands*100:.1f}%)")
print(f"Net Profitable Signals (Post-Tax): {net_pos:,} ({net_pos/total_cands*100:.1f}%)")
print(f"Tax-Killed Signals (Gross+ / Net-):{tax_killed:,} ({tax_killed/total_cands*100:.1f}%)")
print(f"Gross Losing Signals:              {gross_neg:,} ({gross_neg/total_cands*100:.1f}%)")
print("=" * 90)

# Run MILP
j1_trades = solve_milp_clairvoyant(cands_eval, max_slots=4, enforce_symbol_diversity=True)
df_j1 = pd.DataFrame(j1_trades)

print(f"\nJ1 Selected Trades:                {len(df_j1)} out of {net_pos} net-positive candidates")
print(f"J1 Total Gross P&L:                Rs {df_j1['gross_pnl'].sum():+10,.2f}")
print(f"J1 Total Statutory Costs:          Rs {df_j1['costs'].sum():10,.2f}")
print(f"J1 Total Net P&L:                  Rs {df_j1['net_pnl'].sum():+10,.2f}")
print(f"J1 Average Net Win / Trade:        Rs {df_j1['net_pnl'].mean():+10,.2f}")
print(f"J1 Min Net Win in Selected:        Rs {df_j1['net_pnl'].min():+10,.2f}")
print(f"J1 Max Net Win in Selected:        Rs {df_j1['net_pnl'].max():+10,.2f}")
