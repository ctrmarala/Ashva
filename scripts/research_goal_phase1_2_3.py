"""
Ashva Goal Mode: Phase 1 (Alpha Diagnostic), Phase 2 (Capacity Scaling), Phase 3 (Bottleneck Analysis)
====================================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.analytics.indian_costs import IndianCostModel
from scripts.research_pcde_phase2_3 import (
    load_candidates, evaluate_candidate_economics, compute_metrics, solve_milp_clairvoyant
)


def run_goal_phase1_2_3():
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0

    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]
    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")

    # Complete 17 months filter (2025-04 to 2026-08)
    complete_months = [m for m in sorted(df["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df = df[df["month_year"].isin(complete_months)].copy().reset_index(drop=True)
    df["is_win"] = df["net_pnl"] > 0.0

    print("=" * 140)
    print("PHASE 1: 50 PROVEN ALPHAS PERFORMANCE & ECONOMICS AUDIT (17 COMPLETE MONTHS, 7,070 CANDIDATES)")
    print("=" * 140)
    
    alpha_records = []
    for a_id, group in df.groupby("alpha_id"):
        fam = group["family"].iloc[0]
        c_count = len(group)
        wins = int(group["is_win"].sum())
        wr = (wins / c_count) * 100.0
        gross_pnl = float(group["gross_pnl"].sum())
        costs = float(group["costs"].sum())
        net_pnl = float(group["net_pnl"].sum())
        avg_net_trade = float(group["net_pnl"].mean())
        
        pos_sum = group[group["net_pnl"] > 0]["net_pnl"].sum()
        neg_sum = abs(group[group["net_pnl"] < 0]["net_pnl"].sum())
        pf = float(pos_sum / neg_sum) if neg_sum > 0 else 99.0

        # Monthly positive count
        m_pnls = group.groupby("month_year")["net_pnl"].sum()
        pos_months = int((m_pnls > 0).sum())
        active_months = len(m_pnls)
        
        alpha_records.append({
            "alpha_id": a_id,
            "family": fam,
            "candidates": c_count,
            "win_rate_%": wr,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "net_pnl": net_pnl,
            "avg_net_trade": avg_net_trade,
            "profit_factor": pf,
            "pos_months": pos_months,
            "active_months": active_months,
            "win_month_%": (pos_months / max(1, active_months)) * 100.0
        })

    alpha_df = pd.DataFrame(alpha_records).sort_values("net_pnl", ascending=False).reset_index(drop=True)
    
    print(f"{'ALPHA ID':<12} | {'FAMILY':<22} | {'CANDS':<6} | {'WIN RATE':<9} | {'GROSS PNL':<12} | {'COSTS':<10} | {'NET PNL':<12} | {'AVG/TRADE':<10} | {'PF':<5} | {'POS MONTHS':<10}")
    print("-" * 140)
    for idx, r in alpha_df.iterrows():
        print(f"{r['alpha_id']:<12} | {r['family']:<22} | {r['candidates']:<6d} | {r['win_rate_%']:>7.1f}% | Rs {r['gross_pnl']:>+9,.0f} | Rs {r['costs']:>7,.0f} | Rs {r['net_pnl']:>+9,.0f} | Rs {r['avg_net_trade']:>+7.1f} | {r['profit_factor']:>4.2f} | {r['pos_months']:>2d}/{r['active_months']:>2d} ({r['win_month_%']:>4.1f}%)")
    print("=" * 140)

    # Summary: Profitable vs Unprofitable alphas
    prof_alphas = alpha_df[alpha_df["net_pnl"] > 0]
    loss_alphas = alpha_df[alpha_df["net_pnl"] <= 0]
    print(f"\n[*] Standalone Profitable Alphas (Post-Cost): {len(prof_alphas)} / 50 (Total Net PnL: Rs {prof_alphas['net_pnl'].sum():>+10,.0f})")
    print(f"[*] Standalone Unprofitable Alphas (Post-Cost): {len(loss_alphas)} / 50 (Total Net PnL: Rs {loss_alphas['net_pnl'].sum():>+10,.0f})")
    print(f"[*] Total Candidate Stream Gross PnL: Rs {df['gross_pnl'].sum():>+10,.0f} | Total Friction Drag: Rs {df['costs'].sum():>10,.0f} | Net Stream PnL: Rs {df['net_pnl'].sum():>+10,.0f}")

    # PHASE 2: CAPACITY EXPERIMENT (4, 6, 8, 12, 16, 20 SLOTS)
    print("\n" + "=" * 140)
    print("PHASE 2: CAPACITY SCALING EXPERIMENT (IS 5% MATHEMATICALLY POSSIBLE?)")
    print("=" * 140)
    print("[*] Testing J1 Clairvoyant Upper Bound and Naive SymDiv across Slot Capacities [4, 6, 8, 12, 16, 20] on Rs 5,00,000 base capital/month...\n")

    slot_capacities = [4, 6, 8, 12, 16, 20]
    cap_records_j1 = []
    cap_records_symdiv = []

    for slots in slot_capacities:
        m_j1_results = []
        m_sym_results = []
        
        for ym in complete_months:
            m_sub = df[df["month_year"] == ym].copy().reset_index(drop=True)
            m_cands = m_sub.to_dict("records")
            m_groups = list(m_sub.groupby("entry_time"))

            # J1 MILP with 'slots' capacity
            j1_exec = solve_milp_clairvoyant(m_cands, max_slots=slots, enforce_symbol_diversity=True)
            m_j1 = compute_metrics(j1_exec, 500000.0, len(m_cands))
            m_j1_results.append(m_j1)

            # SymDiv with 'slots' capacity
            active_c = []
            exec_c = []
            for entry_time, group in m_groups:
                active_c = [p for p in active_c if p["exit_time"] > entry_time]
                cands_bar = group.to_dict("records")
                cands_bar.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
                for c in cands_bar:
                    if len(active_c) >= slots:
                        continue
                    if c["symbol"] in {p["symbol"] for p in active_c}:
                        continue
                    active_c.append(c)
                    exec_c.append(c)
            m_sym = compute_metrics(exec_c, 500000.0, len(m_cands))
            m_sym_results.append(m_sym)

        # Aggregate J1 stats
        j1_rois = [r["roi_pct"] for r in m_j1_results]
        j1_pnls = [r["net_pnl"] for r in m_j1_results]
        j1_trades = [r["trades"] for r in m_j1_results]
        j1_costs = [r["costs"] for r in m_j1_results]
        j1_pos = sum(1 for r in j1_rois if r > 0)
        
        cap_records_j1.append({
            "slots": slots,
            "avg_monthly_roi": float(np.mean(j1_rois)),
            "med_monthly_roi": float(np.median(j1_rois)),
            "avg_monthly_pnl": float(np.mean(j1_pnls)),
            "total_pnl": float(np.sum(j1_pnls)),
            "avg_trades_month": float(np.mean(j1_trades)),
            "win_month_rate": (j1_pos / len(j1_rois)) * 100.0,
            "total_costs": float(np.sum(j1_costs))
        })

        # Aggregate SymDiv stats
        sym_rois = [r["roi_pct"] for r in m_sym_results]
        sym_pnls = [r["net_pnl"] for r in m_sym_results]
        sym_trades = [r["trades"] for r in m_sym_results]
        sym_costs = [r["costs"] for r in m_sym_results]
        sym_pos = sum(1 for r in sym_rois if r > 0)

        cap_records_symdiv.append({
            "slots": slots,
            "avg_monthly_roi": float(np.mean(sym_rois)),
            "med_monthly_roi": float(np.median(sym_rois)),
            "avg_monthly_pnl": float(np.mean(sym_pnls)),
            "total_pnl": float(np.sum(sym_pnls)),
            "avg_trades_month": float(np.mean(sym_trades)),
            "win_month_rate": (sym_pos / len(sym_rois)) * 100.0,
            "total_costs": float(np.sum(sym_costs))
        })

    print("--- A. J1 CLAIRVOYANT UPPER BOUND VS CAPACITY (MAX MATHEMATICAL OPPORTUNITY) ---")
    df_cap_j1 = pd.DataFrame(cap_records_j1)
    print(df_cap_j1.to_string(index=False))

    print("\n--- B. NAIVE SYMBOL DIVERSITY DISPATCH VS CAPACITY (REALISTIC TURNOVER IMPACT) ---")
    df_cap_sym = pd.DataFrame(cap_records_symdiv)
    print(df_cap_sym.to_string(index=False))


if __name__ == "__main__":
    run_goal_phase1_2_3()
