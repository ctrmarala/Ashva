"""
Ashva PCDE: 18-Month Independent Monthly ROI & Distribution Analytics
=====================================================================

Evaluates each calendar month from March 2025 to September 2026 as an independent
experiment starting with exactly Rs 5,00,000 capital (4 slots @ Rs 1,25,000 nominal).

Computes:
- Month-by-month net P&L and ROI (%)
- Average monthly ROI (Arithmetic Mean)
- Median monthly ROI
- Monthly Volatility (Std Dev of Monthly ROI)
- Profitable Months Count & Win-Month Rate (%)
- Best & Worst Monthly Return
- Maximum Portfolio Drawdown
- Cumulative P&L & ROI
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from scripts.research_pcde_phase2_3 import (
    load_candidates, evaluate_candidate_economics, compute_metrics,
    PointInTimeTracker, compute_n_eff, solve_milp_clairvoyant
)
from src.analytics.indian_costs import IndianCostModel


def run_monthly_dispatcher_analysis():
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0
    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]

    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")
    
    unique_months = sorted(df["month_year"].unique())
    family_map = {c["alpha_id"]: c["family"] for c in cands}

    print("=" * 145)
    print("ASHVA PCDE: 18-MONTH INDEPENDENT MONTHLY ROI & PERFORMANCE DISTRIBUTION")
    print("=" * 145)
    print(f"[*] Total Calendar Months: {len(unique_months)} months ({unique_months[0]} to {unique_months[-1]})")
    print(f"[*] Capital per Month: Rs 5,00,000 (Independent Monthly Allocation | 4 Slots @ Rs 1,25,000)\n")

    # Tracking records for each policy
    monthly_records = {
        "FIFO (Test A)": [],
        "Symbol Diversity (Test C)": [],
        "Walk-Forward Net EV": [],
        "Walk-Forward Full PCDE": [],
        "Clairvoyant Upper Bound (Test J1)": [],
    }

    # Month-by-Month Simulation
    for m_idx, ym in enumerate(unique_months, 1):
        m_df = df[df["month_year"] == ym].copy().reset_index(drop=True)
        m_cands = m_df.to_dict("records")
        m_groups = list(m_df.groupby("entry_time"))

        # Pre-month historical trades for PIT seeding
        pre_m_df = df[df["month_year"] < ym]
        pre_m_cands = pre_m_df.to_dict("records")

        # 1. Test A: Baseline FIFO
        active_a = []
        exec_a = []
        for entry_time, group in m_groups:
            active_a = [p for p in active_a if p["exit_time"] > entry_time]
            cands_bar = group.to_dict("records")
            cands_bar.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
            for c in cands_bar:
                if len(active_a) >= 4:
                    continue
                active_a.append(c)
                exec_a.append(c)
        mA = compute_metrics(exec_a, 500000.0, len(m_cands))
        monthly_records["FIFO (Test A)"].append({
            "month": str(ym), "trades": mA["trades"], "net_pnl": mA["net_pnl"], "roi_pct": mA["roi_pct"], "costs": mA["costs"], "max_dd": mA["max_dd_pct"]
        })

        # 2. Test C: Symbol Diversity
        active_c = []
        exec_c = []
        for entry_time, group in m_groups:
            active_c = [p for p in active_c if p["exit_time"] > entry_time]
            cands_bar = group.to_dict("records")
            cands_bar.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
            for c in cands_bar:
                if len(active_c) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_c}:
                    continue
                active_c.append(c)
                exec_c.append(c)
        mC = compute_metrics(exec_c, 500000.0, len(m_cands))
        monthly_records["Symbol Diversity (Test C)"].append({
            "month": str(ym), "trades": mC["trades"], "net_pnl": mC["net_pnl"], "roi_pct": mC["roi_pct"], "costs": mC["costs"], "max_dd": mC["max_dd_pct"]
        })

        # 3. Walk-Forward Net EV (Seeded strictly prior to current month)
        pit_ev = PointInTimeTracker(prior_weight=5.0)
        for t in pre_m_cands:
            pit_ev.register_closed_trade(t)

        active_ev = []
        exec_ev = []
        for entry_time, group in m_groups:
            still_active = []
            for p in active_ev:
                if p["exit_time"] <= entry_time:
                    pit_ev.register_closed_trade(p)
                else:
                    still_active.append(p)
            active_ev = still_active

            valid = []
            for c in group.to_dict("records"):
                ev = pit_ev.calculate_net_ev(c["alpha_id"], entry_time)
                if ev > 0.0:  # Zero-lookahead economic hurdle
                    c["ev_net"] = ev
                    valid.append(c)
            valid.sort(key=lambda x: x["ev_net"], reverse=True)
            for c in valid:
                if len(active_ev) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_ev}:
                    continue
                active_ev.append(c)
                exec_ev.append(c)
        mEV = compute_metrics(exec_ev, 500000.0, len(m_cands))
        monthly_records["Walk-Forward Net EV"].append({
            "month": str(ym), "trades": mEV["trades"], "net_pnl": mEV["net_pnl"], "roi_pct": mEV["roi_pct"], "costs": mEV["costs"], "max_dd": mEV["max_dd_pct"]
        })

        # 4. Walk-Forward Full PCDE (Seeded strictly prior to current month)
        pit_full = PointInTimeTracker(prior_weight=5.0)
        for t in pre_m_cands:
            pit_full.register_closed_trade(t)

        active_full = []
        exec_full = []
        for entry_time, group in m_groups:
            still_active = []
            for p in active_full:
                if p["exit_time"] <= entry_time:
                    pit_full.register_closed_trade(p)
                else:
                    still_active.append(p)
            active_full = still_active

            cands_bar = group.to_dict("records")
            sym_alphas = {}
            for c in cands_bar:
                sym_alphas.setdefault(c["symbol"], []).append(c["alpha_id"])
            sym_neff = {sym: compute_n_eff(a_list, family_map) for sym, a_list in sym_alphas.items()}

            scored = []
            for c in cands_bar:
                ev = pit_full.calculate_net_ev(c["alpha_id"], entry_time)
                if ev <= 0.0:
                    continue
                pf, _, _, _ = pit_full.get_alpha_metrics(c["alpha_id"], entry_time)
                neff = sym_neff[c["symbol"]]
                c["utility"] = np.log1p(neff) * pf * max(0.1, ev / 100.0)
                scored.append(c)
            scored.sort(key=lambda x: x["utility"], reverse=True)
            for c in scored:
                if len(active_full) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_full}:
                    continue
                active_full.append(c)
                exec_full.append(c)
        mFull = compute_metrics(exec_full, 500000.0, len(m_cands))
        monthly_records["Walk-Forward Full PCDE"].append({
            "month": str(ym), "trades": mFull["trades"], "net_pnl": mFull["net_pnl"], "roi_pct": mFull["roi_pct"], "costs": mFull["costs"], "max_dd": mFull["max_dd_pct"]
        })

        # 5. Test J1: True Clairvoyant Upper Bound
        exec_j1 = solve_milp_clairvoyant(m_cands, max_slots=4, enforce_symbol_diversity=True)
        mJ1 = compute_metrics(exec_j1, 500000.0, len(m_cands))
        monthly_records["Clairvoyant Upper Bound (Test J1)"].append({
            "month": str(ym), "trades": mJ1["trades"], "net_pnl": mJ1["net_pnl"], "roi_pct": mJ1["roi_pct"], "costs": mJ1["costs"], "max_dd": mJ1["max_dd_pct"]
        })

    # Summary Statistics
    print("=" * 145)
    print("18-MONTH INDEPENDENT MONTHLY ROI DISTRIBUTION SCORECARD (Rs 5,00,000 BASE CAPITAL PER MONTH)")
    print("=" * 145)
    header = f"{'METRIC / STATISTIC':<32} | {'FIFO (Test A)':<18} | {'Symbol Div (Test C)':<20} | {'WF Net EV (EV > 0)':<20} | {'WF Full PCDE':<18} | {'Clairvoyant (J1)':<18}"
    print(header)
    print("-" * 145)

    stats = {}
    for policy, records in monthly_records.items():
        df_m = pd.DataFrame(records)
        rois = df_m["roi_pct"].values
        pnls = df_m["net_pnl"].values
        trades = df_m["trades"].values
        costs = df_m["costs"].values
        dds = df_m["max_dd"].values

        avg_roi = float(np.mean(rois))
        med_roi = float(np.median(rois))
        std_roi = float(np.std(rois))
        pos_months = int((rois > 0).sum())
        win_month_rate = (pos_months / len(rois)) * 100.0
        best_month = float(np.max(rois))
        worst_month = float(np.min(rois))
        tot_pnl = float(np.sum(pnls))
        tot_roi = (tot_pnl / 500000.0) * 100.0
        tot_costs = float(np.sum(costs))
        avg_trades = float(np.mean(trades))
        max_dd = float(np.max(dds))

        stats[policy] = {
            "avg_roi": avg_roi, "med_roi": med_roi, "std_roi": std_roi,
            "pos_months": pos_months, "total_months": len(rois), "win_month_rate": win_month_rate,
            "best_month": best_month, "worst_month": worst_month,
            "tot_pnl": tot_pnl, "tot_roi": tot_roi, "tot_costs": tot_costs,
            "avg_trades": avg_trades, "max_dd": max_dd,
        }

    keys = list(monthly_records.keys())
    def fmt_row(label, field, fmt_str):
        vals = [fmt_str.format(stats[k][field]) for k in keys]
        print(f"{label:<32} | {vals[0]:<18} | {vals[1]:<20} | {vals[2]:<20} | {vals[3]:<18} | {vals[4]:<18}")

    fmt_row("Average Monthly ROI", "avg_roi", "{:+6.2f}%")
    fmt_row("Median Monthly ROI", "med_roi", "{:+6.2f}%")
    fmt_row("Monthly ROI Volatility (StdDev)", "std_roi", "{:6.2f}%")
    print(f"{'Profitable Months (Win-Rate)':<32} | {stats[keys[0]]['pos_months']:>2d}/{stats[keys[0]]['total_months']:>2d} ({stats[keys[0]]['win_month_rate']:>5.1f}%)    | {stats[keys[1]]['pos_months']:>2d}/{stats[keys[1]]['total_months']:>2d} ({stats[keys[1]]['win_month_rate']:>5.1f}%)      | {stats[keys[2]]['pos_months']:>2d}/{stats[keys[2]]['total_months']:>2d} ({stats[keys[2]]['win_month_rate']:>5.1f}%)      | {stats[keys[3]]['pos_months']:>2d}/{stats[keys[3]]['total_months']:>2d} ({stats[keys[3]]['win_month_rate']:>5.1f}%)    | {stats[keys[4]]['pos_months']:>2d}/{stats[keys[4]]['total_months']:>2d} ({stats[keys[4]]['win_month_rate']:>5.1f}%)")
    fmt_row("Best Monthly ROI", "best_month", "{:+6.2f}%")
    fmt_row("Worst Monthly ROI", "worst_month", "{:+6.2f}%")
    fmt_row("Average Trades per Month", "avg_trades", "{:6.1f} trades")
    fmt_row("Maximum Intra-Month Drawdown", "max_dd", "{:6.2f}%")
    fmt_row("Total 18-Month Statutory Costs", "tot_costs", "Rs {:>10,.0f}")
    fmt_row("Cumulative Net Realized P&L", "tot_pnl", "Rs {:>+10,.0f}")
    fmt_row("Cumulative Return on Rs 5L", "tot_roi", "{:+6.2f}%")
    print("=" * 145)

    # Print Month-by-Month Breakdown Table
    print("\n" + "=" * 145)
    print("MONTH-BY-MONTH RETURN TABLE (NET P&L / ROI % PER MONTH ACROSS POLICIES)")
    print("=" * 145)
    print(f"{'MONTH':<10} | {'FIFO (Test A)':<24} | {'Symbol Div (Test C)':<24} | {'WF Net EV (EV > 0)':<24} | {'WF Full PCDE':<24} | {'Clairvoyant (J1)':<24}")
    print("-" * 145)

    df_a = pd.DataFrame(monthly_records["FIFO (Test A)"])
    df_c = pd.DataFrame(monthly_records["Symbol Diversity (Test C)"])
    df_ev = pd.DataFrame(monthly_records["Walk-Forward Net EV"])
    df_full = pd.DataFrame(monthly_records["Walk-Forward Full PCDE"])
    df_j1 = pd.DataFrame(monthly_records["Clairvoyant Upper Bound (Test J1)"])

    for i in range(len(df_a)):
        m_str = df_a.iloc[i]["month"]
        a_str = f"Rs {df_a.iloc[i]['net_pnl']:>+7,.0f} ({df_a.iloc[i]['roi_pct']:>+5.2f}%)"
        c_str = f"Rs {df_c.iloc[i]['net_pnl']:>+7,.0f} ({df_c.iloc[i]['roi_pct']:>+5.2f}%)"
        ev_str = f"Rs {df_ev.iloc[i]['net_pnl']:>+7,.0f} ({df_ev.iloc[i]['roi_pct']:>+5.2f}%)"
        full_str = f"Rs {df_full.iloc[i]['net_pnl']:>+7,.0f} ({df_full.iloc[i]['roi_pct']:>+5.2f}%)"
        j1_str = f"Rs {df_j1.iloc[i]['net_pnl']:>+7,.0f} ({df_j1.iloc[i]['roi_pct']:>+5.2f}%)"
        print(f"{m_str:<10} | {a_str:<24} | {c_str:<24} | {ev_str:<24} | {full_str:<24} | {j1_str:<24}")

    print("=" * 145)


if __name__ == "__main__":
    run_monthly_dispatcher_analysis()
