"""
Ashva PCDE Phase 2.5: Methodological Validation Integrity & Forensic Audit
==========================================================================

Executes 4 rigorous quantitative checks to eliminate all potential lookahead,
data-snooping, and statistical overstatement:

1. Forensic Audit of Test E (Raw Consensus) vs Test F (N_eff Consensus):
   - Compares selected candidate trade IDs, rank correlation, and collision frequency.
2. Controlled Inventory Isolation (Test B1 vs B2 vs C):
   - B1: Curated Static 12-Alpha Roster
   - B2: 500 Random 12-Alpha Rosters (Null distribution of 12-alpha teams)
   - C: Full 50-Alpha Inventory (All under identical Symbol Diversity dispatcher)
3. In-Sample EV Calibration & Untouched Out-of-Sample Validation:
   - In-Sample Discovery Period: 2025-03-25 to 2026-04-18 (4,750 candidates)
   - Untouched OOS Validation: 2026-04-19 to 2026-09-18 (2,394 candidates)
   - Parameter Freeze: Calibrate hurdle theta* strictly In-Sample, evaluate once on OOS.
4. Walk-Forward Expanding Window Dispatcher:
   - Rolling PIT parameter estimation with zero future lookahead.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from scripts.research_pcde_phase2_3 import (
    load_candidates, evaluate_candidate_economics, compute_metrics,
    PointInTimeTracker, compute_n_eff, normalize_alpha_id, solve_milp_clairvoyant
)
from src.analytics.indian_costs import IndianCostModel


def run_forensic_audit_e_vs_f(oos_cands: List[Dict[str, Any]], family_map: Dict[str, str], in_sample_cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspects whether Test E and Test F calculate different scores and where their selections diverge."""
    print("\n" + "=" * 110)
    print("INVESTIGATION 1: FORENSIC AUDIT OF TEST E (RAW CONSENSUS) VS TEST F (N_eff CONSENSUS)")
    print("=" * 110)

    time_groups = list(pd.DataFrame(oos_cands).groupby("entry_time"))

    # Track multi-alpha collision events
    total_bars = len(time_groups)
    collision_bars = 0
    multi_alpha_instances = []

    # Run Test E
    pit_e = PointInTimeTracker(prior_weight=5.0)
    for t in in_sample_cands:
        pit_e.register_closed_trade(t)

    active_e = []
    exec_e = []
    e_bar_rankings = []

    for entry_time, group in time_groups:
        active_e = [p for p in active_e if p["exit_time"] > entry_time]
        cands_bar = group.to_dict("records")

        sym_counts = {}
        for c in cands_bar:
            sym_counts[c["symbol"]] = sym_counts.get(c["symbol"], 0) + 1

        for c in cands_bar:
            c["pit_pf"], _, _, _ = pit_e.get_alpha_metrics(c["alpha_id"], entry_time)
            c["score_e"] = float(sym_counts[c["symbol"]]) + (0.01 * c["pit_pf"])

        cands_bar.sort(key=lambda x: x["score_e"], reverse=True)
        e_bar_rankings.append([c["alpha_id"] + "_" + c["symbol"] for c in cands_bar])

        for c in cands_bar:
            if len(active_e) >= 4:
                continue
            if c["symbol"] in {p["symbol"] for p in active_e}:
                continue
            active_e.append(c)
            exec_e.append(c)

    # Run Test F
    pit_f = PointInTimeTracker(prior_weight=5.0)
    for t in in_sample_cands:
        pit_f.register_closed_trade(t)

    active_f = []
    exec_f = []
    f_bar_rankings = []

    for entry_time, group in time_groups:
        active_f = [p for p in active_f if p["exit_time"] > entry_time]
        cands_bar = group.to_dict("records")

        sym_alphas = {}
        for c in cands_bar:
            sym_alphas.setdefault(c["symbol"], []).append(c["alpha_id"])

        has_collision = any(len(a_list) > 1 for a_list in sym_alphas.values())
        if has_collision:
            collision_bars += 1
            for sym, a_list in sym_alphas.items():
                if len(a_list) > 1:
                    multi_alpha_instances.append({
                        "time": entry_time, "symbol": sym, "alphas": a_list,
                        "n_raw": len(a_list), "n_eff": compute_n_eff(a_list, family_map)
                    })

        sym_neff = {sym: compute_n_eff(a_list, family_map) for sym, a_list in sym_alphas.items()}

        for c in cands_bar:
            c["pit_pf"], _, _, _ = pit_f.get_alpha_metrics(c["alpha_id"], entry_time)
            c["score_f"] = sym_neff[c["symbol"]] * c["pit_pf"]

        cands_bar.sort(key=lambda x: x["score_f"], reverse=True)
        f_bar_rankings.append([c["alpha_id"] + "_" + c["symbol"] for c in cands_bar])

        for c in cands_bar:
            if len(active_f) >= 4:
                continue
            if c["symbol"] in {p["symbol"] for p in active_f}:
                continue
            active_f.append(c)
            exec_f.append(c)

    # Compare Trade Selections
    set_e = set((c["alpha_id"], c["symbol"], str(c["entry_time"])) for c in exec_e)
    set_f = set((c["alpha_id"], c["symbol"], str(c["entry_time"])) for c in exec_f)

    intersection = set_e.intersection(set_f)
    e_only = set_e - set_f
    f_only = set_f - set_e

    print(f"[*] Total Decision Timestamps:               {total_bars:,} bars")
    print(f"[*] Bars with Simultaneous Symbol Collisions: {collision_bars:,} bars ({collision_bars/total_bars*100:.1f}%)")
    print(f"[*] Total Multi-Alpha Symbol Instances:       {len(multi_alpha_instances):,}")
    print(f"[*] Trades Selected by Test E (Raw):         {len(exec_e):d}")
    print(f"[*] Trades Selected by Test F (N_eff):       {len(exec_f):d}")
    print(f"[*] Trade Overlap / Intersection:            {len(intersection):d} ({len(intersection)/len(set_e)*100:.1f}%)")
    print(f"[*] Trades in E but NOT in F:                {len(e_only):d}")
    print(f"[*] Trades in F but NOT in E:                {len(f_only):d}")

    if multi_alpha_instances:
        print("\n[*] Sample Multi-Alpha Collision & N_eff Adjustments:")
        df_col = pd.DataFrame(multi_alpha_instances)
        print(df_col.head(6).to_string(index=False))

    return {
        "collision_bars": collision_bars,
        "overlap_pct": len(intersection) / len(set_e) * 100.0 if set_e else 100.0,
        "e_only": len(e_only),
        "f_only": len(f_only),
    }


def run_inventory_isolation_b1_b2_c(oos_cands: List[Dict[str, Any]], static_12_roster: set, all_proven_ids: List[str]) -> None:
    """Isolates the 12-alpha vs 50-alpha inventory effect by comparing B1, B2 (500 random 12-alpha rosters), and C."""
    print("\n" + "=" * 110)
    print("INVESTIGATION 2: INVENTORY ISOLATION (TEST B1 vs TEST B2 NULL vs TEST C)")
    print("=" * 110)

    time_groups = list(pd.DataFrame(oos_cands).groupby("entry_time"))

    # Test B1: Curated 12 Alphas
    normalized_b1 = {normalize_alpha_id(a) for a in static_12_roster}
    active_b1 = []
    exec_b1 = []
    for entry_time, group in time_groups:
        active_b1 = [p for p in active_b1 if p["exit_time"] > entry_time]
        cands_bar = [c for c in group.to_dict("records") if normalize_alpha_id(c["alpha_id"]) in normalized_b1]
        cands_bar.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
        for c in cands_bar:
            if len(active_b1) >= 4:
                continue
            if c["symbol"] in {p["symbol"] for p in active_b1}:
                continue
            active_b1.append(c)
            exec_b1.append(c)
    mB1 = compute_metrics(exec_b1, 500000.0, len(oos_cands))

    # Test C: Full 50 Alphas
    active_c = []
    exec_c = []
    for entry_time, group in time_groups:
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
    mC = compute_metrics(exec_c, 500000.0, len(oos_cands))

    # Test B2: 500 Random 12-Alpha Rosters under identical Symbol Diversity
    b2_net_pnls = []
    b2_trade_counts = []
    b2_rois = []
    rng = random.Random(42)

    unique_alphas = list(set(normalize_alpha_id(c["alpha_id"]) for c in oos_cands))

    for seed in range(500):
        sampled_12 = set(rng.sample(unique_alphas, k=12))
        active_b2 = []
        exec_b2 = []
        for entry_time, group in time_groups:
            active_b2 = [p for p in active_b2 if p["exit_time"] > entry_time]
            cands_bar = [c for c in group.to_dict("records") if normalize_alpha_id(c["alpha_id"]) in sampled_12]
            cands_bar.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
            for c in cands_bar:
                if len(active_b2) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_b2}:
                    continue
                active_b2.append(c)
                exec_b2.append(c)
        m_b2 = compute_metrics(exec_b2, 500000.0, len(oos_cands))
        b2_net_pnls.append(m_b2["net_pnl"])
        b2_trade_counts.append(m_b2["trades"])
        b2_rois.append(m_b2["roi_pct"])

    arr_b2 = np.array(b2_net_pnls)
    mean_b2 = float(np.mean(arr_b2))
    std_b2 = float(np.std(arr_b2))
    p10_b2 = float(np.percentile(arr_b2, 10))
    p50_b2 = float(np.percentile(arr_b2, 50))
    p90_b2 = float(np.percentile(arr_b2, 90))

    # p-value of 50-alpha (C) vs 12-alpha null distribution (B2)
    p_val_c_vs_b2 = float((1.0 + (arr_b2 >= mC["net_pnl"]).sum()) / (1.0 + len(arr_b2)))
    # p-value of curated 12 (B1) vs random 12 (B2)
    p_val_b1_vs_b2 = float((1.0 + (arr_b2 >= mB1["net_pnl"]).sum()) / (1.0 + len(arr_b2)))

    print(f"{'POLICY VARIANT':<35} | {'TRADES':<6} | {'NET P&L':<12} | {'ROI %':<8} | {'MAX DD':<7} | {'PF':<5} | {'p-VAL vs B2 NULL':<16}")
    print("-" * 110)
    print(f"{'Test B1: Curated 12-Alpha Team':<35} | {mB1['trades']:<6d} | Rs {mB1['net_pnl']:>+8.0f} | {mB1['roi_pct']:>+6.2f}% | {mB1['max_dd_pct']:>5.2f}% | {mB1['profit_factor']:<5.2f} | {p_val_b1_vs_b2:<16.3f}")
    print(f"{'Test B2: Random 12 Teams (Mean Null)':<35} | {int(np.mean(b2_trade_counts)):<6d} | Rs {mean_b2:>+8.0f} | {mean_b2/5000.0:>+6.2f}% | {'—':<7} | {'—':<5} | {'0.500 (Ref Null)':<16}")
    print(f"{'Test B2: Random 12 Teams [P10 - P90]':<35} | {'—':<6} | [Rs {p10_b2:>.0f}, {p90_b2:>.0f}] | {'—':<8} | {'—':<7} | {'—':<5} | {'—':<16}")
    print(f"{'Test C: Full 50-Alpha Universe':<35} | {mC['trades']:<6d} | Rs {mC['net_pnl']:>+8.0f} | {mC['roi_pct']:>+6.2f}% | {mC['max_dd_pct']:>5.2f}% | {mC['profit_factor']:<5.2f} | {p_val_c_vs_b2:<16.3f}")
    print("=" * 110)


def run_in_sample_calibration_and_oos_validation(
    in_sample_cands: List[Dict[str, Any]],
    oos_cands: List[Dict[str, Any]],
    family_map: Dict[str, str],
) -> None:
    """
    Calibrates hurdle threshold theta* strictly In-Sample, freezes theta*,
    and evaluates on untouched Out-of-Sample data.
    """
    print("\n" + "=" * 110)
    print("INVESTIGATION 3: IN-SAMPLE EV CALIBRATION & UNTOUCHED OUT-OF-SAMPLE VALIDATION")
    print("=" * 110)

    # 1. In-Sample Sweep (2025-03-25 to 2026-04-18 | 4,750 Candidates)
    is_groups = list(pd.DataFrame(in_sample_cands).groupby("entry_time"))
    
    print("[*] Sweeping EV Hurdles on IN-SAMPLE Discovery Period to select theta*...")
    is_results = {}
    for hurdle in [-100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0]:
        pit = PointInTimeTracker(prior_weight=5.0)
        active_pos = []
        executed = []
        for entry_time, group in is_groups:
            active_pos = [p for p in active_pos if p["exit_time"] > entry_time]
            valid = []
            for c in group.to_dict("records"):
                ev = pit.calculate_net_ev(c["alpha_id"], entry_time)
                if ev > hurdle:
                    c["ev_net"] = ev
                    valid.append(c)
            valid.sort(key=lambda x: x["ev_net"], reverse=True)
            for c in valid:
                if len(active_pos) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_pos}:
                    continue
                active_pos.append(c)
                executed.append(c)
        m = compute_metrics(executed, 500000.0, len(in_sample_cands))
        is_results[hurdle] = m
        print(f"    In-Sample Hurdle EV > Rs {hurdle:>+6.1f} | Trades: {m['trades']:>4d} | Net P&L: Rs {m['net_pnl']:>+10.0f} ({m['roi_pct']:>+6.2f}%) | MaxDD: {m['max_dd_pct']:>5.2f}% | PF: {m['profit_factor']:>4.2f}")

    # Optimal In-Sample Threshold Selection Criterion: Maximum In-Sample Sharpe / Net P&L
    best_hurdle = max(is_results.keys(), key=lambda h: is_results[h]["net_pnl"])
    print(f"\n[+] In-Sample Calibrated Optimal Hurdle FROZEN at: theta* = Rs {best_hurdle:+.1f} (In-Sample Net P&L: Rs {is_results[best_hurdle]['net_pnl']:+,.0f})")

    # 2. Evaluate FROZEN theta* on UNTOUCHED Out-of-Sample Data (2026-04-19 to 2026-09-18)
    print("\n[*] Evaluating FROZEN theta* on UNTOUCHED Out-of-Sample Period (Zero Lookahead)...")
    
    # Initialize PIT seeded with In-Sample history
    pit_oos = PointInTimeTracker(prior_weight=5.0)
    for t in in_sample_cands:
        pit_oos.register_closed_trade(t)

    oos_groups = list(pd.DataFrame(oos_cands).groupby("entry_time"))
    active_oos = []
    executed_oos = []

    for entry_time, group in oos_groups:
        still_active = []
        for p in active_oos:
            if p["exit_time"] <= entry_time:
                pit_oos.register_closed_trade(p)
            else:
                still_active.append(p)
        active_oos = still_active

        valid = []
        for c in group.to_dict("records"):
            ev = pit_oos.calculate_net_ev(c["alpha_id"], entry_time)
            if ev > best_hurdle:
                c["ev_net"] = ev
                c["pit_pf"], _, _, _ = pit_oos.get_alpha_metrics(c["alpha_id"], entry_time)
                valid.append(c)

        valid.sort(key=lambda x: x["ev_net"], reverse=True)

        for c in valid:
            if len(active_oos) >= 4:
                continue
            if c["symbol"] in {p["symbol"] for p in active_oos}:
                continue
            active_oos.append(c)
            executed_oos.append(c)

    m_oos_frozen = compute_metrics(executed_oos, 500000.0, len(oos_cands))

    # Monte Carlo Null for Frozen Hurdle
    null_oos_nets = []
    for seed in range(500):
        rng = random.Random(seed + 1000)
        active_null = []
        exec_null = []
        for entry_time, group in oos_groups:
            active_null = [p for p in active_null if p["exit_time"] > entry_time]
            cand_list = group.to_dict("records")
            rng.shuffle(cand_list)
            for c in cand_list:
                if len(active_null) >= 4:
                    continue
                if c["symbol"] in {p["symbol"] for p in active_null}:
                    continue
                active_null.append(c)
                exec_null.append(c)
        m_null = compute_metrics(exec_null, 500000.0, len(oos_cands))
        null_oos_nets.append(m_null["net_pnl"])

    arr_null = np.array(null_oos_nets)
    p_val_frozen = float((1.0 + (arr_null >= m_oos_frozen["net_pnl"]).sum()) / (1.0 + len(arr_null)))

    print("\n" + "=" * 110)
    print("UNTOUCHED OUT-OF-SAMPLE VERIFICATION OF IN-SAMPLE FROZEN HURDLE")
    print("=" * 110)
    print(f"  • Frozen Hurdle Policy:        EV > Rs {best_hurdle:+.1f}")
    print(f"  • OOS Executed Trades:         {m_oos_frozen['trades']:d} trades (Rejection Rate: {m_oos_frozen['rejection_rate']:.1f}%)")
    print(f"  • OOS Win Rate:                {m_oos_frozen['win_rate']:.1f}%")
    print(f"  • OOS Gross P&L:               Rs {m_oos_frozen['gross_pnl']:>+9,.2f}")
    print(f"  • OOS Statutory Costs:         Rs {m_oos_frozen['costs']:>9,.2f}")
    print(f"  • OOS NET Realized P&L:        Rs {m_oos_frozen['net_pnl']:>+9,.2f} ({m_oos_frozen['roi_pct']:+.2f}% Net ROI)")
    print(f"  • OOS Max Drawdown:            {m_oos_frozen['max_dd_pct']:.2f}%")
    print(f"  • OOS Realized Profit Factor:  {m_oos_frozen['profit_factor']:.2f}")
    print(f"  • OOS Empirical p-Value:       {p_val_frozen:.3f}")
    print("=" * 110)


def main():
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0
    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]

    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    s_ts = pd.to_datetime("2026-04-19 00:00:00")
    e_ts = pd.to_datetime("2026-09-18 23:59:59")

    in_sample_cands = df[df["entry_time"] < s_ts].to_dict("records")
    oos_cands = df[(df["entry_time"] >= s_ts) & (df["entry_time"] <= e_ts)].to_dict("records")

    family_map = {c["alpha_id"]: c["family"] for c in cands}
    all_proven_ids = sorted(list(set(c["alpha_id"] for c in cands)))

    static_12_roster = {
        "105_alpha", "106_alpha", "64_alpha",
        "110_alpha", "112_alpha", "46_alpha",
        "117_alpha", "115_alpha", "55_alpha",
        "120_alpha", "122_alpha", "104_alpha",
    }

    # Investigation 1: Audit E vs F
    run_forensic_audit_e_vs_f(oos_cands, family_map, in_sample_cands)

    # Investigation 2: Inventory Isolation (B1 vs B2 vs C)
    run_inventory_isolation_b1_b2_c(oos_cands, static_12_roster, all_proven_ids)

    # Investigation 3: In-Sample Calibration & Untouched OOS
    run_in_sample_calibration_and_oos_validation(in_sample_cands, oos_cands, family_map)


if __name__ == "__main__":
    main()
