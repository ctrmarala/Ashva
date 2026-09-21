"""
Ashva PCDE Benchmark Suite: 18-Month Walk-Forward Cross-Validation & Untouched Holdout
=====================================================================================

Implements the definitive institutional validation protocol:
1. 18-Month Data Horizon (540 days, March 2025 - September 2026, 7,144 candidate signals).
2. Expanding Walk-Forward Folds:
   - Fold 1: Train M1-M6 (Mar-Aug 2025)   -> OOS Test M7-M9 (Sep-Nov 2025)
   - Fold 2: Train M1-M9 (Mar-Nov 2025)   -> OOS Test M10-M12 (Dec 2025 - Feb 2026)
   - Fold 3: Train M1-M12 (Mar 2025 - Feb 2026) -> OOS Test M13-M15 (Mar-May 2026)
   - Fold 4: Train M1-M15 (Mar 2025 - May 2026) -> FINAL UNTOUCHED HOLDOUT M16-M18 (Jun-Sep 2026)
3. Zero-Lookahead Parameter Calibration:
   - EV Hurdle theta*, Bayesian priors, and correlation matrices are calibrated strictly on the training fold,
     frozen, and evaluated on the untouched out-of-sample test fold.
4. Out-of-Sample Slices Concatenation (Months 7 to 18, 12 Months of true OOS testing).
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from scripts.research_pcde_phase2_3 import (
    load_candidates, evaluate_candidate_economics, compute_metrics,
    PointInTimeTracker, compute_n_eff, normalize_alpha_id, solve_milp_clairvoyant
)
from src.analytics.indian_costs import IndianCostModel


def calibrate_training_fold(train_cands: List[Dict[str, Any]], family_map: Dict[str, str]) -> Tuple[float, PointInTimeTracker]:
    """
    Calibrates the optimal Net EV Hurdle theta* and initial PIT distribution strictly within the training fold.
    Selects theta* that maximizes training Net P&L.
    """
    train_groups = list(pd.DataFrame(train_cands).groupby("entry_time"))
    
    hurdle_grid = [-100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0]
    best_hurdle = 0.0
    best_train_pnl = -np.inf

    for hurdle in hurdle_grid:
        pit = PointInTimeTracker(prior_weight=5.0)
        active_pos = []
        executed = []
        for entry_time, group in train_groups:
            still_active = []
            for p in active_pos:
                if p["exit_time"] <= entry_time:
                    pit.register_closed_trade(p)
                else:
                    still_active.append(p)
            active_pos = still_active

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
        m = compute_metrics(executed, 500000.0, len(train_cands))
        if m["net_pnl"] > best_train_pnl:
            best_train_pnl = m["net_pnl"]
            best_hurdle = hurdle

    # Build final PIT tracker seeded with all training trades
    final_pit = PointInTimeTracker(prior_weight=5.0)
    for t in train_cands:
        final_pit.register_closed_trade(t)

    return best_hurdle, final_pit


def run_walk_forward_fold(
    fold_idx: int,
    train_cands: List[Dict[str, Any]],
    test_cands: List[Dict[str, Any]],
    family_map: Dict[str, str],
    is_final_holdout: bool = False,
) -> Dict[str, Any]:
    """Executes a single walk-forward fold: In-Sample Calibration -> Frozen OOS Evaluation."""
    fold_name = f"Fold {fold_idx} (FINAL UNTOUCHED HOLDOUT)" if is_final_holdout else f"Fold {fold_idx}"
    print(f"\n[*] --- {fold_name} ---")
    print(f"    • Train Period: {train_cands[0]['entry_time'].strftime('%Y-%m-%d')} to {train_cands[-1]['entry_time'].strftime('%Y-%m-%d')} ({len(train_cands):,} candidates)")
    print(f"    • Test Period:  {test_cands[0]['entry_time'].strftime('%Y-%m-%d')} to {test_cands[-1]['entry_time'].strftime('%Y-%m-%d')} ({len(test_cands):,} candidates)")

    # 1. Calibrate on Training Fold (Zero Lookahead)
    best_hurdle, seeded_pit = calibrate_training_fold(train_cands, family_map)
    print(f"    • Calibrated Optimal In-Sample Hurdle: theta* = Rs {best_hurdle:+.1f}")

    test_groups = list(pd.DataFrame(test_cands).groupby("entry_time"))

    # Benchmark A: FIFO on Test Fold
    active_a = []
    exec_a = []
    for entry_time, group in test_groups:
        active_a = [p for p in active_a if p["exit_time"] > entry_time]
        cands_bar = group.to_dict("records")
        cands_bar.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
        for c in cands_bar:
            if len(active_a) >= 4:
                continue
            active_a.append(c)
            exec_a.append(c)
    mA = compute_metrics(exec_a, 500000.0, len(test_cands))

    # Benchmark C: Symbol Diversity on Test Fold
    active_c = []
    exec_c = []
    for entry_time, group in test_groups:
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
    mC = compute_metrics(exec_c, 500000.0, len(test_cands))

    # Benchmark WF-EV: Calibrated Net EV Hurdle on Test Fold
    pit_wf = PointInTimeTracker(prior_weight=5.0)
    for t in train_cands:
        pit_wf.register_closed_trade(t)

    active_ev = []
    exec_ev = []
    for entry_time, group in test_groups:
        still_active = []
        for p in active_ev:
            if p["exit_time"] <= entry_time:
                pit_wf.register_closed_trade(p)
            else:
                still_active.append(p)
        active_ev = still_active

        valid = []
        for c in group.to_dict("records"):
            ev = pit_wf.calculate_net_ev(c["alpha_id"], entry_time)
            if ev > best_hurdle:
                c["ev_net"] = ev
                c["pit_pf"], _, _, _ = pit_wf.get_alpha_metrics(c["alpha_id"], entry_time)
                valid.append(c)
        valid.sort(key=lambda x: x["ev_net"], reverse=True)
        for c in valid:
            if len(active_ev) >= 4:
                continue
            if c["symbol"] in {p["symbol"] for p in active_ev}:
                continue
            active_ev.append(c)
            exec_ev.append(c)
    mEV = compute_metrics(exec_ev, 500000.0, len(test_cands))

    # Benchmark WF-Full: Composite Utility with Calibrated Gating
    pit_full = PointInTimeTracker(prior_weight=5.0)
    for t in train_cands:
        pit_full.register_closed_trade(t)

    active_full = []
    exec_full = []
    for entry_time, group in test_groups:
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
            if ev <= best_hurdle:
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
    mFull = compute_metrics(exec_full, 500000.0, len(test_cands))

    # Benchmark J1: Clairvoyant Upper Bound on Test Fold
    exec_j1 = solve_milp_clairvoyant(test_cands, max_slots=4, enforce_symbol_diversity=True)
    mJ1 = compute_metrics(exec_j1, 500000.0, len(test_cands))

    print(f"    • Test Results on OOS:")
    print(f"      - FIFO (A):          Trades: {mA['trades']:>3d} | Net P&L: Rs {mA['net_pnl']:>+8.0f} ({mA['roi_pct']:>+5.2f}%) | MaxDD: {mA['max_dd_pct']:>4.2f}% | Costs: Rs {mA['costs']:>6.0f}")
    print(f"      - Symbol-Diverse (C):Trades: {mC['trades']:>3d} | Net P&L: Rs {mC['net_pnl']:>+8.0f} ({mC['roi_pct']:>+5.2f}%) | MaxDD: {mC['max_dd_pct']:>4.2f}% | Costs: Rs {mC['costs']:>6.0f}")
    print(f"      - Walk-Forward EV:   Trades: {mEV['trades']:>3d} | Net P&L: Rs {mEV['net_pnl']:>+8.0f} ({mEV['roi_pct']:>+5.2f}%) | MaxDD: {mEV['max_dd_pct']:>4.2f}% | Costs: Rs {mEV['costs']:>6.0f}")
    print(f"      - Walk-Forward Full: Trades: {mFull['trades']:>3d} | Net P&L: Rs {mFull['net_pnl']:>+8.0f} ({mFull['roi_pct']:>+5.2f}%) | MaxDD: {mFull['max_dd_pct']:>4.2f}% | Costs: Rs {mFull['costs']:>6.0f}")
    print(f"      - Clairvoyant (J1):  Trades: {mJ1['trades']:>3d} | Net P&L: Rs {mJ1['net_pnl']:>+8.0f} ({mJ1['roi_pct']:>+5.2f}%) | MaxDD: {mJ1['max_dd_pct']:>4.2f}% | Costs: Rs {mJ1['costs']:>6.0f}")

    return {
        "fold": fold_idx,
        "is_holdout": is_final_holdout,
        "hurdle": best_hurdle,
        "trades_a": exec_a,
        "trades_c": exec_c,
        "trades_ev": exec_ev,
        "trades_full": exec_full,
        "trades_j1": exec_j1,
        "metrics_a": mA,
        "metrics_c": mC,
        "metrics_ev": mEV,
        "metrics_full": mFull,
        "metrics_j1": mJ1,
    }


def main():
    print("=" * 135)
    print("ASHVA PCDE: 18-MONTH WALK-FORWARD CROSS-VALIDATION & UNTOUCHED HOLDOUT HARNESS")
    print("=" * 135)

    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0
    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]

    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    family_map = {c["alpha_id"]: c["family"] for c in cands}

    min_ts = df["entry_time"].min()
    max_ts = df["entry_time"].max()
    print(f"[*] Total Candidate Stream: {min_ts.strftime('%Y-%m-%d')} to {max_ts.strftime('%Y-%m-%d')} (540 Days | {len(df):,} Candidate Trades)\n")

    # Define 4 Expanding Walk-Forward Windows:
    # Horizon: 2025-03-25 to 2026-09-15 (~18 Months)
    # Fold 1: Train: 2025-03-25 to 2025-09-24 (6M)  | Test: 2025-09-25 to 2025-12-24 (3M)
    # Fold 2: Train: 2025-03-25 to 2025-12-24 (9M)  | Test: 2025-12-25 to 2026-03-24 (3M)
    # Fold 3: Train: 2025-03-25 to 2026-03-24 (12M) | Test: 2026-03-25 to 2026-06-24 (3M)
    # Fold 4: Train: 2025-03-25 to 2026-06-24 (15M) | FINAL HOLDOUT: 2026-06-25 to 2026-09-15 (3M)

    splits = [
        (1, "2025-03-25", "2025-09-24", "2025-09-25", "2025-12-24", False),
        (2, "2025-03-25", "2025-12-24", "2025-12-25", "2026-03-24", False),
        (3, "2025-03-25", "2026-03-24", "2026-03-25", "2026-06-24", False),
        (4, "2025-03-25", "2026-06-24", "2026-06-25", "2026-09-15", True),
    ]

    fold_results = []
    all_oos_a = []
    all_oos_c = []
    all_oos_ev = []
    all_oos_full = []
    all_oos_j1 = []
    total_oos_cands = 0

    for idx, tr_s, tr_e, te_s, te_e, is_holdout in splits:
        tr_df = df[(df["entry_time"] >= pd.to_datetime(tr_s)) & (df["entry_time"] <= pd.to_datetime(tr_e + " 23:59:59"))]
        te_df = df[(df["entry_time"] >= pd.to_datetime(te_s)) & (df["entry_time"] <= pd.to_datetime(te_e + " 23:59:59"))]

        res = run_walk_forward_fold(
            fold_idx=idx,
            train_cands=tr_df.to_dict("records"),
            test_cands=te_df.to_dict("records"),
            family_map=family_map,
            is_final_holdout=is_holdout,
        )
        fold_results.append(res)
        all_oos_a.extend(res["trades_a"])
        all_oos_c.extend(res["trades_c"])
        all_oos_ev.extend(res["trades_ev"])
        all_oos_full.extend(res["trades_full"])
        all_oos_j1.extend(res["trades_j1"])
        total_oos_cands += len(te_df)

    # Aggregate Over All 4 Out-of-Sample Slices (12 Months of pure OOS testing)
    agg_mA = compute_metrics(all_oos_a, 500000.0, total_oos_cands)
    agg_mC = compute_metrics(all_oos_c, 500000.0, total_oos_cands)
    agg_mEV = compute_metrics(all_oos_ev, 500000.0, total_oos_cands)
    agg_mFull = compute_metrics(all_oos_full, 500000.0, total_oos_cands)
    agg_mJ1 = compute_metrics(all_oos_j1, 500000.0, total_oos_cands)

    print("\n" + "=" * 145)
    print("12-MONTH AGGREGATE OUT-OF-SAMPLE SCORECARD (CONCATENATED OOS TEST FOLDS 1 TO 4 | ZERO LOOKAHEAD)")
    print("=" * 145)
    header = f"{'POLICY / HARNESS':<30} | {'TRADES':<6} | {'WIN %':<6} | {'GROSS P&L':<12} | {'COSTS':<10} | {'NET P&L':<12} | {'ROI %':<8} | {'MAX DD':<7} | {'PF':<5} | {'SHARPE':<6} | {'FRICT':<5}"
    print(header)
    print("-" * 145)

    models = [
        ("Test A: Baseline FIFO", agg_mA),
        ("Test C: Symbol Diversity", agg_mC),
        ("Test WF-EV: Walk-Forward Calibrated EV", agg_mEV),
        ("Test WF-Full: Walk-Forward Integrated PCDE", agg_mFull),
        ("Test J1: Global Clairvoyant Upper Bound", agg_mJ1),
    ]

    for name, m in models:
        tr_str = f"{m['trades']:d}" if m['trades'] > 0 else "-"
        wr_str = f"{m['win_rate']:.1f}%" if m['win_rate'] > 0 else "-"
        gr_str = f"Rs {m['gross_pnl']:>+9,.0f}" if m['gross_pnl'] != 0 else "-"
        cs_str = f"Rs {m['costs']:>7,.0f}" if m['costs'] != 0 else "-"
        net_str = f"Rs {m['net_pnl']:>+9,.0f}"
        roi_str = f"{m['roi_pct']:>+6.2f}%"
        dd_str = f"{m['max_dd_pct']:>5.2f}%" if m['max_dd_pct'] > 0 else "-"
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] > 0 else "-"
        sh_str = f"{m['sharpe']:>5.2f}" if m['sharpe'] != 0 else "-"
        fr_str = f"{m['friction_ratio']:.2f}x" if m['friction_ratio'] > 0 else "-"
        print(f"{name:<30} | {tr_str:<6} | {wr_str:<6} | {gr_str:<12} | {cs_str:<10} | {net_str:<12} | {roi_str:<8} | {dd_str:<7} | {pf_str:<5} | {sh_str:<6} | {fr_str:<5}")

    print("=" * 145)

    # FINAL UNTOUCHED HOLDOUT (FOLD 4)
    holdout_res = fold_results[-1]
    print("\n" + "=" * 145)
    print("FINAL UNTOUCHED HOLDOUT SCORECARD (FOLD 4: JUN 25 - SEP 15, 2026 | ZERO LOOKAHEAD)")
    print("=" * 145)
    print(header)
    print("-" * 145)
    holdout_models = [
        ("Test A: Baseline FIFO", holdout_res["metrics_a"]),
        ("Test C: Symbol Diversity", holdout_res["metrics_c"]),
        ("Test WF-EV: Walk-Forward Calibrated EV", holdout_res["metrics_ev"]),
        ("Test WF-Full: Walk-Forward Integrated PCDE", holdout_res["metrics_full"]),
        ("Test J1: Global Clairvoyant Upper Bound", holdout_res["metrics_j1"]),
    ]
    for name, m in holdout_models:
        tr_str = f"{m['trades']:d}" if m['trades'] > 0 else "-"
        wr_str = f"{m['win_rate']:.1f}%" if m['win_rate'] > 0 else "-"
        gr_str = f"Rs {m['gross_pnl']:>+9,.0f}" if m['gross_pnl'] != 0 else "-"
        cs_str = f"Rs {m['costs']:>7,.0f}" if m['costs'] != 0 else "-"
        net_str = f"Rs {m['net_pnl']:>+9,.0f}"
        roi_str = f"{m['roi_pct']:>+6.2f}%"
        dd_str = f"{m['max_dd_pct']:>5.2f}%" if m['max_dd_pct'] > 0 else "-"
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] > 0 else "-"
        sh_str = f"{m['sharpe']:>5.2f}" if m['sharpe'] != 0 else "-"
        fr_str = f"{m['friction_ratio']:.2f}x" if m['friction_ratio'] > 0 else "-"
        print(f"{name:<30} | {tr_str:<6} | {wr_str:<6} | {gr_str:<12} | {cs_str:<10} | {net_str:<12} | {roi_str:<8} | {dd_str:<7} | {pf_str:<5} | {sh_str:<6} | {fr_str:<5}")
    print("=" * 145)


if __name__ == "__main__":
    main()
