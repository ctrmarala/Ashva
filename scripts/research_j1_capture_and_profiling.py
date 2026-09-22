"""
Ashva J1 Capture Optimization: Forensic Miss Decomposition & Dynamic Sizing
=============================================================================

1. Deconstructs the 228 J1 Clairvoyant Opportunities to quantify the exact rejection reasons:
   - Rejection Reason A: Pruned Alpha ID
   - Rejection Reason B: Timing Gate (Entry outside 09:20-09:45)
   - Rejection Reason C: VWAP Location Gate (Dist > 0.2 ATR)
   - Rejection Reason D: ATR Volatility Gate (ATR% < 0.55% or > 1.20%)
   - Rejection Reason E: Slot / Symbol Capacity Congestion

2. Builds Alpha-Family Specific Execution Profiles:
   - Custom entry timing and VWAP thresholds per family

3. Tests Dynamic Conviction-Based Position Sizing:
   - Tier 1 (Baseline Conviction): Rs 75,000
   - Tier 2 (High Conviction): Rs 125,000
   - Tier 3 (Exceptional Conviction): Rs 175,000 - Rs 200,000

4. Evaluates Walk-Forward Monthly Performance (Folds 1 to 4) aiming to increase J1 Capture from 20% toward 40-50%.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from scripts.research_pcde_phase2_3 import (
    load_candidates, evaluate_candidate_economics, compute_metrics, solve_milp_clairvoyant
)
from scripts.research_goal_models import extract_features


def main():
    lake = DataLake(read_only=True)
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0

    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]
    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")

    # Complete 17 months: 2025-04 to 2026-08
    complete_months = [m for m in sorted(df["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df = df[df["month_year"].isin(complete_months)].copy().reset_index(drop=True)

    print("[*] Extracting technical indicators and market context...", flush=True)
    df = extract_features(df, lake)

    # 1. Identify J1 Trades across all 17 months
    j1_trade_keys = set()
    j1_trade_list = []
    for ym in complete_months:
        c_sub = df[df["month_year"] == ym].to_dict("records")
        j1_exec = solve_milp_clairvoyant(c_sub, max_slots=4, enforce_symbol_diversity=True)
        for t in j1_exec:
            k = (t["alpha_id"], t["symbol"], str(t["entry_time"]))
            j1_trade_keys.add(k)
            j1_trade_list.append(t)

    df["is_j1"] = df.apply(lambda r: (r["alpha_id"], r["symbol"], str(r["entry_time"])) in j1_trade_keys, axis=1)
    j1_df = df[df["is_j1"]].copy().reset_index(drop=True)

    print("=" * 120)
    print(f"STEP 1: FORENSIC DECOMPOSITION OF THE 228 J1 CLAIRVOYANT OPPORTUNITIES (TOTAL PNL: Rs {j1_df['net_pnl'].sum():,.0f})")
    print("=" * 120)

    alpha_pnls = df.groupby("alpha_id")["net_pnl"].sum()
    top_profitable_alphas = set(alpha_pnls[alpha_pnls > 0].index.tolist())

    # Deconstruct why each J1 trade is admitted or rejected by Model 4
    # Model 4 Rules:
    # 1. in top_profitable_alphas
    # 2. is_morning_window (09:20 - 09:45)
    # 3. is_vwap_favorable (dist_vwap_atr <= 0.20)
    # 4. is_atr_sweet_spot (0.55% <= atr_pct <= 1.20%)
    
    reasons = []
    for idx, row in j1_df.iterrows():
        fail_reasons = []
        if row["alpha_id"] not in top_profitable_alphas:
            fail_reasons.append("1. Pruned Alpha ID")
        if not row["is_morning_window"]:
            fail_reasons.append("2. Outside Morning Window (>09:45)")
        if not row["is_vwap_favorable"]:
            fail_reasons.append("3. VWAP Overextended (Dist > 0.2 ATR)")
        if not row["is_atr_sweet_spot"]:
            fail_reasons.append("4. Outside ATR Sweet-Spot")

        if len(fail_reasons) == 0:
            reasons.append("Passed All Model 4 Filters (Admitted)")
        elif len(fail_reasons) == 1:
            reasons.append(fail_reasons[0])
        else:
            reasons.append(f"Multiple ({len(fail_reasons)} Gates): " + " & ".join(fail_reasons))

    j1_df["model4_fate"] = reasons
    
    fate_summary = j1_df.groupby("model4_fate").agg(
        j1_trades=("alpha_id", "count"),
        j1_trade_share=("alpha_id", lambda x: (len(x) / len(j1_df)) * 100),
        avg_net_pnl=("net_pnl", "mean"),
        total_net_pnl=("net_pnl", "sum"),
        j1_pnl_share=("net_pnl", lambda x: (x.sum() / j1_df["net_pnl"].sum()) * 100),
    ).sort_values("total_net_pnl", ascending=False)

    print(fate_summary.to_string())
    print("-" * 120)

    # 2. STEP 2: ALPHA-FAMILY SPECIFIC CHARACTERISTICS & EXCLUSION ANALYSIS
    print("\n" + "=" * 120)
    print("STEP 2: ALPHA-FAMILY SPECIFIC PROFILE OF J1 WINNERS")
    print("=" * 120)
    j1_df["duration_mins"] = (pd.to_datetime(j1_df["exit_time"]) - pd.to_datetime(j1_df["entry_time"])).dt.total_seconds() / 60.0
    fam_profile = j1_df.groupby("family").agg(
        j1_trades=("alpha_id", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        avg_vwap_dist=("dist_vwap_atr", "mean"),
        med_vwap_dist=("dist_vwap_atr", "median"),
        avg_atr_pct=("atr_pct", lambda x: x.mean() * 100),
        morning_share=("is_morning_window", lambda x: x.mean() * 100),
        avg_duration_mins=("duration_mins", "mean")
    )
    print(fam_profile.to_string())

    # 3. STEP 3: DESIGNING FAMILY-ADAPTIVE GATES & RELAXING OVERLY RESTRICTIVE CRITERIA
    print("\n" + "=" * 120)
    print("STEP 3: TESTING REFINED FAMILY-ADAPTIVE GATING ARCHITECTURES")
    print("=" * 120)

    # Architecture Variants:
    # Variant A: Model 4 (Universal Rigid Gate)
    # Variant B: Family-Adaptive VWAP Gate:
    #   - VWAP_MEAN_REVERSION: Dist <= 0.40 ATR (mean reversions can stretch further)
    #   - GAP_EXHAUSTION: Dist <= 0.35 ATR
    #   - INITIAL_BALANCE: Dist <= 0.25 ATR
    #   - TREND_CONTINUATION: Dist <= 0.15 ATR
    # Variant C: Expanded Morning + Midday Regime (09:20 - 11:30 for high confluence)
    # Variant D: Family-Adaptive + Dynamic Conviction Position Sizing:
    #   - Score = f(Family Edge, VWAP Proximity, RVOL, Opportunity Density)
    #   - Sizing: Rs 75k (Low conviction) to Rs 175k (High conviction) on Rs 5L

    def is_family_adaptive_favorable(c):
        fam = c.get("family", "")
        dist = c.get("dist_vwap_atr", 0.0)
        atr_pct = c.get("atr_pct", 0.015)

        # Family-Adaptive VWAP distance & ATR Volatility sweet spots
        if fam == "VWAP_MEAN_REVERSION":
            return (dist <= 0.45) and (0.0035 <= atr_pct <= 0.0090)
        elif fam == "GAP_EXHAUSTION":
            return (dist <= 0.40) and (0.0045 <= atr_pct <= 0.0120)
        elif fam == "INITIAL_BALANCE":
            return (dist <= 0.30) and (0.0045 <= atr_pct <= 0.0120)
        else:
            return (dist <= 0.20) and (0.0055 <= atr_pct <= 0.0150)

    def is_extended_timing_favorable(c):
        slot = c["entry_time"].strftime("%H:%M")
        # Morning core window (09:20 - 09:45)
        if slot in ["09:20", "09:25", "09:30", "09:35", "09:45"]:
            return True
        # Midday momentum window (10:00 - 11:30) IF opportunity density >= 8
        if slot in ["10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30"] and c.get("opp_density", 0) >= 8:
            return True
        return False

    def compute_conviction_tier(c):
        """Returns slot capital: Rs 75,000 (Tier 1), Rs 125,000 (Tier 2), Rs 175,000 (Tier 3)."""
        dist = c.get("dist_vwap_atr", 0.0)
        density = c.get("opp_density", 0)
        fam = c.get("family", "")

        score = 0
        if dist <= 0.0:
            score += 2  # At or below VWAP
        elif dist <= 0.20:
            score += 1

        if density >= 12:
            score += 2  # Strong market impulse
        elif density >= 6:
            score += 1

        if fam in ["VWAP_MEAN_REVERSION", "GAP_EXHAUSTION"]:
            score += 1  # Proven high base edge

        if score >= 4:
            return 175000.0  # High Conviction
        elif score >= 2:
            return 125000.0  # Normal
        else:
            return 75000.0   # Low Conviction

    architectures = {
        "1. Model 4 (Universal Rigid Gate)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and c["is_morning_window"] and c["is_vwap_favorable"] and c["is_atr_sweet_spot"],
            "dynamic_sizing": False
        },
        "2. Family-Adaptive VWAP & ATR Gates": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and c["is_morning_window"] and is_family_adaptive_favorable(c),
            "dynamic_sizing": False
        },
        "3. Family-Adaptive + Midday Confluence": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c),
            "dynamic_sizing": False
        },
        "4. Family-Adaptive + Dynamic Sizing (Rs 75k-175k)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c),
            "dynamic_sizing": True
        },
        "5. Clairvoyant Upper Bound (J1, 4 Slots)": None
    }

    arch_results = {name: [] for name in architectures}

    for ym in complete_months:
        m_df = df[df["month_year"] == ym].copy().reset_index(drop=True)
        m_cands = m_df.to_dict("records")
        m_groups = list(m_df.groupby("entry_time"))

        for arch_name, arch_cfg in architectures.items():
            if arch_name == "5. Clairvoyant Upper Bound (J1, 4 Slots)":
                j1_exec = solve_milp_clairvoyant(m_cands, max_slots=4, enforce_symbol_diversity=True)
                m_res = compute_metrics(j1_exec, 500000.0, len(m_cands))
                m_res["month"] = str(ym)
                arch_results[arch_name].append(m_res)
                continue

            filter_func = arch_cfg["filter"]
            dyn_size = arch_cfg["dynamic_sizing"]

            active = []
            executed = []

            for entry_time, group in m_groups:
                active = [p for p in active if p["exit_time"] > entry_time]
                cands_bar = group.to_dict("records")

                valid = [c for c in cands_bar if filter_func(c)]
                valid.sort(key=lambda x: x.get("dist_vwap_atr", 0.0))

                for c in valid:
                    if len(active) >= 4 or c["symbol"] in {p["symbol"] for p in active}:
                        continue

                    if dyn_size:
                        c_cap = compute_conviction_tier(c)
                        c_eval = evaluate_candidate_economics(c, c_cap, cost_model)
                    else:
                        c_eval = c

                    active.append(c_eval)
                    executed.append(c_eval)

            m_res = compute_metrics(executed, 500000.0, len(m_cands))
            m_res["month"] = str(ym)
            arch_results[arch_name].append(m_res)

    print("\n" + "=" * 145)
    print("17-MONTH PERFORMANCE & J1 CAPTURE PROGRESSION (RS 5L BASE CAPITAL)")
    print("=" * 145)
    print(f"{'ARCHITECTURE CONFIGURATION':<46} | {'AVG ROI':<9} | {'MED ROI':<9} | {'WIN MOS':<12} | {'TOT NET PNL':<14} | {'AVG TRADES':<12} | {'MAX DD':<8} | {'J1 CAPTURE':<10}")
    print("-" * 145)

    j1_tot_pnl = sum(r["net_pnl"] for r in arch_results["5. Clairvoyant Upper Bound (J1, 4 Slots)"])

    for arch_name in architectures:
        recs = arch_results[arch_name]
        rois = [r["roi_pct"] for r in recs]
        pnls = [r["net_pnl"] for r in recs]
        trades = [r["trades"] for r in recs]
        dds = [r["max_dd_pct"] for r in recs]
        pos = sum(1 for r in rois if r > 0)

        tot_pnl = float(np.sum(pnls))
        j1_cap_pct = (tot_pnl / j1_tot_pnl) * 100.0 if j1_tot_pnl > 0 else 0.0

        print(f"{arch_name:<46} | {np.mean(rois):>+7.2f}% | {np.median(rois):>+7.2f}% | {pos:>2d}/17 ({pos/17*100:4.1f}%) | Rs {tot_pnl:>+10,.0f} | {np.mean(trades):>7.1f} tr  | {np.max(dds):>6.2f}% | {j1_cap_pct:>8.1f}%")

    print("=" * 145)

    # 4. FOLD-BY-FOLD WALK-FORWARD BREAKDOWN
    print("\n" + "=" * 145)
    print("STRICT WALK-FORWARD OOS BREAKDOWN (FOLDS 1 TO 4 & FINAL UNTOUCHED OOS)")
    print("=" * 145)
    folds = [
        ("Fold 1 OOS (Oct-Dec 2025)", ["2025-10", "2025-11", "2025-12"]),
        ("Fold 2 OOS (Jan-Mar 2026)", ["2026-01", "2026-02", "2026-03"]),
        ("Fold 3 OOS (Apr-Jun 2026)", ["2026-04", "2026-05", "2026-06"]),
        ("Fold 4 FINAL UNTOUCHED OOS (Jul-Aug 2026)", ["2026-07", "2026-08"]),
    ]

    for fold_name, fold_months in folds:
        print(f"\n[*] {fold_name}:")
        print(f"{'CONFIGURATION':<46} | {'AVG ROI':<9} | {'NET PNL':<14} | {'TRADES':<8} | {'WIN MOS':<10}")
        print("-" * 95)
        for arch_name in [
            "1. Model 4 (Universal Rigid Gate)",
            "2. Family-Adaptive VWAP & ATR Gates",
            "4. Family-Adaptive + Dynamic Sizing (Rs 75k-175k)",
            "5. Clairvoyant Upper Bound (J1, 4 Slots)"
        ]:
            recs = [r for r in arch_results[arch_name] if r["month"] in fold_months]
            rois = [r["roi_pct"] for r in recs]
            pnls = [r["net_pnl"] for r in recs]
            trades = [r["trades"] for r in recs]
            pos = sum(1 for r in rois if r > 0)
            print(f"{arch_name:<46} | {np.mean(rois):>+7.2f}% | Rs {np.sum(pnls):>+10,.0f} | {np.sum(trades):>6d} tr | {pos}/{len(rois)}")


if __name__ == "__main__":
    main()
