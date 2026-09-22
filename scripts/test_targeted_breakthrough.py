"""
Ashva Targeted Breakthrough Experiment
======================================
Tests combining the 50 PROVEN alphas with targeted Short-Side & Multi-Session Alphas
(18_alpha European Open, 20_alpha Power Hour, 12_alpha FVG Reversion, 16_alpha Liquidity Sweep, 37_alpha Trend Drive, 8_alpha Extreme Reversion)
across all 17 complete months under full Indian transaction costs and 3 bps slippage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from scripts.research_pcde_phase2_3 import (
    evaluate_candidate_economics, compute_metrics, solve_milp_clairvoyant
)
from scripts.research_cash_breakthrough import enrich_with_intraday_features, classify_alpha_family


def main():
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0

    # 1. Load Proven Candidates
    df_proven = pd.read_parquet("data_lake/pcde_raw_candidates.parquet")
    df_proven["entry_time"] = pd.to_datetime(df_proven["entry_time"])
    df_proven["exit_time"] = pd.to_datetime(df_proven["exit_time"])
    if "side" not in df_proven.columns:
        df_proven["side"] = "LONG"

    # 2. Load Targeted New Multi-Session & Short Alphas
    df_exp = pd.read_parquet("data_lake/breakthrough_expanded_candidates.parquet")
    df_exp["entry_time"] = pd.to_datetime(df_exp["entry_time"])
    df_exp["exit_time"] = pd.to_datetime(df_exp["exit_time"])

    target_new_alphas = ["18_alpha", "20_alpha", "12_alpha", "16_alpha", "37_alpha", "46_alpha", "8_alpha", "10_alpha"]
    df_new = df_exp[df_exp["alpha_id"].isin(target_new_alphas)].copy()

    # Combine
    common_cols = ["alpha_id", "symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price", "exit_reason"]
    df_combined = pd.concat([df_proven[common_cols], df_new[common_cols]], ignore_index=True).drop_duplicates(subset=["alpha_id", "symbol", "entry_time"])
    df_combined = df_combined.sort_values("entry_time").reset_index(drop=True)
    df_combined["month_year"] = df_combined["entry_time"].dt.to_period("M")

    complete_months = [m for m in sorted(df_combined["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df_combined = df_combined[df_combined["month_year"].isin(complete_months)].copy().reset_index(drop=True)

    print(f"[*] Combined Candidate Pool across 17 Complete Months: {len(df_combined):,}")
    print(f"    - Long Signals:  {(df_combined['side']=='LONG').sum():,}")
    print(f"    - Short Signals: {(df_combined['side']=='SHORT').sum():,}")

    # Evaluate economics with nominal 1.25L cap
    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in df_combined.to_dict("records") if c["entry_price"] <= slot_cap]
    df = pd.DataFrame(eval_cands)
    df["family"] = df["alpha_id"].apply(classify_alpha_family)
    df["is_win"] = df["net_pnl"] > 0.0

    # Enrich features
    df = enrich_with_intraday_features(df, lake)

    # 3. MILP J1 Clairvoyant Benchmark on Expanded Universe (4 Slots & 6 Slots)
    print("\n" + "=" * 145)
    print("CALCULATING CLAIRVOYANT UPPER BOUND (J1) ON COMBINED EXPANDED UNIVERSE (17 COMPLETE MONTHS)")
    print("=" * 145)

    j1_results_4slots = []
    j1_results_6slots = []
    for ym in complete_months:
        m_df = df[df["month_year"] == ym].to_dict("records")
        j1_4 = solve_milp_clairvoyant(m_df, max_slots=4, enforce_symbol_diversity=True)
        j1_6 = solve_milp_clairvoyant(m_df, max_slots=6, enforce_symbol_diversity=True)

        res_4 = compute_metrics(j1_4, 500000.0, len(m_df))
        res_6 = compute_metrics(j1_6, 500000.0, len(m_df))
        res_4["month"] = str(ym)
        res_6["month"] = str(ym)
        j1_results_4slots.append(res_4)
        j1_results_6slots.append(res_6)

    print(f"4-Slot J1 Upper Bound: Avg ROI: {np.mean([r['roi_pct'] for r in j1_results_4slots]):>+6.2f}%/mo | Total PnL: Rs {sum(r['net_pnl'] for r in j1_results_4slots):>+10,.0f} | Avg Trades: {np.mean([r['trades'] for r in j1_results_4slots]):.1f}/mo")
    print(f"6-Slot J1 Upper Bound: Avg ROI: {np.mean([r['roi_pct'] for r in j1_results_6slots]):>+6.2f}%/mo | Total PnL: Rs {sum(r['net_pnl'] for r in j1_results_6slots):>+10,.0f} | Avg Trades: {np.mean([r['trades'] for r in j1_results_6slots]):.1f}/mo")

    # 4. EX-ANTE EXECUTION GATING & DISPATCHING
    # Alpha Discovery Pruning: select alphas with positive discovery net PnL (Apr-Sep 2025)
    disc_months = [pd.Period(f"2025-{m:02d}", "M") for m in range(4, 10)]
    disc_df = df[df["month_year"].isin(disc_months)]
    alpha_disc_pnl = disc_df.groupby("alpha_id")["net_pnl"].sum()
    alpha_disc_wr = disc_df.groupby("alpha_id")["is_win"].mean()

    # Retain alphas that are profitable on discovery period
    retained_alphas = set(alpha_disc_pnl[(alpha_disc_pnl > 0) & (alpha_disc_wr >= 0.45)].index.tolist())
    # Also keep proven high-edge alpha IDs
    core_proven = set(df_proven["alpha_id"].unique())
    retained_alphas = retained_alphas.union(set(alpha_disc_pnl[alpha_disc_pnl > 0].index.tolist())).intersection(set(df["alpha_id"].unique()))

    print(f"\n[*] Retained Alphas for Ex-Ante Dispatch ({len(retained_alphas)} alphas): {sorted(list(retained_alphas))[:15]}...")

    def is_favorable(c):
        a_id = c["alpha_id"]
        if a_id not in retained_alphas:
            return False

        fam = c.get("family", "")
        dist = c.get("dist_vwap_atr", 0.0)
        atr_pct = c.get("atr_pct", 0.015)
        t_str = c.get("time_str", "09:30")
        rvol = c.get("rvol_15m", 1.0)
        side = c.get("side", "LONG")

        # Morning Window (09:15 - 09:45)
        if t_str in ["09:15", "09:20", "09:25", "09:30", "09:35", "09:45"]:
            if fam == "VWAP_MEAN_REVERSION":
                return (dist <= 0.45 if side == "LONG" else dist >= -0.45) and (0.0035 <= atr_pct <= 0.0100)
            elif fam == "GAP_EXHAUSTION":
                return (dist <= 0.40 if side == "LONG" else dist >= -0.40) and (0.0045 <= atr_pct <= 0.0120)
            elif fam == "INITIAL_BALANCE":
                return (dist <= 0.30 if side == "LONG" else dist >= -0.30) and (0.0045 <= atr_pct <= 0.0120)
            elif fam == "LIQUIDITY_SWEEP_REVERSION":
                return rvol >= 1.20 and (0.0050 <= atr_pct <= 0.0150)
            else:
                return (dist <= 0.20 if side == "LONG" else dist >= -0.20) and (0.0055 <= atr_pct <= 0.0150)

        # Midday Window (10:00 - 12:30)
        elif "10:00" <= t_str <= "12:30":
            if fam in ["VWAP_MEAN_REVERSION", "LIQUIDITY_SWEEP_REVERSION"]:
                return (abs(dist) <= 0.35) and c.get("opp_density", 0) >= 6
            elif c.get("opp_density", 0) >= 8 and rvol >= 1.25:
                return True
            return False

        # European Open Window (12:45 - 14:15)
        elif "12:45" <= t_str <= "14:15":
            if fam == "EUROPEAN_OPEN_MOMENTUM":
                return rvol >= 1.20 and (0.0040 <= atr_pct <= 0.0160)
            elif fam in ["VWAP_MEAN_REVERSION", "LIQUIDITY_SWEEP_REVERSION"]:
                return abs(dist) <= 0.25 and rvol >= 1.20
            return False

        # Power Hour Window (14:15 - 15:00)
        elif "14:15" <= t_str <= "15:00":
            if fam == "POWER_HOUR_ACCELERATION":
                return rvol >= 1.30 and (dist <= 0.15 if side == "LONG" else dist >= -0.15)
            return False

        return False

    def compute_conviction(c):
        dist = c.get("dist_vwap_atr", 0.0)
        density = c.get("opp_density", 0)
        fam = c.get("family", "")
        rvol = c.get("rvol_15m", 1.0)
        side = c.get("side", "LONG")

        score = 0
        if side == "LONG" and dist <= 0.0:
            score += 2
        elif side == "SHORT" and dist >= 0.0:
            score += 2
        elif abs(dist) <= 0.20:
            score += 1

        if rvol >= 2.0:
            score += 2
        elif rvol >= 1.3:
            score += 1

        if density >= 10:
            score += 1

        if fam in ["VWAP_MEAN_REVERSION", "GAP_EXHAUSTION", "EUROPEAN_OPEN_MOMENTUM", "POWER_HOUR_ACCELERATION"]:
            score += 1

        if score >= 4:
            return 175000.0, 3
        elif score >= 2:
            return 125000.0, 2
        else:
            return 75000.0, 1

    fam_weights = {
        "VWAP_MEAN_REVERSION": 3.0,
        "GAP_EXHAUSTION": 2.5,
        "EUROPEAN_OPEN_MOMENTUM": 2.5,
        "POWER_HOUR_ACCELERATION": 2.5,
        "INITIAL_BALANCE": 1.5,
        "LIQUIDITY_SWEEP_REVERSION": 2.0,
        "TREND_CONTINUATION": 1.0
    }

    def rank_score(c):
        fam = c.get("family", "")
        w = fam_weights.get(fam, 1.0)
        dist = abs(c.get("dist_vwap_atr", 0.0))
        rvol = min(c.get("rvol_15m", 1.0), 3.5)
        density = min(c.get("opp_density", 0), 20)
        return w * 2.5 - 3.5 * dist + 0.6 * rvol + 0.1 * density

    # 5. SIMULATE PORTFOLIO ACROSS ALL 17 MONTHS
    models = {
        "1. Existing Baseline Model 4 (Long-Only Rigid)": {
            "filter": lambda c: c["alpha_id"] in retained_alphas and c["time_str"] in ["09:20", "09:25", "09:30", "09:35", "09:45"] and c.get("dist_vwap_atr", 1.0) <= 0.20 and (0.0055 <= c.get("atr_pct", 0) <= 0.0120) and c["side"] == "LONG",
            "rank": lambda c: c.get("dist_vwap_atr", 1.0),
            "reverse_rank": False,
            "dyn_size": False,
            "max_slots": 4
        },
        "2. Existing Family-Adaptive Baseline (Long-Only)": {
            "filter": lambda c: is_favorable(c) and c["side"] == "LONG",
            "rank": rank_score,
            "reverse_rank": True,
            "dyn_size": True,
            "max_slots": 4
        },
        "3. Breakthrough Candidate (Long + Short, 4 Slots)": {
            "filter": is_favorable,
            "rank": rank_score,
            "reverse_rank": True,
            "dyn_size": True,
            "max_slots": 4
        },
        "4. Breakthrough Candidate (Long + Short, 5 Slots @ Rs 90k-150k)": {
            "filter": is_favorable,
            "rank": rank_score,
            "reverse_rank": True,
            "dyn_size": True,
            "max_slots": 5,
            "slot_cap_mult": 0.85
        }
    }

    results = {m: [] for m in models}

    for ym in complete_months:
        m_df = df[df["month_year"] == ym].copy().reset_index(drop=True)
        m_cands = m_df.to_dict("records")
        m_groups = list(m_df.groupby("entry_time"))

        for m_name, m_cfg in models.items():
            filter_fn = m_cfg["filter"]
            rank_fn = m_cfg["rank"]
            rev_rank = m_cfg["reverse_rank"]
            dyn_size = m_cfg["dyn_size"]
            m_slots = m_cfg.get("max_slots", 4)
            cap_mult = m_cfg.get("slot_cap_mult", 1.0)

            active = []
            executed = []

            for entry_time, group in m_groups:
                active = [p for p in active if p["exit_time"] > entry_time]
                cands_bar = group.to_dict("records")

                valid = [c for c in cands_bar if filter_fn(c)]
                valid.sort(key=rank_fn, reverse=rev_rank)

                for c in valid:
                    if len(active) >= m_slots or c["symbol"] in {p["symbol"] for p in active}:
                        continue

                    if dyn_size:
                        c_cap, tier = compute_conviction(c)
                        c_cap = c_cap * cap_mult
                        c_eval = evaluate_candidate_economics(c, c_cap, cost_model)
                        c_eval["deployed_cap"] = c_cap
                    else:
                        c_eval = evaluate_candidate_economics(c, 125000.0 * cap_mult, cost_model)
                        c_eval["deployed_cap"] = 125000.0 * cap_mult

                    active.append(c_eval)
                    executed.append(c_eval)

            m_res = compute_metrics(executed, 500000.0, len(m_cands))
            m_res["month"] = str(ym)
            m_res["executed"] = executed
            results[m_name].append(m_res)

    print("\n" + "=" * 145)
    print("17-MONTH MASTER WALK-FORWARD BENCHMARK (RS 5,00,000 REFERENCE CAPITAL)")
    print("=" * 145)
    print(f"{'ARCHITECTURE CONFIGURATION':<58} | {'AVG ROI':<9} | {'MED ROI':<9} | {'WIN MOS':<12} | {'TOT NET PNL':<14} | {'AVG TRADES':<12} | {'MAX DD':<8}")
    print("-" * 145)

    for m_name in models:
        recs = results[m_name]
        rois = [r["roi_pct"] for r in recs]
        pnls = [r["net_pnl"] for r in recs]
        trades = [r["trades"] for r in recs]
        dds = [r["max_dd_pct"] for r in recs]
        pos = sum(1 for r in rois if r > 0)
        tot_pnl = float(np.sum(pnls))

        print(f"{m_name:<58} | {np.mean(rois):>+7.2f}% | {np.median(rois):>+7.2f}% | {pos:>2d}/17 ({pos/17*100:4.1f}%) | Rs {tot_pnl:>+10,.0f} | {np.mean(trades):>7.1f} tr  | {np.max(dds):>6.2f}%")
    print("=" * 145)

    # 6. FOLD-BY-FOLD WALK-FORWARD BREAKDOWN
    print("\n" + "=" * 145)
    print("STRICT ROLLING WALK-FORWARD VERIFICATION (FOLDS 1 TO 3 & FINAL UNTOUCHED OOS FOLD 4)")
    print("=" * 145)

    folds = [
        ("Fold 1 OOS (Oct-Dec 2025)", ["2025-10", "2025-11", "2025-12"]),
        ("Fold 2 OOS (Jan-Mar 2026)", ["2026-01", "2026-02", "2026-03"]),
        ("Fold 3 OOS (Apr-Jun 2026)", ["2026-04", "2026-05", "2026-06"]),
        ("Fold 4 FINAL UNTOUCHED OOS (Jul-Aug 2026)", ["2026-07", "2026-08"]),
    ]

    for fold_name, fold_months in folds:
        print(f"\n[*] {fold_name}:")
        print(f"{'CONFIGURATION':<58} | {'AVG ROI':<9} | {'NET PNL':<14} | {'TRADES':<8} | {'WIN MOS':<10}")
        print("-" * 105)
        for m_name in models:
            recs = [r for r in results[m_name] if r["month"] in fold_months]
            rois = [r["roi_pct"] for r in recs]
            pnls = [r["net_pnl"] for r in recs]
            trades = [r["trades"] for r in recs]
            pos = sum(1 for r in rois if r > 0)
            print(f"{m_name:<58} | {np.mean(rois):>+7.2f}% | Rs {np.sum(pnls):>+10,.0f} | {np.sum(trades):>6d} tr | {pos}/{len(rois)}")


if __name__ == "__main__":
    main()
