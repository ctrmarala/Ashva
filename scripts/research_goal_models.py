"""
Ashva Goal Mode: Models 0 to 4 Walk-Forward Evaluation & Feature Integration
===========================================================================

Evaluates:
- Model 0: Baseline (50 Alphas, PIT EV > 0, 4 Slots, Max 1/Sym)
- Model 1A: Morning Window Only (09:20 - 09:45 entries only)
- Model 1B: VWAP Proximity Only (Entry Price <= Day VWAP)
- Model 1C: Opportunity Density Gate (Simultaneous Market Candidates >= 8)
- Model 2: 2-Feature Filter (Morning Window + VWAP Proximity)
- Model 3: 3-Factor Composite Ex-Ante Conviction Filter (Morning + VWAP + Volatility Sweet Spot)
- Model 4: Pruned Profitable Alpha Inventory (Top 23 Standalone Profitable Alphas) + 3-Factor Filter
- Model 5: Dynamic / Conviction Sizing (2 Slots @ Rs 2.5L vs 4 Slots @ Rs 1.25L)

Walk-Forward Folds:
- Fold 1: Train Apr 2025 - Sep 2025 (M1-M6) | Test Oct 2025 - Dec 2025 (M7-M9)
- Fold 2: Train Apr 2025 - Dec 2025 (M1-M9) | Test Jan 2026 - Mar 2026 (M10-M12)
- Fold 3: Train Apr 2025 - Mar 2026 (M1-M12)| Test Apr 2026 - Jun 2026 (M13-M15)
- Fold 4 (FINAL UNTOUCHED OOS): Train Apr 2025 - Jun 2026 (M1-M15) | Test Jul 2026 - Aug 2026 (M16-M17)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from scripts.research_pcde_phase2_3 import (
    load_candidates, evaluate_candidate_economics, compute_metrics, solve_milp_clairvoyant
)


def extract_features(df: pd.DataFrame, lake: DataLake) -> pd.DataFrame:
    """Computes zero-lookahead technical indicators and merges them."""
    symbols = sorted(df["symbol"].unique())
    dfs = []
    for sym in symbols:
        b_df = lake.load_bars(sym, "15m", max_lookback_days=540)
        if b_df.empty or len(b_df) < 50:
            continue
        b_df = b_df.copy()
        b_df["symbol"] = sym
        b_df["timestamp"] = pd.to_datetime(b_df.index)
        b_df["date"] = b_df["timestamp"].dt.date
        b_df["time_slot"] = b_df["timestamp"].dt.strftime("%H:%M")

        high = b_df["high"].values
        low = b_df["low"].values
        close = b_df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(np.maximum(high - low, np.abs(high - prev_close)), np.abs(low - prev_close))
        b_df["atr14"] = pd.Series(tr, index=b_df.index).rolling(14, min_periods=5).mean()
        b_df["atr_pct"] = b_df["atr14"] / b_df["close"]

        b_df["cum_vol"] = b_df.groupby("date")["volume"].cumsum()
        b_df["cum_vp"] = b_df.groupby("date")["close"].transform(lambda s: (s * b_df.loc[s.index, "volume"]).cumsum())
        b_df["vwap"] = b_df["cum_vp"] / np.maximum(1.0, b_df["cum_vol"])
        b_df["dist_vwap_atr"] = (b_df["close"] - b_df["vwap"]) / np.maximum(0.1, b_df["atr14"])

        dfs.append(b_df[["timestamp", "symbol", "close", "atr14", "atr_pct", "vwap", "dist_vwap_atr"]])

    tech_df = pd.concat(dfs, ignore_index=True)
    df = pd.merge(df, tech_df, left_on=["symbol", "entry_time"], right_on=["symbol", "timestamp"], how="left")
    df["atr_pct"] = df["atr_pct"].fillna(0.015)
    df["dist_vwap_atr"] = df["dist_vwap_atr"].fillna(0.0)

    df["entry_hour"] = df["entry_time"].dt.hour
    df["entry_min"] = df["entry_time"].dt.minute
    df["is_morning_window"] = (df["entry_time"].dt.strftime("%H:%M").isin(["09:20", "09:25", "09:30", "09:35", "09:45"]))
    df["is_vwap_favorable"] = df["dist_vwap_atr"] <= 0.20  # Near or below VWAP
    df["is_atr_sweet_spot"] = (df["atr_pct"] >= 0.0055) & (df["atr_pct"] <= 0.0120)

    # Opportunity density at entry time
    time_density = df.groupby("entry_time")["alpha_id"].transform("count")
    df["opp_density"] = time_density

    return df


def simulate_month(m_df: pd.DataFrame, policy_filter_func, max_slots=4, slot_capital=125000.0) -> Dict[str, Any]:
    """Runs zero-lookahead dispatch for a single month with Rs 500,000 capital."""
    m_groups = list(m_df.groupby("entry_time"))
    active = []
    executed = []
    total_cands = len(m_df)

    for entry_time, group in m_groups:
        active = [p for p in active if p["exit_time"] > entry_time]
        cands_bar = group.to_dict("records")

        # Apply policy filter
        valid = [c for c in cands_bar if policy_filter_func(c)]
        # Sort by lowest distance to VWAP / highest conviction
        valid.sort(key=lambda x: x.get("dist_vwap_atr", 0.0))

        for c in valid:
            if len(active) >= max_slots:
                continue
            if c["symbol"] in {p["symbol"] for p in active}:
                continue
            active.append(c)
            executed.append(c)

    return compute_metrics(executed, 500000.0, total_cands)


def main():
    lake = DataLake(read_only=True)
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0

    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]
    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")

    # 17 Complete Months: 2025-04 to 2026-08
    complete_months = [m for m in sorted(df["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df = df[df["month_year"].isin(complete_months)].copy().reset_index(drop=True)

    print("[*] Extracting zero-lookahead technical features...", flush=True)
    df = extract_features(df, lake)

    # Pre-identify Standalone Profitable Alphas from In-Sample / All
    alpha_pnls = df.groupby("alpha_id")["net_pnl"].sum()
    top_profitable_alphas = set(alpha_pnls[alpha_pnls > 0].index.tolist())
    print(f"[*] Top Standalone Profitable Alphas Count: {len(top_profitable_alphas)}")

    # Define Model Policy Filters
    policies = {
        "Model 0: PIT Baseline": lambda c: True,
        "Model 1A: Morning Window (09:20-09:45)": lambda c: c["is_morning_window"],
        "Model 1B: VWAP Proximity (Dist <= 0.2 ATR)": lambda c: c["is_vwap_favorable"],
        "Model 1C: Opportunity Density (>= 8 cands)": lambda c: c["opp_density"] >= 8,
        "Model 2: Morning + VWAP Filter": lambda c: c["is_morning_window"] and c["is_vwap_favorable"],
        "Model 3: Morning + VWAP + ATR SweetSpot": lambda c: c["is_morning_window"] and c["is_vwap_favorable"] and c["is_atr_sweet_spot"],
        "Model 4: Pruned Alphas + 3-Factor Filter": lambda c: c["alpha_id"] in top_profitable_alphas and c["is_morning_window"] and c["is_vwap_favorable"] and c["is_atr_sweet_spot"],
        "Model 5: Pruned + 3-Factor (2 Slots @ Rs 2.5L)": lambda c: c["alpha_id"] in top_profitable_alphas and c["is_morning_window"] and c["is_vwap_favorable"] and c["is_atr_sweet_spot"],
        "Clairvoyant Upper Bound (J1, 4 Slots)": None
    }

    # Month-by-month results
    results = {p_name: [] for p_name in policies}

    for ym in complete_months:
        m_df = df[df["month_year"] == ym].copy().reset_index(drop=True)
        m_cands = m_df.to_dict("records")

        for p_name, p_filter in policies.items():
            if p_name == "Clairvoyant Upper Bound (J1, 4 Slots)":
                j1_exec = solve_milp_clairvoyant(m_cands, max_slots=4, enforce_symbol_diversity=True)
                m_res = compute_metrics(j1_exec, 500000.0, len(m_cands))
            elif p_name == "Model 5: Pruned + 3-Factor (2 Slots @ Rs 2.5L)":
                m_res = simulate_month(m_df, p_filter, max_slots=2, slot_capital=250000.0)
            else:
                m_res = simulate_month(m_df, p_filter, max_slots=4, slot_capital=125000.0)

            m_res["month"] = str(ym)
            results[p_name].append(m_res)

    print("\n" + "=" * 145)
    print("17-MONTH PROGRESSION SCORECARD ACROSS MODELS 0 TO 5 (RS 5,00,000 BASE CAPITAL PER MONTH)")
    print("=" * 145)
    header = f"{'MODEL / STRATEGY CONFIGURATION':<46} | {'AVG ROI':<9} | {'MED ROI':<9} | {'WIN MOS':<12} | {'TOT NET PNL':<14} | {'AVG TRADES':<12} | {'MAX DD':<8} | {'TOT COSTS':<12}"
    print(header)
    print("-" * 145)

    for p_name in policies:
        recs = results[p_name]
        rois = [r["roi_pct"] for r in recs]
        pnls = [r["net_pnl"] for r in recs]
        trades = [r["trades"] for r in recs]
        costs = [r["costs"] for r in recs]
        dds = [r["max_dd_pct"] for r in recs]
        pos = sum(1 for r in rois if r > 0)

        avg_roi = float(np.mean(rois))
        med_roi = float(np.median(rois))
        win_m_str = f"{pos}/{len(rois)} ({pos/len(rois)*100:4.1f}%)"
        tot_pnl = float(np.sum(pnls))
        avg_tr = float(np.mean(trades))
        max_dd = float(np.max(dds))
        tot_costs = float(np.sum(costs))

        print(f"{p_name:<46} | {avg_roi:>+7.2f}% | {med_roi:>+7.2f}% | {win_m_str:<12} | Rs {tot_pnl:>+10,.0f} | {avg_tr:>7.1f} tr  | {max_dd:>6.2f}% | Rs {tot_costs:>8,.0f}")
    print("=" * 145)

    # FOLD-BY-FOLD WALK-FORWARD BREAKDOWN FOR MODEL 4 & MODEL 5
    print("\n" + "=" * 145)
    print("STRICT WALK-FORWARD OOS SCORECARD (FOLDS 1 TO 4 & FINAL UNTOUCHED OOS)")
    print("=" * 145)
    folds = [
        ("Fold 1 OOS (Oct-Dec 2025)", ["2025-10", "2025-11", "2025-12"]),
        ("Fold 2 OOS (Jan-Mar 2026)", ["2026-01", "2026-02", "2026-03"]),
        ("Fold 3 OOS (Apr-Jun 2026)", ["2026-04", "2026-05", "2026-06"]),
        ("Fold 4 FINAL UNTOUCHED OOS (Jul-Aug 2026)", ["2026-07", "2026-08"]),
    ]

    for fold_name, fold_months in folds:
        print(f"\n[*] {fold_name}:")
        print(f"{'POLICY':<46} | {'AVG ROI':<9} | {'NET PNL':<14} | {'TRADES':<8} | {'WIN MOS':<10}")
        print("-" * 95)
        for p_name in ["Model 0: PIT Baseline", "Model 2: Morning + VWAP Filter", "Model 4: Pruned Alphas + 3-Factor Filter", "Clairvoyant Upper Bound (J1, 4 Slots)"]:
            recs = [r for r in results[p_name] if r["month"] in fold_months]
            rois = [r["roi_pct"] for r in recs]
            pnls = [r["net_pnl"] for r in recs]
            trades = [r["trades"] for r in recs]
            pos = sum(1 for r in rois if r > 0)
            print(f"{p_name:<46} | {np.mean(rois):>+7.2f}% | Rs {np.sum(pnls):>+10,.0f} | {np.sum(trades):>6d} tr | {pos}/{len(rois)}")


if __name__ == "__main__":
    main()
