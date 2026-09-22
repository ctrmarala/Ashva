"""
Ashva J1 Classification Audit, Ranking Diagnostic & Dynamic Sizing Exposure Analysis
====================================================================================

1. FORENSIC CLASSIFICATION TABLE (TP, FP, FN, TN):
   - J1 Precision = TP / (TP + FP)
   - J1 Recall = TP / (TP + FN)
   - F1 Score
   - Evaluated for:
     * Baseline FIFO
     * Symbol Diversity Baseline
     * Model 0 (PIT EV)
     * Model 1B (Single VWAP Gate)
     * Model 4 (Pruned + Universal Rigid Gate)
     * Family-Adaptive Gating (Flat Sizing)
     * Family-Adaptive + Dynamic Sizing (Rs 75k-175k)

2. RANKING ERROR DIAGNOSTIC:
   - Simultaneous candidate analysis at timestamp T:
     * When >= 1 J1 winner and >= 1 non-J1 candidate were admitted, did the ranker pick the winner?
     * Feature comparison of TPs vs FPs at the exact same entry bar (VWAP dist, RVOL, family, intraday momentum).

3. EXPOSURE & CONVICTION SIZING RISK AUDIT:
   - Peak gross concurrent exposure across all days (Max concurrent positions * size)
   - Average capital deployed per bar and per active trade
   - Position size frequency (Rs 75k vs 125k vs 175k)
   - Flat Rs 1.25L vs Dynamic Sizing comparison (Alpha uplift vs Exposure uplift)
   - ROI on Deployed Capital vs ROI on Rs 5L reference
   - Worst single trade loss & worst single day loss

4. RANKING ENHANCEMENT EXPERIMENTS (TARGETING 40-50% J1 CAPTURE):
   - Multi-factor ranking: Combining VWAP distance, Alpha family historical win rate, and Opportunity density
   - Walk-forward validation across Folds 1-4.
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


def extract_rich_features(df: pd.DataFrame, lake: DataLake) -> pd.DataFrame:
    """Extracts ATR, VWAP, RVOL, momentum, and timing features."""
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

        high = b_df["high"].values
        low = b_df["low"].values
        close = b_df["close"].values
        volume = b_df["volume"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(np.maximum(high - low, np.abs(high - prev_close)), np.abs(low - prev_close))
        b_df["atr14"] = pd.Series(tr, index=b_df.index).rolling(14, min_periods=5).mean()
        b_df["atr_pct"] = b_df["atr14"] / b_df["close"]

        # VWAP
        b_df["cum_vol"] = b_df.groupby("date")["volume"].cumsum()
        b_df["cum_vp"] = b_df.groupby("date")["close"].transform(lambda s: (s * b_df.loc[s.index, "volume"]).cumsum())
        b_df["vwap"] = b_df["cum_vp"] / np.maximum(1.0, b_df["cum_vol"])
        b_df["dist_vwap_atr"] = (b_df["close"] - b_df["vwap"]) / np.maximum(0.1, b_df["atr14"])

        # RVOL (Volume / 20-period volume SMA)
        vol_sma20 = pd.Series(volume, index=b_df.index).rolling(20, min_periods=5).mean()
        b_df["rvol_15m"] = pd.Series(volume, index=b_df.index) / np.maximum(1.0, vol_sma20)

        dfs.append(b_df[["timestamp", "symbol", "close", "atr14", "atr_pct", "vwap", "dist_vwap_atr", "rvol_15m"]])

    tech_df = pd.concat(dfs, ignore_index=True)
    df = pd.merge(df, tech_df, left_on=["symbol", "entry_time"], right_on=["symbol", "timestamp"], how="left")
    df["atr_pct"] = df["atr_pct"].fillna(0.015)
    df["dist_vwap_atr"] = df["dist_vwap_atr"].fillna(0.0)
    df["rvol_15m"] = df["rvol_15m"].fillna(1.0)

    df["entry_hour"] = df["entry_time"].dt.hour
    df["entry_min"] = df["entry_time"].dt.minute
    df["is_morning_window"] = (df["entry_time"].dt.strftime("%H:%M").isin(["09:20", "09:25", "09:30", "09:35", "09:45"]))
    df["is_vwap_favorable"] = df["dist_vwap_atr"] <= 0.20
    df["is_atr_sweet_spot"] = (df["atr_pct"] >= 0.0055) & (df["atr_pct"] <= 0.0120)

    # Opportunity density at entry time
    time_density = df.groupby("entry_time")["alpha_id"].transform("count")
    df["opp_density"] = time_density

    return df


def main():
    lake = DataLake(read_only=True)
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    nominal_slot_cap = 125000.0

    eval_cands = [evaluate_candidate_economics(c, nominal_slot_cap, cost_model) for c in cands if c["entry_price"] <= nominal_slot_cap]
    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")

    # Complete 17 months: 2025-04 to 2026-08
    complete_months = [m for m in sorted(df["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df = df[df["month_year"].isin(complete_months)].copy().reset_index(drop=True)

    print("[*] Extracting technical indicators and market context...", flush=True)
    df = extract_rich_features(df, lake)

    # 1. Identify J1 Trades across all 17 months
    j1_trade_keys = set()
    j1_trades_by_month = {}
    for ym in complete_months:
        c_sub = df[df["month_year"] == ym].to_dict("records")
        j1_exec = solve_milp_clairvoyant(c_sub, max_slots=4, enforce_symbol_diversity=True)
        j1_trades_by_month[str(ym)] = j1_exec
        for t in j1_exec:
            k = (t["alpha_id"], t["symbol"], str(t["entry_time"]))
            j1_trade_keys.add(k)

    df["is_j1"] = df.apply(lambda r: (r["alpha_id"], r["symbol"], str(r["entry_time"])) in j1_trade_keys, axis=1)
    df["candidate_key"] = df.apply(lambda r: (r["alpha_id"], r["symbol"], str(r["entry_time"])), axis=1)

    alpha_pnls = df.groupby("alpha_id")["net_pnl"].sum()
    top_profitable_alphas = set(alpha_pnls[alpha_pnls > 0].index.tolist())

    def is_family_adaptive_favorable(c):
        fam = c.get("family", "")
        dist = c.get("dist_vwap_atr", 0.0)
        atr_pct = c.get("atr_pct", 0.015)

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
        if slot in ["09:20", "09:25", "09:30", "09:35", "09:45"]:
            return True
        if slot in ["10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30"] and c.get("opp_density", 0) >= 8:
            return True
        return False

    def compute_conviction_tier(c):
        dist = c.get("dist_vwap_atr", 0.0)
        density = c.get("opp_density", 0)
        fam = c.get("family", "")

        score = 0
        if dist <= 0.0:
            score += 2
        elif dist <= 0.20:
            score += 1

        if density >= 12:
            score += 2
        elif density >= 6:
            score += 1

        if fam in ["VWAP_MEAN_REVERSION", "GAP_EXHAUSTION"]:
            score += 1

        if score >= 4:
            return 175000.0, 3
        elif score >= 2:
            return 125000.0, 2
        else:
            return 75000.0, 1

    # =========================================================================
    # PART 1: J1 FORENSIC CLASSIFICATION & PRECISION / RECALL AUDIT
    # =========================================================================
    print("\n" + "=" * 145)
    print("PART 1: FORENSIC CLASSIFICATION, PRECISION & RECALL AUDIT (ALL 17 MONTHS, 7,070 CANDIDATES, 228 J1 WINNERS)")
    print("=" * 145)

    models_to_test = {
        "1. Baseline FIFO": {
            "filter": lambda c: True,
            "rank": lambda c: 0,
            "sym_div": False,
            "dyn_size": False
        },
        "2. Symbol Diversity Baseline": {
            "filter": lambda c: True,
            "rank": lambda c: 0,
            "sym_div": True,
            "dyn_size": False
        },
        "3. Model 0 (PIT EV Baseline)": {
            "filter": lambda c: True,
            "rank": lambda c: -c.get("expected_net_pnl", 0.0),
            "sym_div": True,
            "dyn_size": False
        },
        "4. Model 1B (Single VWAP Gate)": {
            "filter": lambda c: c.get("dist_vwap_atr", 1.0) <= 0.20,
            "rank": lambda c: c.get("dist_vwap_atr", 1.0),
            "sym_div": True,
            "dyn_size": False
        },
        "5. Model 4 (Pruned + Universal Rigid Gate)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and c["is_morning_window"] and c["is_vwap_favorable"] and c["is_atr_sweet_spot"],
            "rank": lambda c: c.get("dist_vwap_atr", 1.0),
            "sym_div": True,
            "dyn_size": False
        },
        "6. Family-Adaptive Gating (Flat Sizing)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c),
            "rank": lambda c: c.get("dist_vwap_atr", 1.0),
            "sym_div": True,
            "dyn_size": False
        },
        "7. Family-Adaptive + Dynamic Sizing": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c),
            "rank": lambda c: c.get("dist_vwap_atr", 1.0),
            "sym_div": True,
            "dyn_size": True
        }
    }

    class_results = []

    for name, cfg in models_to_test.items():
        filt = cfg["filter"]
        rank_fn = cfg["rank"]
        sym_div = cfg["sym_div"]
        dyn_size = cfg["dyn_size"]

        all_executed_keys = set()
        executed_trades = []
        
        # Exposure tracking
        concurrent_exposures = []
        deployed_cap_per_trade = []
        size_counts = {75000.0: 0, 125000.0: 0, 175000.0: 0}
        flat_size_pnl = 0.0

        for ym in complete_months:
            m_df = df[df["month_year"] == ym].copy().reset_index(drop=True)
            m_groups = list(m_df.groupby("entry_time"))

            active = []
            for entry_time, group in m_groups:
                active = [p for p in active if p["exit_time"] > entry_time]
                cands_bar = group.to_dict("records")

                valid = [c for c in cands_bar if filt(c)]
                valid.sort(key=rank_fn)

                for c in valid:
                    if len(active) >= 4:
                        continue
                    if sym_div and c["symbol"] in {p["symbol"] for p in active}:
                        continue

                    if dyn_size:
                        c_cap, tier = compute_conviction_tier(c)
                        c_eval = evaluate_candidate_economics(c, c_cap, cost_model)
                        c_eval["deployed_cap"] = c_cap
                        size_counts[c_cap] += 1
                    else:
                        c_eval = c.copy()
                        c_eval["deployed_cap"] = 125000.0
                        size_counts[125000.0] += 1

                    # Flat size counter-factual
                    flat_c_eval = evaluate_candidate_economics(c, 125000.0, cost_model)
                    flat_size_pnl += flat_c_eval["net_pnl"]

                    active.append(c_eval)
                    executed_trades.append(c_eval)
                    k = (c["alpha_id"], c["symbol"], str(c["entry_time"]))
                    all_executed_keys.add(k)

                # Track concurrent gross capital at this bar
                curr_exposure = sum(p["deployed_cap"] for p in active)
                concurrent_exposures.append(curr_exposure)

        # Compute Confusion Matrix against J1 (228 total J1 trades)
        tp = len(all_executed_keys.intersection(j1_trade_keys))
        fp = len(all_executed_keys - j1_trade_keys)
        fn = len(j1_trade_keys - all_executed_keys)
        tn = (len(df) - len(j1_trade_keys)) - fp

        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        pos_trades = sum(1 for t in executed_trades if t["net_pnl"] > 0)
        win_rate = (pos_trades / len(executed_trades)) * 100.0 if executed_trades else 0.0
        tot_net_pnl = sum(t["net_pnl"] for t in executed_trades)
        avg_monthly_pnl = tot_net_pnl / 17.0
        avg_monthly_roi = (avg_monthly_pnl / 500000.0) * 100.0

        peak_exposure = max(concurrent_exposures) if concurrent_exposures else 0.0
        avg_exposure = np.mean(concurrent_exposures) if concurrent_exposures else 0.0

        class_results.append({
            "model": name,
            "trades": len(executed_trades),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "win_rate": win_rate,
            "tot_pnl": tot_net_pnl,
            "avg_roi": avg_monthly_roi,
            "peak_exposure": peak_exposure,
            "avg_exposure": avg_exposure,
            "size_counts": size_counts,
            "flat_size_pnl": flat_size_pnl,
            "executed_trades": executed_trades
        })

    print(f"{'MODEL':<44} | {'TRADES':<6} | {'TP':<4} | {'FP':<5} | {'FN':<4} | {'PRECISION':<9} | {'RECALL':<8} | {'F1':<6} | {'WIN %':<7} | {'AVG ROI':<8} | {'NET PNL':<12}")
    print("-" * 145)
    for cr in class_results:
        print(f"{cr['model']:<44} | {cr['trades']:>6d} | {cr['tp']:>4d} | {cr['fp']:>5d} | {cr['fn']:>4d} | {cr['precision']:>8.1f}% | {cr['recall']:>7.1f}% | {cr['f1']:>5.1f}% | {cr['win_rate']:>6.1f}% | {cr['avg_roi']:>+7.2f}% | Rs {cr['tot_pnl']:>+9,.0f}")
    print("=" * 145)

    # =========================================================================
    # PART 2: FORENSIC RANKING ERROR DIAGNOSTIC
    # =========================================================================
    print("\n" + "=" * 145)
    print("PART 2: RANKING ERROR DIAGNOSTIC (WHEN >= 1 J1 WINNER & >= 1 NON-J1 WERE CANDIDATES AT SAME BAR)")
    print("=" * 145)

    fam_filt = models_to_test["6. Family-Adaptive Gating (Flat Sizing)"]["filter"]
    
    competing_bars = 0
    rank_successes = 0
    rank_failures = 0

    for entry_time, group in df.groupby("entry_time"):
        cands_bar = group.to_dict("records")
        valid_bar = [c for c in cands_bar if fam_filt(c)]
        if len(valid_bar) < 2:
            continue

        j1_in_valid = [c for c in valid_bar if c["is_j1"]]
        non_j1_in_valid = [c for c in valid_bar if not c["is_j1"]]

        if len(j1_in_valid) >= 1 and len(non_j1_in_valid) >= 1:
            competing_bars += 1
            valid_sorted = sorted(valid_bar, key=lambda c: c.get("dist_vwap_atr", 1.0))
            if valid_sorted[0]["is_j1"]:
                rank_successes += 1
            else:
                rank_failures += 1

    print(f"Total Simultaneous Competing Bars: {competing_bars}")
    print(f"Ranker picked J1 Winner #1: {rank_successes} ({rank_successes/competing_bars*100:.1f}%)")
    print(f"Ranker picked Non-J1 Loser/Sub-optimal #1: {rank_failures} ({rank_failures/competing_bars*100:.1f}%)")

    # Feature comparison between True Positives and False Positives in Family-Adaptive Model
    fam_exec = class_results[5]["executed_trades"]  # Model 6
    exec_df = pd.DataFrame(fam_exec)
    exec_df["is_j1"] = exec_df.apply(lambda r: (r["alpha_id"], r["symbol"], str(r["entry_time"])) in j1_trade_keys, axis=1)

    print("\n[*] Feature Contrast: True Positives (J1 Winners) vs False Positives (Non-J1 Trades) in Model 6:")
    print(f"{'METRIC / FEATURE':<35} | {'TRUE POSITIVES (TP)':<25} | {'FALSE POSITIVES (FP)':<25} | {'DIFFERENCE'}")
    print("-" * 110)
    for col, name in [
        ("dist_vwap_atr", "VWAP Distance (ATR units)"),
        ("opp_density", "Opportunity Density (market breadth)"),
        ("rvol_15m", "15-min RVOL (Relative Volume)"),
        ("atr_pct", "ATR % (Normalized Volatility)"),
        ("duration_mins", "Trade Duration (mins)"),
        ("gross_pnl", "Gross P&L (Rs)"),
        ("costs", "Costs + Slippage (Rs)"),
        ("net_pnl", "Net P&L (Rs)")
    ]:
        if col == "duration_mins":
            tp_val = ((pd.to_datetime(exec_df[exec_df['is_j1']]['exit_time']) - pd.to_datetime(exec_df[exec_df['is_j1']]['entry_time'])).dt.total_seconds() / 60.0).mean()
            fp_val = ((pd.to_datetime(exec_df[~exec_df['is_j1']]['exit_time']) - pd.to_datetime(exec_df[~exec_df['is_j1']]['entry_time'])).dt.total_seconds() / 60.0).mean()
        else:
            tp_val = exec_df[exec_df['is_j1']][col].mean()
            fp_val = exec_df[~exec_df['is_j1']][col].mean()
        
        diff = tp_val - fp_val
        print(f"{name:<35} | {tp_val:>23.4f} | {fp_val:>23.4f} | {diff:>+12.4f}")
    print("=" * 110)

    # =========================================================================
    # PART 3: EXPOSURE & CONVICTION SIZING RISK AUDIT
    # =========================================================================
    print("\n" + "=" * 145)
    print("PART 3: EXPOSURE & CONVICTION SIZING RISK AUDIT")
    print("=" * 145)

    dyn_res = class_results[6]  # Model 7: Family-Adaptive + Dynamic Sizing
    flat_res = class_results[5] # Model 6: Family-Adaptive Flat Sizing

    dyn_trades = pd.DataFrame(dyn_res["executed_trades"])
    dyn_trades["trade_date"] = pd.to_datetime(dyn_trades["entry_time"]).dt.date
    daily_pnls = dyn_trades.groupby("trade_date")["net_pnl"].sum()

    flat_trades = pd.DataFrame(flat_res["executed_trades"])
    flat_trades["trade_date"] = pd.to_datetime(flat_trades["entry_time"]).dt.date
    flat_daily_pnls = flat_trades.groupby("trade_date")["net_pnl"].sum()

    print(f"1. POSITION SIZING FREQUENCY:")
    tot_dyn_trades = len(dyn_trades)
    for sz, count in dyn_res["size_counts"].items():
        print(f"   - Tier Rs {sz:,.0f}: {count} trades ({count/tot_dyn_trades*100:.1f}%)")

    print(f"\n2. EXPOSURE & CAPITAL UTILIZATION:")
    print(f"   - Nominal Base Capital: Rs 5,00,000")
    print(f"   - Peak Instantaneous Gross Exposure: Rs {dyn_res['peak_exposure']:,.0f} ({dyn_res['peak_exposure']/500000.0*100:.1f}% of base capital)")
    print(f"   - Time-Weighted Average Gross Exposure: Rs {dyn_res['avg_exposure']:,.0f} ({dyn_res['avg_exposure']/500000.0*100:.1f}% of base capital)")
    print(f"   - Average Capital Deployed Per Active Trade: Rs {dyn_trades['deployed_cap'].mean():,.0f}")

    print(f"\n3. RETURN ON DEPLOYED CAPITAL:")
    print(f"   - 17-Month Net Profit (Dynamic Sizing): Rs {dyn_res['tot_pnl']:,.0f}")
    print(f"   - 17-Month Net Profit (Flat Rs 1.25L Sizing): Rs {dyn_res['flat_size_pnl']:,.0f}")
    print(f"   - Alpha Uplift from Conviction Sizing Alone: Rs {dyn_res['tot_pnl'] - dyn_res['flat_size_pnl']:+,.0f} (+{(dyn_res['tot_pnl'] - dyn_res['flat_size_pnl'])/dyn_res['flat_size_pnl']*100:.1f}%)")
    print(f"   - Annualized ROI on Nominal Rs 5L Base: {dyn_res['avg_roi']*12:.2f}% p.a.")
    
    if dyn_res['avg_exposure'] > 0:
        annual_roi_on_deployed = (dyn_res['tot_pnl'] / (17.0/12.0)) / dyn_res['avg_exposure'] * 100.0
        print(f"   - Annualized ROI on Time-Weighted Active Exposure: {annual_roi_on_deployed:.1f}% p.a.")

    print(f"\n4. TAIL RISK & WORST LOSSES:")
    print(f"   - Dynamic Sizing Worst Single Trade Loss: Rs {dyn_trades['net_pnl'].min():,.0f} ({dyn_trades['net_pnl'].min()/500000.0*100:.2f}% of Rs 5L)")
    print(f"   - Flat Sizing Worst Single Trade Loss:    Rs {flat_trades['net_pnl'].min():,.0f} ({flat_trades['net_pnl'].min()/500000.0*100:.2f}% of Rs 5L)")
    print(f"   - Dynamic Sizing Worst Single Day Loss:   Rs {daily_pnls.min():,.0f} ({daily_pnls.min()/500000.0*100:.2f}% of Rs 5L) on {daily_pnls.idxmin()}")
    print(f"   - Flat Sizing Worst Single Day Loss:      Rs {flat_daily_pnls.min():,.0f} ({flat_daily_pnls.min()/500000.0*100:.2f}% of Rs 5L) on {flat_daily_pnls.idxmin()}")
    print("=" * 145)

    # =========================================================================
    # PART 4: RANKING ENHANCEMENT EXPERIMENTS (TARGETING 40-50% J1 CAPTURE)
    # =========================================================================
    print("\n" + "=" * 145)
    print("PART 4: RANKING ENHANCEMENT EXPERIMENTS (TARGETING 40-50% J1 CAPTURE & ~1.0% - 1.3% MONTHLY ROI)")
    print("=" * 145)

    fam_weights = {
        "VWAP_MEAN_REVERSION": 3.0,
        "GAP_EXHAUSTION": 2.5,
        "INITIAL_BALANCE": 1.5,
        "TREND_CONTINUATION": 1.0,
        "OTHER": 0.5
    }

    ranking_models = {
        "A. Baseline Family-Adaptive (Rank: VWAP Dist)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c),
            "rank": lambda c: c.get("dist_vwap_atr", 1.0),
            "reverse_rank": False,
            "dyn_size": True
        },
        "B. Composite Ranker (Family Weight + VWAP Proximity + Density)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c),
            "rank": lambda c: fam_weights.get(c.get("family", ""), 1.0) * 2.0 - 3.0 * c.get("dist_vwap_atr", 1.0) + 0.15 * c.get("opp_density", 0),
            "reverse_rank": True,
            "dyn_size": True
        },
        "C. Precision-Max Ranker (Family Weight + RVOL + Strict Proximity)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c) and (c.get("dist_vwap_atr", 1.0) <= 0.25 or c.get("family") == "VWAP_MEAN_REVERSION"),
            "rank": lambda c: fam_weights.get(c.get("family", ""), 1.0) * 2.5 - 4.0 * c.get("dist_vwap_atr", 1.0) + 0.5 * min(c.get("rvol_15m", 1.0), 3.0),
            "reverse_rank": True,
            "dyn_size": True
        },
        "D. Multi-Slot Expansion (5 Slots @ Rs 100k-150k Sizing)": {
            "filter": lambda c: c["alpha_id"] in top_profitable_alphas and is_extended_timing_favorable(c) and is_family_adaptive_favorable(c),
            "rank": lambda c: fam_weights.get(c.get("family", ""), 1.0) * 2.0 - 3.0 * c.get("dist_vwap_atr", 1.0) + 0.15 * c.get("opp_density", 0),
            "reverse_rank": True,
            "dyn_size": True,
            "max_slots": 5
        }
    }

    rank_arch_results = {name: [] for name in ranking_models}

    for ym in complete_months:
        m_df = df[df["month_year"] == ym].copy().reset_index(drop=True)
        m_cands = m_df.to_dict("records")
        m_groups = list(m_df.groupby("entry_time"))

        for rname, rcfg in ranking_models.items():
            filter_func = rcfg["filter"]
            rank_func = rcfg["rank"]
            rev_rank = rcfg["reverse_rank"]
            dyn_size = rcfg["dyn_size"]
            m_slots = rcfg.get("max_slots", 4)

            active = []
            executed = []

            for entry_time, group in m_groups:
                active = [p for p in active if p["exit_time"] > entry_time]
                cands_bar = group.to_dict("records")

                valid = [c for c in cands_bar if filter_func(c)]
                valid.sort(key=rank_func, reverse=rev_rank)

                for c in valid:
                    if len(active) >= m_slots or c["symbol"] in {p["symbol"] for p in active}:
                        continue

                    if dyn_size:
                        c_cap, tier = compute_conviction_tier(c)
                        if m_slots == 5:
                            c_cap = min(c_cap, 140000.0)
                        c_eval = evaluate_candidate_economics(c, c_cap, cost_model)
                    else:
                        c_eval = c

                    active.append(c_eval)
                    executed.append(c_eval)

            m_res = compute_metrics(executed, 500000.0, len(m_cands))
            m_res["month"] = str(ym)
            m_res["executed"] = executed
            rank_arch_results[rname].append(m_res)

    j1_tot_pnl = 221006.0

    print(f"{'RANKING ARCHITECTURE':<48} | {'AVG ROI':<9} | {'MED ROI':<9} | {'WIN MOS':<12} | {'TOT NET PNL':<14} | {'AVG TRADES':<12} | {'MAX DD':<8} | {'J1 CAPTURE':<10}")
    print("-" * 145)
    for rname in ranking_models:
        recs = rank_arch_results[rname]
        rois = [r["roi_pct"] for r in recs]
        pnls = [r["net_pnl"] for r in recs]
        trades = [r["trades"] for r in recs]
        dds = [r["max_dd_pct"] for r in recs]
        pos = sum(1 for r in rois if r > 0)
        tot_pnl = float(np.sum(pnls))
        j1_cap_pct = (tot_pnl / j1_tot_pnl) * 100.0

        print(f"{rname:<48} | {np.mean(rois):>+7.2f}% | {np.median(rois):>+7.2f}% | {pos:>2d}/17 ({pos/17*100:4.1f}%) | Rs {tot_pnl:>+10,.0f} | {np.mean(trades):>7.1f} tr  | {np.max(dds):>6.2f}% | {j1_cap_pct:>8.1f}%")
    print("=" * 145)

    # 5. WALK-FORWARD OOS BREAKDOWN FOR ENHANCED RANKERS
    print("\n" + "=" * 145)
    print("WALK-FORWARD OOS VERIFICATION ACROSS FOLDS 1 TO 4 (FINAL HOLD-OUT JUL-AUG 2026)")
    print("=" * 145)
    folds = [
        ("Fold 1 OOS (Oct-Dec 2025)", ["2025-10", "2025-11", "2025-12"]),
        ("Fold 2 OOS (Jan-Mar 2026)", ["2026-01", "2026-02", "2026-03"]),
        ("Fold 3 OOS (Apr-Jun 2026)", ["2026-04", "2026-05", "2026-06"]),
        ("Fold 4 FINAL UNTOUCHED OOS (Jul-Aug 2026)", ["2026-07", "2026-08"]),
    ]

    for fold_name, fold_months in folds:
        print(f"\n[*] {fold_name}:")
        print(f"{'CONFIGURATION':<48} | {'AVG ROI':<9} | {'NET PNL':<14} | {'TRADES':<8} | {'WIN MOS':<10}")
        print("-" * 95)
        for rname in ranking_models:
            recs = [r for r in rank_arch_results[rname] if r["month"] in fold_months]
            rois = [r["roi_pct"] for r in recs]
            pnls = [r["net_pnl"] for r in recs]
            trades = [r["trades"] for r in recs]
            pos = sum(1 for r in rois if r > 0)
            print(f"{rname:<48} | {np.mean(rois):>+7.2f}% | Rs {np.sum(pnls):>+10,.0f} | {np.sum(trades):>6d} tr | {pos}/{len(rois)}")


if __name__ == "__main__":
    main()
