"""
Ashva PCDE: Exploratory Feature Audit (Ex-Ante Entry-Time Characteristics)
========================================================================

Evaluates zero-lookahead feature domains across all 7,070 candidates in the
17 complete evaluation months (April 2025 to August 2026).
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from scripts.research_pcde_phase2_3 import (
    load_candidates, evaluate_candidate_economics, solve_milp_clairvoyant
)


def compute_technical_features(lake: DataLake, symbols: List[str]) -> pd.DataFrame:
    """Precompute bar-level technical context across all 77 symbols (540 days)."""
    print("[*] Precomputing bar-level technical context (ATR, VWAP, RVOL)...", flush=True)
    dfs = []
    for sym in symbols:
        df = lake.load_bars(sym, "15m", max_lookback_days=540)
        if df.empty or len(df) < 50:
            continue
        df = df.copy()
        df["symbol"] = sym
        df["timestamp"] = pd.to_datetime(df.index)
        df["date"] = df["timestamp"].dt.date
        df["time_slot"] = df["timestamp"].dt.strftime("%H:%M")

        # True Range & 14-bar ATR
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        df["atr14"] = pd.Series(tr, index=df.index).rolling(14, min_periods=5).mean()
        df["atr_pct"] = df["atr14"] / df["close"]

        # Intraday VWAP
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()
        df["cum_vp"] = df.groupby("date").apply(lambda g: (g["close"] * g["volume"]).cumsum()).reset_index(level=0, drop=True)
        df["vwap"] = df["cum_vp"] / np.maximum(1.0, df["cum_vol"])
        df["dist_vwap_atr"] = (df["close"] - df["vwap"]) / np.maximum(0.1, df["atr14"])

        # RVOL vs 20-Day Median for the same time bucket
        slot_medians = df.groupby("time_slot")["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).median())
        df["rvol"] = df["volume"] / np.maximum(1.0, slot_medians.fillna(df["volume"].median()))

        dfs.append(df[["timestamp", "symbol", "close", "atr14", "atr_pct", "vwap", "dist_vwap_atr", "rvol"]])

    all_tech = pd.concat(dfs, ignore_index=True)
    return all_tech


def main():
    lake = DataLake(read_only=True)
    cands = load_candidates()
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    slot_cap = 125000.0

    eval_cands = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in cands if c["entry_price"] <= slot_cap]
    df = pd.DataFrame(eval_cands).sort_values("entry_time").reset_index(drop=True)
    df["month_year"] = df["entry_time"].dt.to_period("M")

    # Complete 17 months filter (2025-04 to 2026-08)
    complete_months = [m for m in sorted(df["month_year"].unique()) if m not in [pd.Period("2025-03", "M"), pd.Period("2026-09", "M")]]
    df = df[df["month_year"].isin(complete_months)].copy().reset_index(drop=True)
    print(f"[*] Total Candidates in 17 Complete Months: {len(df):,}")

    # 1. Identify J1 trades for 17 months
    j1_trade_keys = set()
    for ym in complete_months:
        c_sub = df[df["month_year"] == ym].to_dict("records")
        j1_trades = solve_milp_clairvoyant(c_sub, max_slots=4, enforce_symbol_diversity=True)
        for t in j1_trades:
            j1_trade_keys.add((t["alpha_id"], t["symbol"], str(t["entry_time"])))

    df["is_j1"] = df.apply(lambda r: (r["alpha_id"], r["symbol"], str(r["entry_time"])) in j1_trade_keys, axis=1)
    df["is_win"] = df["net_pnl"] > 0.0

    # 2. Time-of-Day P&L Contribution Analysis
    df["time_slot"] = df["entry_time"].dt.strftime("%H:%M")
    print("\n" + "=" * 120)
    print("SECTION 1: TIME-OF-DAY INTRADAY ENTRY BUCKET DECOMPOSITION")
    print("=" * 120)
    tod_summary = df.groupby("time_slot").agg(
        total_cands=("alpha_id", "count"),
        j1_trades=("is_j1", "sum"),
        j1_selection_pct=("is_j1", lambda x: (x.sum() / df["is_j1"].sum()) * 100),
        win_rate=("is_win", lambda x: x.mean() * 100),
        avg_gross_pnl=("gross_pnl", "mean"),
        avg_costs=("costs", "mean"),
        avg_net_pnl=("net_pnl", "mean"),
        total_net_pnl=("net_pnl", "sum"),
        j1_total_net_pnl=("net_pnl", lambda x: df.loc[x.index[df.loc[x.index, "is_j1"]], "net_pnl"].sum())
    )
    tod_summary["j1_pnl_share_%"] = (tod_summary["j1_total_net_pnl"] / df[df["is_j1"]]["net_pnl"].sum()) * 100
    print(tod_summary.to_string())

    # 3. Technical & Market Context Features Precomputation
    symbols = sorted(df["symbol"].unique())
    tech_df = compute_technical_features(lake, symbols)

    # Merge technical features on (symbol, entry_time)
    df = pd.merge(df, tech_df, left_on=["symbol", "entry_time"], right_on=["symbol", "timestamp"], how="left")
    df["atr14"] = df["atr14"].fillna(df["entry_price"] * 0.015)
    df["atr_pct"] = df["atr_pct"].fillna(0.015)
    df["rvol"] = df["rvol"].fillna(1.0)
    df["dist_vwap_atr"] = df["dist_vwap_atr"].fillna(0.0)

    # F2: Economic Payoff Buffer (ATR / Roundtrip Statutory Friction in Rs)
    df["f_econ_buffer"] = (df["atr14"] * (slot_cap / df["entry_price"])) / np.maximum(20.0, df["costs"])

    # F6: Intraday Timing (Minutes from 09:15)
    df["f_mins_from_open"] = (df["entry_time"].dt.hour * 60 + df["entry_time"].dt.minute) - (9 * 60 + 15)

    # F7: Opportunity Density (Total simultaneous candidates firing across all symbols at entry_time)
    time_density = df.groupby("entry_time")["alpha_id"].transform("count")
    df["f_opp_density"] = time_density

    # F8 & F9: Same-Symbol Agreement & Family Consensus
    sym_fam_counts = df.groupby(["entry_time", "symbol"])["family"].transform("nunique")
    df["f_distinct_families"] = sym_fam_counts

    df["_long_count"] = df.groupby(["entry_time", "symbol"])["side"].transform(lambda s: (s == "BUY").sum())
    df["_short_count"] = df.groupby(["entry_time", "symbol"])["side"].transform(lambda s: (s == "SELL").sum())
    df["f_directional_consensus"] = df.apply(lambda r: r["_long_count"] if r["side"] == "BUY" else r["_short_count"], axis=1)

    # F10: Rolling 30-Day Historical Alpha Win Rate (Point-in-Time)
    print("[*] Computing Point-in-Time 30-Day Rolling Alpha Win Rates...", flush=True)
    rolling_alpha_wr = []
    for idx, row in df.iterrows():
        t_entry = row["entry_time"]
        a_id = row["alpha_id"]
        t_30d_prior = t_entry - pd.Timedelta(days=30)
        hist = df[(df["alpha_id"] == a_id) & (df["exit_time"] < t_entry) & (df["entry_time"] >= t_30d_prior)]
        if len(hist) >= 3:
            rolling_alpha_wr.append(float(hist["is_win"].mean()))
        else:
            rolling_alpha_wr.append(0.50)  # Neutral prior
    df["f_rolling_30d_wr"] = rolling_alpha_wr

    feature_cols = {
        "F1: ATR % (Volatility)": "atr_pct",
        "F2: Economic Payoff Buffer": "f_econ_buffer",
        "F3: Relative Volume (RVOL)": "rvol",
        "F4: Distance to VWAP (/ATR)": "dist_vwap_atr",
        "F6: Minutes From Open (Time)": "f_mins_from_open",
        "F7: Opportunity Density": "f_opp_density",
        "F8: Distinct Family Count": "f_distinct_families",
        "F9: Directional Consensus": "f_directional_consensus",
        "F10: Trailing 30D Alpha WR": "f_rolling_30d_wr"
    }

    # 4. Feature Information Coefficient (IC) & Statistics Scorecard
    print("\n" + "=" * 140)
    print("SECTION 2: EX-ANTE FEATURE STATISTICAL AUDIT & INFORMATION COEFFICIENT (17 MONTHS, 7,070 CANDIDATES)")
    print("=" * 140)
    print(f"{'FEATURE NAME':<30} | {'OVERALL IC':<11} | {'t-stat':<8} | {'p-val':<8} | {'MONTHLY IC MEAN (STD)':<23} | {'% POS MOS':<10} | {'J1 WINNER MEAN':<15} | {'NON-J1 MEAN':<15}")
    print("-" * 140)

    for f_label, col in feature_cols.items():
        vals = df[col].values
        targets = df["net_pnl"].values

        valid = ~np.isnan(vals) & ~np.isnan(targets)
        v_vals = vals[valid]
        v_targ = targets[valid]

        ic, p_val = stats.spearmanr(v_vals, v_targ)
        n = len(v_vals)
        t_stat = ic * np.sqrt((n - 2) / max(1e-6, 1.0 - ic ** 2))

        # Monthly IC stability
        m_ics = []
        for ym in complete_months:
            m_sub = df[df["month_year"] == ym]
            if len(m_sub) > 10:
                m_ic, _ = stats.spearmanr(m_sub[col], m_sub["net_pnl"])
                if not np.isnan(m_ic):
                    m_ics.append(m_ic)
        m_ic_mean = np.mean(m_ics) if m_ics else 0.0
        m_ic_std = np.std(m_ics) if m_ics else 0.0
        pos_m_pct = (sum(1 for x in m_ics if x > 0) / len(m_ics) * 100) if m_ics else 0.0

        j1_mean = df[df["is_j1"]][col].mean()
        non_j1_mean = df[~df["is_j1"]][col].mean()

        print(f"{f_label:<30} | {ic:>+9.4f}  | {t_stat:>+7.2f} | {p_val:>7.4f} | {m_ic_mean:>+6.3f} (±{m_ic_std:>5.3f})        | {pos_m_pct:>6.1f}%    | {j1_mean:>13.3f}   | {non_j1_mean:>13.3f}")

    print("=" * 140)

    # 5. Top vs Bottom Quantile Net P&L Trajectory
    print("\n" + "=" * 120)
    print("SECTION 3: QUINTILE ANALYSIS (Q1 LOWEST TO Q5 HIGHEST) - NET P&L & WIN RATE")
    print("=" * 120)
    for f_label, col in feature_cols.items():
        try:
            df["q"] = pd.qcut(df[col].rank(method="first"), 5, labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"])
            q_res = df.groupby("q", observed=False).agg(
                cands=("alpha_id", "count"),
                win_rate=("is_win", lambda x: x.mean() * 100),
                avg_net_pnl=("net_pnl", "mean"),
                j1_rate=("is_j1", lambda x: x.mean() * 100),
            )
            print(f"\n[*] Feature: {f_label}")
            print(q_res.to_string())
        except Exception as e:
            print(f"[!] Could not compute 5-quantiles for {f_label}: {e}")

    # 6. Feature Correlation Matrix
    print("\n" + "=" * 120)
    print("SECTION 4: FEATURE CORRELATION MATRIX (SPEARMAN RANK)")
    print("=" * 120)
    corr_df = df[list(feature_cols.values())].rename(columns={v: k for k, v in feature_cols.items()}).corr(method="spearman")
    print(corr_df.round(2).to_string())


if __name__ == "__main__":
    main()
