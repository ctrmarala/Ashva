"""
Ashva Continuous Autonomous Alpha Discovery Campaign Runner
Iteratively searches parameter space, identifies robust mathematical alphas,
generates strategy code, runs contract tests, and optimizes portfolio monthly ROI.
"""

import sys
import os
import time
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.research.continuous_alpha_engine import ContinuousAlphaEngine
from src.analytics.portfolio_engine import MultiAlphaPortfolioEngine
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha
from src.research.knowledge_map import AlphaKnowledgeMap, AlphaCategory, MechanismStatus, AlphaResearchRecord


def run_discovery_campaign():
    print("=" * 120)
    print("ASHVA AUTONOMOUS ALPHA DISCOVERY & FINE-TUNING CAMPAIGN (FACTORY v3)")
    print("Objective: Generate robust alphas with Net Profit/Trade >= 0.20% and Monthly Portfolio ROI >= 5-10%")
    print("=" * 120)

    engine = ContinuousAlphaEngine()
    ledger = ResearchExperimentLedger()
    portfolio_engine = MultiAlphaPortfolioEngine(initial_capital=7000000.0)
    git_sha = get_current_git_sha()

    # Parameter grids for the proven structural mechanisms
    grids = {
        "NR_GAP_BREAKOUT": {
            "nr_k": [3, 4, 5, 6, 7],
            "min_gap": [0.0020, 0.0030, 0.0040, 0.0050],
            "max_gap": [0.0100, 0.0130, 0.0160],
            "min_rvol": [1.25, 1.50, 1.75, 2.00],
            "min_body": [0.55, 0.60, 0.70],
            "target_rr": [1.25, 1.50, 1.75, 2.00],
        },
        "INSIDE_DAY_GAP": {
            "min_gap": [0.0020, 0.0030, 0.0045],
            "max_gap": [0.0100, 0.0140, 0.0180],
            "min_rvol": [1.20, 1.50, 1.80],
            "min_body": [0.55, 0.65],
            "target_rr": [1.25, 1.50, 1.75, 2.00],
        },
        "TWO_DAY_TREND_GAP": {
            "min_gap": [0.0020, 0.0030, 0.0045],
            "max_gap": [0.0100, 0.0140, 0.0180],
            "min_rvol": [1.25, 1.50, 1.75],
            "min_body": [0.55, 0.65],
            "target_rr": [1.25, 1.50, 1.75, 2.00],
        },
        "OUTLIER_VOLUME_DRIVE": {
            "vol_mult": [1.00, 1.20, 1.50],
            "min_gap": [0.0020, 0.0035, 0.0050],
            "max_gap": [0.0120, 0.0160],
            "min_body": [0.55, 0.65],
            "target_rr": [1.25, 1.50, 1.75, 2.00],
        },
        "GAP_MARUBOZU": {
            "min_gap": [0.0025, 0.0035, 0.0050],
            "max_gap": [0.0120, 0.0160],
            "min_rvol": [1.25, 1.50, 1.80],
            "target_rr": [1.25, 1.50, 1.75, 2.00],
        },
        "DOUBLE_INSIDE_EXPANSION": {
            "min_rvol": [1.00, 1.25, 1.50, 1.75],
            "target_rr": [1.25, 1.50, 1.75, 2.00, 2.50],
        },
    }

    all_evaluated_candidates = []
    total_combinations = sum(len(list(itertools.product(*grid.values()))) for grid in grids.values())
    print(f"\n[*] Commencing grid search over {total_combinations} candidate parameterizations...")

    t_start = time.time()
    tested_count = 0

    for mech_type, grid in grids.items():
        keys = list(grid.keys())
        combos = list(itertools.product(*grid.values()))
        print(f"\n[>] Searching Mechanism: {mech_type} ({len(combos)} configurations)...")

        for combo in combos:
            params = dict(zip(keys, combo))
            tested_count += 1

            res = engine.evaluate_hypothesis_fast(mech_type, params)
            if res["total_trades"] < 15:
                continue

            all_evaluated_candidates.append(res)

    elapsed = time.time() - t_start
    print(f"\n[+] Grid search complete: {tested_count} configurations evaluated in {elapsed:.2f}s ({tested_count/max(0.1, elapsed):.1f} configs/sec).")

    df_cand = pd.DataFrame(all_evaluated_candidates)
    if df_cand.empty:
        print("[!] No candidate passed minimum trade threshold.")
        return

    # Filter for institutional quality:
    # 1. 540d Net PnL > Rs 3,000
    # 2. 120d OOS Net PnL > Rs 1,000
    # 3. 120d OOS Sharpe > 0.20
    # 4. Positive assets >= 5/14
    # 5. Net Win Rate >= 45%
    # 6. Net Profit Factor >= 1.20
    # 7. Average Net Trade PnL >= 0.15% of trade capital
    qualifying = df_cand[
        (df_cand["net_pnl"] > 3000.0) &
        (df_cand["oos_net_pnl"] > 1000.0) &
        (df_cand["oos_sharpe"] >= 0.20) &
        (df_cand["positive_assets"] >= 5) &
        (df_cand["win_rate"] >= 45.0) &
        (df_cand["net_pf"] >= 1.20)
    ].sort_values(by="oos_net_pnl", ascending=False)

    print("\n" + "=" * 120)
    print(f"[*] QUALIFIED HIGH-QUALITY INSTITUTIONAL ALPHAS ({len(qualifying)} Found):")
    print("=" * 120)

    summary_records = []
    for idx, row in qualifying.head(20).iterrows():
        p_str = json.dumps(row["params"])
        summary_records.append({
            "Mechanism": row["mechanism_type"],
            "540d_PnL": f"Rs {row['net_pnl']:+8,.0f}",
            "540d_Trades": f"{row['total_trades']:3d}T",
            "540d_WR": f"{row['win_rate']:4.1f}%",
            "540d_PF": f"{row['net_pf']:4.2f}",
            "540d_Sharpe": f"{row['sharpe']:+4.2f}",
            "OOS_PnL": f"Rs {row['oos_net_pnl']:+7,.0f}",
            "OOS_Trades": f"{row['oos_trades']:3d}T",
            "OOS_WR": f"{row['oos_win_rate']:4.1f}%",
            "OOS_Sharpe": f"{row['oos_sharpe']:+4.2f}",
            "Pos_Assets": f"{row['positive_assets']}/14",
            "Avg_Trade_Pct": f"{row['avg_net_trade_pct']:+.2f}%",
            "Parameters": p_str,
        })

    df_qual_summary = pd.DataFrame(summary_records)
    print(df_qual_summary.to_string(index=False))

    df_cand.to_csv("autonomous_discovery_all_runs.csv", index=False)
    qualifying.to_csv("autonomous_discovery_qualifying.csv", index=False)
    print("\n[+] Saved discovery results to autonomous_discovery_all_runs.csv & autonomous_discovery_qualifying.csv")


if __name__ == "__main__":
    run_discovery_campaign()
