"""
Ashva Combinatorial Purged Cross-Validation (CPCV) Engine
Strictly implements Marcos López de Prado's Purging and Embargoing methodology
for financial time series with overlapping holding periods.
"""

from itertools import combinations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


class CPCVEngine:
    """
    CPCV Engine with temporal purging and post-test embargoing to eliminate
    serial correlation and label overlap leakage.
    """

    def __init__(
        self,
        n_partitions: int = 6,
        k_test_partitions: int = 2,
        embargo_pct: float = 0.01,  # 1% post-test embargo window
    ):
        self.n_partitions = n_partitions
        self.k_test_partitions = k_test_partitions
        self.embargo_pct = embargo_pct

    def evaluate_trades(
        self,
        trades_df: pd.DataFrame,  # Must contain 'entry_time', 'exit_time', 'net_pnl'
    ) -> Dict[str, Any]:
        """
        Executes CPCV with rigorous Purging & Embargoing across all C(N, k) paths.
        """
        if trades_df.empty or len(trades_df) < (self.n_partitions * 3):
            return {
                "total_trades": len(trades_df),
                "is_overfitted": True,
                "pbo": 1.0,
                "pbo_pct": "100.0%",
                "mean_oos_sharpe": 0.0,
                "median_oos_sharpe": 0.0,
                "degradation_ratio": 0.0,
            }

        df = trades_df.sort_values("entry_time").reset_index(drop=True)
        n_trades = len(df)
        embargo_bars = max(1, int(n_trades * self.embargo_pct))

        # Partition index boundaries
        partition_indices = np.array_split(np.arange(n_trades), self.n_partitions)
        all_combos = list(combinations(range(self.n_partitions), self.k_test_partitions))

        is_sharpes = []
        oos_sharpes = []

        for test_parts in all_combos:
            train_parts = [i for i in range(self.n_partitions) if i not in test_parts]

            # Collect test indices, test time intervals, and embargo index sets
            test_idx = np.concatenate([partition_indices[i] for i in test_parts])
            test_trades = df.iloc[test_idx]

            test_intervals = []
            embargo_indices = set()
            for i in test_parts:
                part_indices = partition_indices[i]
                part_df = df.iloc[part_indices]
                test_intervals.append((part_df["entry_time"].min(), part_df["exit_time"].max()))
                
                # Embargo window: the next embargo_bars trades following the end of this test partition
                max_test_idx = part_indices[-1]
                for emb_i in range(max_test_idx + 1, min(n_trades, max_test_idx + 1 + embargo_bars)):
                    embargo_indices.add(emb_i)

            # -------------------------------------------------------------
            # PURGING & EMBARGOING TRAINING SET
            # -------------------------------------------------------------
            raw_train_idx = np.concatenate([partition_indices[i] for i in train_parts])
            train_candidates = df.iloc[raw_train_idx]

            purged_train_indices = []
            for t_idx, tr_row in train_candidates.iterrows():
                # 1. Embargo Check: Drop if index is within post-test embargo window
                if t_idx in embargo_indices:
                    continue

                # 2. Purge Check: Drop if training holding period overlaps with ANY test interval
                tr_entry = tr_row["entry_time"]
                tr_exit = tr_row["exit_time"]
                overlaps = False
                for test_start, test_end in test_intervals:
                    if not (tr_exit < test_start or tr_entry > test_end):
                        overlaps = True
                        break
                if overlaps:
                    continue

                purged_train_indices.append(t_idx)

            if len(purged_train_indices) < 10 or len(test_trades) < 5:
                continue

            purged_train_trades = df.loc[purged_train_indices]

            # Compute Annualized Sharpe for IS (purged) and OOS
            is_pnl = purged_train_trades["net_pnl"]
            oos_pnl = test_trades["net_pnl"]

            is_s = float((is_pnl.mean() / (is_pnl.std() + 1e-6)) * np.sqrt(252))
            oos_s = float((oos_pnl.mean() / (oos_pnl.std() + 1e-6)) * np.sqrt(252))

            is_sharpes.append(is_s)
            oos_sharpes.append(oos_s)

        if not is_sharpes:
            return {"is_overfitted": True, "pbo": 1.0, "pbo_pct": "100.0%"}

        is_arr = np.array(is_sharpes)
        oos_arr = np.array(oos_sharpes)

        # Probability of Backtest Overfitting (PBO): P(OOS Sharpe <= 0)
        pbo = float(np.mean(oos_arr <= 0.0))
        mean_is = float(np.mean(is_arr))
        mean_oos = float(np.mean(oos_arr))
        degradation = float(mean_oos / (mean_is + 1e-6)) if mean_is > 0 else 0.0

        return {
            "total_trades": n_trades,
            "combinatorial_paths": len(all_combos),
            "mean_in_sample_sharpe": round(mean_is, 2),
            "mean_oos_sharpe": round(mean_oos, 2),
            "median_oos_sharpe": round(float(np.median(oos_arr)), 2),
            "pbo": round(pbo, 3),
            "pbo_pct": f"{round(pbo * 100.0, 1)}%",
            "degradation_ratio": round(degradation, 2),
            "is_overfitted": (pbo > 0.30 or mean_oos < 0.50),
        }
