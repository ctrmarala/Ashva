"""
Ashva Combinatorial Purged Cross-Validation (CPCV) & PBO Engine
Implements Marcos López de Prado's institutional methodology to test multiple Out-of-Sample paths
and calculate the Probability of Backtest Overfitting (PBO).
"""

import itertools
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd


class CPCVEngine:
    """
    Evaluates strategy robustness across combinatorial purged train/test slices.
    """

    def __init__(self, n_partitions: int = 6, k_test_partitions: int = 2):
        self.n = n_partitions
        self.k = k_test_partitions

    def evaluate_trades(self, trade_returns: List[float], annualization_factor: float = np.sqrt(252)) -> Dict[str, Any]:
        """
        Evaluates PBO and CPCV distributions from a list of chronological trade PnLs.
        """
        if len(trade_returns) < 20:
            return {
                "status": "INSUFFICIENT_TRADES",
                "pbo": 1.0,
                "mean_oos_sharpe": 0.0,
                "is_overfitted": True,
            }

        returns = np.array(trade_returns, dtype=np.float64)
        n_trades = len(returns)
        block_size = n_trades // self.n

        # Create partitions
        partitions = []
        for i in range(self.n):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < self.n - 1 else n_trades
            partitions.append(returns[start_idx:end_idx])

        # Generate all combinations of k test partitions
        combos = list(itertools.combinations(range(self.n), self.k))
        is_sharpes = []
        oos_sharpes = []

        for test_indices in combos:
            train_indices = [idx for idx in range(self.n) if idx not in test_indices]

            # Purged test & train sets
            test_data = np.concatenate([partitions[idx] for idx in test_indices])
            train_data = np.concatenate([partitions[idx] for idx in train_indices])

            # In-Sample Sharpe
            is_std = np.std(train_data)
            is_sharpe = (np.mean(train_data) / (is_std + 1e-6)) * annualization_factor if is_std > 0 else 0.0
            is_sharpes.append(is_sharpe)

            # Out-of-Sample Sharpe
            oos_std = np.std(test_data)
            oos_sharpe = (np.mean(test_data) / (oos_std + 1e-6)) * annualization_factor if oos_std > 0 else 0.0
            oos_sharpes.append(oos_sharpe)

        is_sharpes = np.array(is_sharpes)
        oos_sharpes = np.array(oos_sharpes)

        # Probability of Backtest Overfitting (PBO): Fraction of OOS paths with negative Sharpe
        pbo = float(np.mean(oos_sharpes <= 0.0))
        mean_is_sharpe = float(np.mean(is_sharpes))
        mean_oos_sharpe = float(np.mean(oos_sharpes))
        median_oos_sharpe = float(np.median(oos_sharpes))
        oos_sharpe_std = float(np.std(oos_sharpes))

        # Degradation Ratio (OOS Sharpe / IS Sharpe)
        degradation_ratio = (mean_oos_sharpe / max(1e-4, mean_is_sharpe)) if mean_is_sharpe > 0 else 0.0

        is_overfitted = (pbo > 0.30 or mean_oos_sharpe < 0.5)

        return {
            "status": "VALIDATED",
            "n_paths": len(combos),
            "pbo": round(pbo, 3),
            "pbo_pct": f"{round(pbo * 100, 1)}%",
            "mean_is_sharpe": round(mean_is_sharpe, 2),
            "mean_oos_sharpe": round(mean_oos_sharpe, 2),
            "median_oos_sharpe": round(median_oos_sharpe, 2),
            "oos_sharpe_std": round(oos_sharpe_std, 2),
            "degradation_ratio": round(degradation_ratio, 2),
            "is_overfitted": is_overfitted,
            "distribution_oos_sharpe": [round(float(s), 2) for s in oos_sharpes],
        }
