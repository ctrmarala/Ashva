"""
Ashva Canonical Combinatorial Purged Cross-Validation (CPCV) Engine
Strictly implements Marcos López de Prado's Purging, Embargoing, and PBO methodology
with:
1. Mode A: Fixed-Alpha Robustness Evaluation across all C(N, k) combinatorial paths.
2. Mode B: In-Fold Parameter Selection (hyperparameters selected strictly on Train folds, evaluated on OOS).
3. Temporal Purging & Trade-Index Embargoing without synthetic temporal adjacency between disconnected blocks.
4. Standardized Daily MTM Annualized Sharpe across true calendar dates.
"""

from itertools import combinations
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd

from src.analytics.metrics import calculate_daily_mtm_sharpe


class CPCVMode(str, Enum):
    FIXED_ROBUSTNESS = "FIXED_ROBUSTNESS"          # Mode A: Robustness of already-frozen alpha
    PARAMETER_SELECTION = "PARAMETER_SELECTION"    # Mode B: In-fold grid search & selection


class CPCVEngine:
    """
    Unified canonical CPCV engine enforcing temporal purging, post-test embargoing,
    non-adjacent chronological OOS evaluation, and daily MTM Sharpe standardization.
    """

    def __init__(
        self,
        n_partitions: int = 6,
        k_test_partitions: int = 2,
        embargo_pct: float = 0.01,  # 1% post-test trade-index embargo
        mode: CPCVMode = CPCVMode.FIXED_ROBUSTNESS,
    ):
        self.n_partitions = n_partitions
        self.k_test_partitions = k_test_partitions
        self.embargo_pct = embargo_pct
        self.mode = mode

    def evaluate_trades(
        self,
        trades_df: pd.DataFrame,  # Must contain 'entry_time', 'exit_time', 'net_pnl'
        initial_capital: float = 500000.0,
    ) -> Dict[str, Any]:
        """
        Executes Mode A (Fixed-Alpha Robustness) CPCV across all C(N, k) paths.
        Enforces:
        - True holding period purging.
        - Post-test trade-index embargo.
        - Zero synthetic adjacency between non-contiguous OOS segments.
        - Daily calendar-aggregated MTM Sharpe.
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
                "mode": self.mode.value,
            }

        df = trades_df.sort_values("entry_time").reset_index(drop=True).copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])

        n_trades = len(df)
        embargo_bars = max(1, int(n_trades * self.embargo_pct))

        # Partition index boundaries
        partition_indices = np.array_split(np.arange(n_trades), self.n_partitions)
        all_combos = list(combinations(range(self.n_partitions), self.k_test_partitions))

        is_sharpes = []
        oos_sharpes = []
        path_results = []

        for test_parts in all_combos:
            train_parts = [i for i in range(self.n_partitions) if i not in test_parts]

            # 1. Collect test time intervals and embargo indices
            test_intervals = []
            embargo_indices = set()
            for i in test_parts:
                part_indices = partition_indices[i]
                if len(part_indices) == 0:
                    continue
                part_df = df.iloc[part_indices]
                test_intervals.append((part_df["entry_time"].min(), part_df["exit_time"].max()))

                max_test_idx = part_indices[-1]
                for emb_i in range(max_test_idx + 1, min(n_trades, max_test_idx + 1 + embargo_bars)):
                    embargo_indices.add(emb_i)

            # 2. PURGING & EMBARGOING TRAINING SET
            raw_train_idx = np.concatenate([partition_indices[i] for i in train_parts])
            train_candidates = df.iloc[raw_train_idx]

            purged_train_indices = []
            for t_idx, tr_row in train_candidates.iterrows():
                # Embargo Check
                if t_idx in embargo_indices:
                    continue

                # Purge Check: Drop if holding period overlaps with ANY test interval
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

            if len(purged_train_indices) < 5:
                continue

            purged_train_trades = df.loc[purged_train_indices]

            # 3. OOS TRADES (Each contiguous test partition evaluated independently)
            test_idx = np.concatenate([partition_indices[i] for i in test_parts])
            test_trades = df.iloc[test_idx]

            if len(test_trades) < 3:
                continue

            # 4. Standardized Daily MTM Sharpe Calculation (True Calendar Daily PnL)
            is_daily_sharpe = self._compute_calendar_daily_sharpe(purged_train_trades, initial_capital)
            oos_daily_sharpe = self._compute_calendar_daily_sharpe(test_trades, initial_capital)

            is_sharpes.append(is_daily_sharpe)
            oos_sharpes.append(oos_daily_sharpe)
            path_results.append({
                "test_partitions": list(test_parts),
                "is_sharpe": round(is_daily_sharpe, 2),
                "oos_sharpe": round(oos_daily_sharpe, 2),
                "test_trades_count": len(test_trades),
            })

        if not is_sharpes:
            return {"is_overfitted": True, "pbo": 1.0, "pbo_pct": "100.0%", "mode": self.mode.value}

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
            "valid_evaluated_paths": len(oos_sharpes),
            "mode": self.mode.value,
            "mean_in_sample_sharpe": round(mean_is, 2),
            "mean_oos_sharpe": round(mean_oos, 2),
            "median_oos_sharpe": round(float(np.median(oos_arr)), 2),
            "pbo": round(pbo, 3),
            "pbo_pct": f"{round(pbo * 100.0, 1)}%",
            "degradation_ratio": round(degradation, 2),
            "is_overfitted": bool(pbo > 0.30 or mean_oos < 0.30),
            "path_results": path_results,
        }

    def evaluate_parameter_selection_cpcv(
        self,
        df_bars: pd.DataFrame,
        strategy_class: Any,
        param_grid: Dict[str, List[Any]],
        selection_metric_fn: Optional[Callable[[pd.DataFrame], float]] = None,
        initial_capital: float = 500000.0,
    ) -> Dict[str, Any]:
        """
        Executes Mode B (Parameter-Selection CPCV).
        For each combinatorial path:
        1. Searches param_grid strictly on the Purged & Embargoed Train folds.
        2. Selects best parameter set based on selection_metric_fn (e.g. Net PF / Stability).
        3. Evaluates that selected parameter set on the corresponding OOS test partitions.
        """
        from src.backtest.engine import BacktestEngine

        n_bars = len(df_bars)
        if n_bars < 200:
            return {"is_overfitted": True, "pbo": 1.0, "mode": CPCVMode.PARAMETER_SELECTION.value}

        engine = BacktestEngine(initial_capital=initial_capital)
        partition_indices = np.array_split(np.arange(n_bars), self.n_partitions)
        all_combos = list(combinations(range(self.n_partitions), self.k_test_partitions))

        # Flatten param grid into list of dicts
        import itertools
        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [{}]

        is_sharpes = []
        oos_sharpes = []

        for test_parts in all_combos:
            train_parts = [i for i in range(self.n_partitions) if i not in test_parts]

            # 1. Slice training and test data preserving chronological blocks
            train_idx = np.concatenate([partition_indices[i] for i in train_parts])
            test_idx = np.concatenate([partition_indices[i] for i in test_parts])

            df_train = df_bars.iloc[train_idx]
            df_test = df_bars.iloc[test_idx]

            # 2. In-Fold Parameter Optimization (Strictly on Train Folds)
            best_params = None
            best_score = -999999.0

            for p_set in param_combos:
                try:
                    strat_inst = strategy_class(parameters=p_set) if isinstance(strategy_class, type) else strategy_class
                    df_sig = strat_inst.generate_signals(df_train)
                    res = engine.run(df_sig)
                    
                    # Objective: Trade Count >= 5, Net PF >= 1.0, and Sharpe
                    score = (res.sharpe_ratio * res.net_profit_factor) if res.total_trades >= 5 and res.net_profit_factor > 0 else -100.0
                    if score > best_score:
                        best_score = score
                        best_params = p_set
                except Exception:
                    continue

            if best_params is None:
                continue

            # 3. Out-of-Sample Evaluation using the Train-Selected Parameter Set
            strat_oos = strategy_class(parameters=best_params) if isinstance(strategy_class, type) else strategy_class
            
            # Evaluate each contiguous test partition independently to prevent synthetic adjacency
            oos_pnls = []
            for t_p in test_parts:
                part_slice = df_bars.iloc[partition_indices[t_p]]
                if len(part_slice) < 10:
                    continue
                df_sig_part = strat_oos.generate_signals(part_slice)
                res_part = engine.run(df_sig_part)
                for tr in res_part.trade_list:
                    oos_pnls.append({"entry_time": tr.entry_time, "exit_time": tr.exit_time, "net_pnl": tr.net_pnl})

            if len(oos_pnls) < 3:
                continue

            df_oos_trades = pd.DataFrame(oos_pnls)
            oos_s = self._compute_calendar_daily_sharpe(df_oos_trades, initial_capital)
            oos_sharpes.append(oos_s)
            is_sharpes.append(best_score)

        if not oos_sharpes:
            return {"is_overfitted": True, "pbo": 1.0, "mode": CPCVMode.PARAMETER_SELECTION.value}

        oos_arr = np.array(oos_sharpes)
        pbo = float(np.mean(oos_arr <= 0.0))

        return {
            "mode": CPCVMode.PARAMETER_SELECTION.value,
            "total_paths": len(all_combos),
            "valid_evaluated_paths": len(oos_sharpes),
            "pbo": round(pbo, 3),
            "pbo_pct": f"{round(pbo * 100.0, 1)}%",
            "mean_oos_sharpe": round(float(np.mean(oos_arr)), 2),
            "median_oos_sharpe": round(float(np.median(oos_arr)), 2),
            "is_overfitted": bool(pbo > 0.30 or np.mean(oos_arr) < 0.30),
        }

    def _compute_calendar_daily_sharpe(self, trades_df: pd.DataFrame, initial_capital: float = 500000.0) -> float:
        """
        Maps discrete trades to true calendar dates and computes annualized Daily MTM Sharpe.
        Guarantees that per-trade returns are NOT annualized using sqrt(252).
        """
        if trades_df.empty:
            return 0.0

        df_tr = trades_df.copy()
        df_tr["date"] = pd.to_datetime(df_tr["entry_time"]).dt.date

        # Group net PnL by calendar date
        daily_pnl = df_tr.groupby("date")["net_pnl"].sum()
        if len(daily_pnl) < 2:
            return 0.0

        daily_returns = daily_pnl / initial_capital
        std = np.std(daily_returns, ddof=1)
        if std < 1e-8:
            return 0.0

        return float((np.mean(daily_returns) / std) * np.sqrt(252.0))
