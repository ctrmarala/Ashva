"""
Ashva Institutional Statistical Alpha Validator & Lopez de Prado Hypothesis Rejector
Implements:
1. True Combinatorial Purged & Embargoed Cross-Validation (CPCV) with (N choose k) path combinations.
2. Deflated Sharpe Ratio (DSR) with Bailey & López de Prado (2014) non-normal asymptotic corrections.
3. 5,000+ Run Monte Carlo Permutation Stress Testing for tail-risk drawdown estimation.
4. Exact trade-by-trade Indian Regulatory Cost Modeling (STT, GST, SEBI, ₹20 Brokerage) via BacktestEngine.
"""

from datetime import datetime
import logging
import json
from math import comb
from itertools import combinations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from src.research.hypothesis import BaseHypothesis, HypothesisStatus, HypothesisValidationReport
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Institutional statistical filter to eliminate data snooping, multiple testing bias,
    and unviable strategies before capital allocation.
    """

    def __init__(
        self,
        cost_model: Optional[IndianCostModel] = None,
        experiment_ledger: Optional[ResearchExperimentLedger] = None,
        min_net_profit_factor: float = 1.20,
        max_dsr_p_value: float = 0.05,          # Statistical significance threshold
        max_cpcv_degradation_pct: float = 50.0, # Max allowed performance drop OOS
        max_monte_carlo_dd_pct: float = 15.0,   # Max 95th percentile drawdown
    ):
        self.cost_model = cost_model or IndianCostModel()
        self.experiment_ledger = experiment_ledger or ResearchExperimentLedger()
        self.min_net_profit_factor = min_net_profit_factor
        self.max_dsr_p_value = max_dsr_p_value
        self.max_cpcv_degradation_pct = max_cpcv_degradation_pct
        self.max_monte_carlo_dd_pct = max_monte_carlo_dd_pct

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.065, periods_per_year: int = 252 * 25) -> float:
        """
        Calculates annualized Sharpe Ratio for intraday returns.
        """
        clean_ret = returns[~np.isnan(returns)]
        if len(clean_ret) < 2:
            return 0.0
        mean_ret = np.mean(clean_ret)
        std_ret = np.std(clean_ret, ddof=1)
        if std_ret < 1e-8:
            return 0.0
        
        rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        excess_mean = mean_ret - rf_per_period
        annualized_sharpe = (excess_mean / std_ret) * np.sqrt(periods_per_year)
        return float(annualized_sharpe)

    @classmethod
    def calculate_deflated_sharpe_ratio(
        cls,
        strategy_returns: np.ndarray,
        num_trials: int = 1,
        benchmark_sharpe_var: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Computes Deflated Sharpe Ratio (DSR) and p-value (Bailey & López de Prado, 2014).
        Penalizes Sharpe ratio for multiple testing trials and non-normal (skewed/fat-tailed) returns.
        """
        clean_ret = strategy_returns[~np.isnan(strategy_returns)]
        t = len(clean_ret)
        if t < 5:
            return 0.0, 1.0

        sr = cls.calculate_sharpe_ratio(clean_ret)
        if sr <= 0:
            return 0.0, 1.0

        gamma_3 = float(skew(clean_ret))
        gamma_4 = float(kurtosis(clean_ret, fisher=False))  # Pearson kurtosis (normal=3)

        # Expected maximum Sharpe ratio among N independent trials under Null Hypothesis
        euler_gamma = 0.5772156649
        if num_trials > 1:
            z_1 = norm.ppf(1.0 - 1.0 / num_trials)
            z_2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
            expected_max_sr = np.sqrt(benchmark_sharpe_var) * ((1.0 - euler_gamma) * z_1 + euler_gamma * z_2)
        else:
            expected_max_sr = 0.0

        # Standard error adjustment for non-normality
        denom_sq = 1.0 - gamma_3 * sr + ((gamma_4 - 1.0) / 4.0) * (sr ** 2)
        if denom_sq <= 0:
            denom_sq = 1.0
        sigma_sr = np.sqrt(denom_sq / max(t - 1, 1))

        # DSR Z-Score and p-value
        dsr_stat = (sr - expected_max_sr) / sigma_sr
        p_value = float(1.0 - norm.cdf(dsr_stat))

        return float(dsr_stat), float(p_value)

    @classmethod
    def run_monte_carlo_drawdown_test(
        cls,
        trade_returns: np.ndarray,
        num_simulations: int = 5000,
    ) -> Dict[str, float]:
        """
        Simulates 5,000+ random permutations of trade return sequences
        to evaluate the true tail risk and maximum drawdown distribution.
        """
        if len(trade_returns) < 5:
            return {"mean_max_dd": 0.0, "p95_max_dd": 0.0, "p99_max_dd": 0.0}

        max_drawdowns = []
        for _ in range(num_simulations):
            shuffled = np.random.permutation(trade_returns)
            equity_curve = np.cumprod(1.0 + shuffled)
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / peak
            max_drawdowns.append(np.max(drawdowns) * 100.0)

        dd_array = np.array(max_drawdowns)
        return {
            "mean_max_dd": float(np.mean(dd_array)),
            "p95_max_dd": float(np.percentile(dd_array, 95)),
            "p99_max_dd": float(np.percentile(dd_array, 99)),
        }

    def run_cpcv(
        self,
        df: pd.DataFrame,
        hypothesis: BaseHypothesis,
        n_splits: int = 6,
        k_test: int = 2,
        purge_bars: int = 5,
        embargo_bars: int = 5,
    ) -> Dict[str, Any]:
        """
        True Combinatorial Purged & Embargoed Cross-Validation (CPCV).
        1. Partitions series into N non-overlapping temporal blocks.
        2. Generates all (N choose k) out-of-sample path combinations.
        3. Applies Purging: Drops train samples within purge_bars of test boundary to prevent label overlap.
        4. Applies Embargoing: Enforces embargo_bars blackout buffer after test slices before subsequent training.
        """
        n_bars = len(df)
        if n_bars < 100:
            return {"mean_oos_sharpe": 0.0, "std_oos_sharpe": 0.0, "path_results": []}

        block_size = n_bars // n_splits
        blocks = []
        for i in range(n_splits):
            start = i * block_size
            end = (i + 1) * block_size if i < n_splits - 1 else n_bars
            blocks.append((start, end))

        test_combinations = list(combinations(range(n_splits), k_test))
        engine = BacktestEngine(cost_model=self.cost_model, initial_capital=500000.0)
        
        path_sharpes = []
        path_pfs = []
        path_drawdowns = []

        for combo in test_combinations:
            test_set_indices = set(combo)
            train_set_indices = set(range(n_splits)) - test_set_indices

            # 1. Construct Purged & Embargoed Training Set
            train_slices = []
            for tr_idx in sorted(train_set_indices):
                tr_start, tr_end = blocks[tr_idx]

                # If immediately preceded by a test block -> apply Embargo (shift start forward)
                if (tr_idx - 1) in test_set_indices:
                    tr_start = min(tr_start + embargo_bars, tr_end)

                # If immediately followed by a test block -> apply Purge (shift end backward)
                if (tr_idx + 1) in test_set_indices:
                    tr_end = max(tr_start, tr_end - purge_bars)

                if tr_end - tr_start > 10:
                    train_slices.append(df.iloc[tr_start:tr_end].copy())

            # 2. Construct Test Set (Out-of-Sample evaluation slices)
            test_slices = []
            for te_idx in sorted(test_set_indices):
                te_start, te_end = blocks[te_idx]
                test_slice = df.iloc[te_start:te_end].copy()
                if len(test_slice) > 10:
                    test_slices.append(test_slice)

            if not test_slices:
                continue

            test_combined = pd.concat(test_slices)

            # 3. Model Fitting on Purged/Embargoed Train Data (if model has trainable weights)
            if train_slices and hasattr(hypothesis, "fit_meta_model"):
                train_combined = pd.concat(train_slices)
                hypothesis.fit_meta_model(train_combined)

            # 4. Generate Signals and Backtest on untouched Test Slices
            sig_test = hypothesis.generate_signals(test_combined)
            res = engine.run(sig_test, symbol=getattr(hypothesis, "target_instruments", ["ASSET"])[0], capital_per_trade_pct=0.50)
            
            path_sharpes.append(res.sharpe_ratio)
            path_pfs.append(res.net_profit_factor)
            path_drawdowns.append(res.max_drawdown_pct)

        mean_sharpe = float(np.mean(path_sharpes)) if path_sharpes else 0.0
        std_sharpe = float(np.std(path_sharpes)) if path_sharpes else 0.0
        mean_pf = float(np.mean(path_pfs)) if path_pfs else 0.0
        mean_dd = float(np.mean(path_drawdowns)) if path_drawdowns else 0.0

        return {
            "total_paths": len(test_combinations),
            "mean_oos_sharpe": mean_sharpe,
            "std_oos_sharpe": std_sharpe,
            "mean_oos_net_profit_factor": mean_pf,
            "mean_oos_max_drawdown_pct": mean_dd,
            "path_sharpes": path_sharpes,
        }

    def validate_hypothesis(
        self,
        hypothesis: BaseHypothesis,
        df: pd.DataFrame,
        num_trials: Optional[int] = None,
        train_test_split: float = 0.70,
    ) -> HypothesisValidationReport:
        """
        Full 4-Gate Institutional Statistical Validation:
        Gate 1: Deflated Sharpe Ratio (DSR p <= 0.05) with automated multiple testing trial accounting
        Gate 2: Combinatorial Purged & Embargoed Cross-Validation (CPCV Degradation <= 50%)
        Gate 3: 5,000-Run Monte Carlo Tail Risk (95th MaxDD <= 15%)
        Gate 4: Full Indian Regulatory Cost Profit Factor (Net PF >= 1.20)
        """
        rejection_reasons = []

        # 1. Automated Trial Accounting for Multiple Testing (N)
        if num_trials is None or num_trials <= 1:
            grid = getattr(hypothesis, "parameter_grid", {})
            grid_size = 1
            for p_vals in grid.values():
                grid_size *= max(1, len(p_vals))
            n_symbols = len(getattr(hypothesis, "target_instruments", ["ASSET"]))
            trials_in_this_run = max(1, grid_size * n_symbols)
        else:
            trials_in_this_run = num_trials

        prior_trials = self.experiment_ledger.get_total_trials()
        effective_trials = prior_trials + trials_in_this_run

        # 2. Generate Signals and Run Baseline In-Sample / Full Backtests
        signals_df = hypothesis.generate_signals(df)
        if "signal" not in signals_df.columns:
            raise ValueError("Hypothesis must produce a 'signal' column")

        split_idx = int(len(signals_df) * train_test_split)
        train_df = signals_df.iloc[:split_idx]

        engine = BacktestEngine(cost_model=self.cost_model, initial_capital=500000.0)
        is_result = engine.run(train_df, symbol=getattr(hypothesis, "target_instruments", ["ASSET"])[0], capital_per_trade_pct=0.50)
        full_result = engine.run(signals_df, symbol=getattr(hypothesis, "target_instruments", ["ASSET"])[0], capital_per_trade_pct=0.50)

        is_sharpe = is_result.sharpe_ratio

        # 3. Gate 1: Deflated Sharpe Ratio (DSR) Test on Continuous Mark-to-Market Returns
        continuous_returns = full_result.equity_curve.pct_change().dropna().values
        dsr_stat, dsr_p_val = self.calculate_deflated_sharpe_ratio(continuous_returns, num_trials=effective_trials)
        if dsr_p_val > self.max_dsr_p_value:
            rejection_reasons.append(
                f"DSR Test Failed: p-value {dsr_p_val:.4f} > {self.max_dsr_p_value} across {effective_trials} tested trials (High probability of selection bias)"
            )

        # 4. Gate 2: True Combinatorial Purged Cross-Validation (CPCV)
        cpcv_results = self.run_cpcv(df, hypothesis, n_splits=6, k_test=2)
        cpcv_mean_sharpe = cpcv_results["mean_oos_sharpe"]

        if is_sharpe > 0:
            degradation = ((is_sharpe - cpcv_mean_sharpe) / is_sharpe) * 100.0
        else:
            degradation = 100.0 if cpcv_mean_sharpe <= 0 else 0.0

        # Accept if out-of-sample Sharpe remains strong (>= 1.0) or degradation is within bounds
        if (degradation > self.max_cpcv_degradation_pct and cpcv_mean_sharpe < 1.0) or cpcv_mean_sharpe <= 0:
            rejection_reasons.append(
                f"CPCV OOS Degradation Failed: Strategy degraded by {degradation:.1f}% across {cpcv_results['total_paths']} combinatorial paths (IS Sharpe: {is_sharpe:.2f}, CPCV OOS Mean: {cpcv_mean_sharpe:.2f})"
            )

        # 5. Gate 3: 5,000-Run Monte Carlo Permutation Stress Test
        trade_returns = np.array([t.net_pnl / 250000.0 for t in full_result.trade_list]) if full_result.trade_list else np.array([])
        mc_results = self.run_monte_carlo_drawdown_test(trade_returns, num_simulations=5000)
        p95_dd = mc_results["p95_max_dd"]
        if p95_dd > self.max_monte_carlo_dd_pct:
            rejection_reasons.append(
                f"Monte Carlo Tail Risk Failed: 95th percentile Max Drawdown {p95_dd:.1f}% exceeds tolerance {self.max_monte_carlo_dd_pct}%"
            )

        # 6. Gate 4: Real Post-Tax Net Profit Factor via IndianCostModel
        net_pf = full_result.net_profit_factor
        if net_pf < self.min_net_profit_factor:
            rejection_reasons.append(
                f"Post-Tax Profit Factor Failed: Real Net PF {net_pf:.2f} < {self.min_net_profit_factor} (Total Brokerage & STT: Rs {full_result.total_taxes_paid:,.2f})"
            )

        status = HypothesisStatus.ACCEPTED if not rejection_reasons else HypothesisStatus.REJECTED
        hypothesis.status = status

        report = HypothesisValidationReport(
            hypothesis_id=hypothesis.metadata.hypothesis_id,
            status=status,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=cpcv_mean_sharpe,
            deflated_sharpe_p_value=dsr_p_val,
            cpcv_mean_sharpe=cpcv_mean_sharpe,
            cpcv_degradation_pct=degradation,
            monte_carlo_95_max_dd_pct=p95_dd,
            net_profit_factor_post_tax=net_pf,
            rejection_reasons=rejection_reasons,
        )

        # Automatically record to immutable Research Experiment Ledger (Closed-Loop Trial Accounting)
        try:
            exp_record = ExperimentRecord(
                experiment_id=f"EXP_{hypothesis.metadata.hypothesis_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                strategy_id=hypothesis.metadata.name,
                symbol_universe=",".join(getattr(hypothesis, "target_instruments", ["ASSET"])),
                timeframe=getattr(hypothesis, "timeframe", "15m"),
                parameters_json=json.dumps(hypothesis.parameters, default=str),
                in_sample_sharpe=is_sharpe,
                cpcv_oos_sharpe=cpcv_mean_sharpe,
                deflated_sharpe_p_value=dsr_p_val,
                net_profit_factor=net_pf,
                monte_carlo_95_max_dd=p95_dd,
                trials_in_experiment=trials_in_this_run,
                total_trials_cumulative=effective_trials,
                git_commit_sha=get_current_git_sha(),
                status=status.value,
                rejection_reasons_json=json.dumps(rejection_reasons),
            )
            updated_count = self.experiment_ledger.log_experiment(exp_record)
            logger.info(f"Experiment logged to ledger: {exp_record.experiment_id} | Trials in run: {trials_in_this_run} | Total Cumulative Trials: {updated_count}")
        except Exception as e:
            logger.error(f"Failed to log experiment to ledger: {e}")

        return report
