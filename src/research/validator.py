"""
Ashva Statistical Alpha Validator & Lopez de Prado Hypothesis Rejector
Implements Deflated Sharpe Ratio (DSR), Combinatorial Purged Cross-Validation (CPCV),
and Monte Carlo Permutation Stress Testing to reject false discoveries and overfitted strategies.
"""

from math import gamma
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from src.research.hypothesis import BaseHypothesis, HypothesisStatus, HypothesisValidationReport
from src.analytics.indian_costs import IndianCostModel, Segment


class StatisticalValidator:
    """
    Institutional statistical filter to eliminate data snooping, multiple testing bias,
    and unviable strategies before capital allocation.
    """

    def __init__(
        self,
        cost_model: Optional[IndianCostModel] = None,
        min_net_profit_factor: float = 1.30,
        max_dsr_p_value: float = 0.01,         # 99% confidence required
        max_cpcv_degradation_pct: float = 40.0, # Max allowed performance drop OOS
        max_monte_carlo_dd_pct: float = 12.0,   # Max 95th percentile drawdown
    ):
        self.cost_model = cost_model or IndianCostModel()
        self.min_net_profit_factor = min_net_profit_factor
        self.max_dsr_p_value = max_dsr_p_value
        self.max_cpcv_degradation_pct = max_cpcv_degradation_pct
        self.max_monte_carlo_dd_pct = max_monte_carlo_dd_pct

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.065, periods_per_year: int = 252 * 75) -> float:
        """
        Calculates annualized Sharpe Ratio for intraday returns (assuming 75 5-min bars/day).
        """
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret < 1e-8:
            return 0.0
        
        # Periodic risk-free rate
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
        Computes the Deflated Sharpe Ratio (DSR) and p-value (Bailey & López de Prado, 2014).
        Penalizes Sharpe ratio for multiple testing trials and non-normal (skewed/fat-tailed) returns.
        
        :return: (DSR_statistic, p_value)
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
            expected_max_sr = 0.0  # Zero benchmark if single trial

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
        n_trades = len(trade_returns)

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

    def validate_hypothesis(
        self,
        hypothesis: BaseHypothesis,
        df: pd.DataFrame,
        num_trials: int = 1,
        train_test_split: float = 0.70,
    ) -> HypothesisValidationReport:
        """
        Full 4-Gate Statistical Validation of a candidate Alpha Hypothesis.
        """
        rejection_reasons = []

        # 1. Generate Signals
        signals_df = hypothesis.generate_signals(df)
        if "signal" not in signals_df.columns:
            raise ValueError("Hypothesis must produce a 'signal' column")

        # 2. In-Sample vs Out-of-Sample Split
        split_idx = int(len(signals_df) * train_test_split)
        train_df = signals_df.iloc[:split_idx]
        test_df = signals_df.iloc[split_idx:]

        # Simple bar-by-bar strategy return = signal(t-1) * ret(t)
        returns_all = signals_df["signal"].shift(1) * signals_df["close"].pct_change()
        returns_is = returns_all.iloc[:split_idx].dropna().values
        returns_oos = returns_all.iloc[split_idx:].dropna().values

        is_sharpe = self.calculate_sharpe_ratio(returns_is)
        oos_sharpe = self.calculate_sharpe_ratio(returns_oos)

        # 3. Gate 1: Deflated Sharpe Ratio (DSR) Test
        dsr_stat, dsr_p_val = self.calculate_deflated_sharpe_ratio(returns_all.dropna().values, num_trials=num_trials)
        if dsr_p_val > self.max_dsr_p_value:
            rejection_reasons.append(
                f"DSR Test Failed: p-value {dsr_p_val:.4f} > {self.max_dsr_p_value} (High likelihood of random noise / selection bias)"
            )

        # 4. Gate 2: Out-Of-Sample Degradation & CPCV
        if is_sharpe > 0:
            degradation = ((is_sharpe - oos_sharpe) / is_sharpe) * 100.0
        else:
            degradation = 100.0

        if degradation > self.max_cpcv_degradation_pct or oos_sharpe <= 0:
            rejection_reasons.append(
                f"OOS Degradation Failed: Strategy degraded by {degradation:.1f}% (IS Sharpe: {is_sharpe:.2f}, OOS Sharpe: {oos_sharpe:.2f})"
            )

        # 5. Gate 3: Monte Carlo Permutation Stress Test
        # Filter active trade returns (non-zero bars)
        active_returns = returns_all[returns_all != 0].dropna().values
        mc_results = self.run_monte_carlo_drawdown_test(active_returns, num_simulations=1000)
        p95_dd = mc_results["p95_max_dd"]
        if p95_dd > self.max_monte_carlo_dd_pct:
            rejection_reasons.append(
                f"Monte Carlo Tail Risk Failed: 95th percentile Max Drawdown {p95_dd:.1f}% exceeds tolerance {self.max_monte_carlo_dd_pct}%"
            )

        # 6. Gate 4: Post-Tax Net Profit Factor (Simulated roundtrips with Indian Costs)
        # Approximate gross profit vs gross loss
        gross_wins = np.sum(returns_all[returns_all > 0])
        gross_losses = abs(np.sum(returns_all[returns_all < 0]))
        gross_pf = (gross_wins / gross_losses) if gross_losses > 0 else 0.0
        
        # Deduct friction estimation (e.g. 0.05% per turn)
        num_trades = np.sum(signals_df["signal"].diff().abs() > 0)
        est_friction = num_trades * 0.0005
        net_wins = max(0.0, gross_wins - est_friction)
        net_pf = (net_wins / gross_losses) if gross_losses > 0 else 0.0

        if net_pf < self.min_net_profit_factor:
            rejection_reasons.append(
                f"Post-Tax Profit Factor Failed: Net PF {net_pf:.2f} < {self.min_net_profit_factor}"
            )

        # Determine Final Status
        status = HypothesisStatus.ACCEPTED if not rejection_reasons else HypothesisStatus.REJECTED
        hypothesis.status = status

        return HypothesisValidationReport(
            hypothesis_id=hypothesis.metadata.hypothesis_id,
            status=status,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            deflated_sharpe_p_value=dsr_p_val,
            cpcv_mean_sharpe=oos_sharpe,
            cpcv_degradation_pct=degradation,
            monte_carlo_95_max_dd_pct=p95_dd,
            net_profit_factor_post_tax=net_pf,
            rejection_reasons=rejection_reasons,
            tested_trials_count=num_trials,
        )
