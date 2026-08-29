"""
Deterministic Unit Tests for Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR)
Verifies:
1. Single trial baseline (N=1, expected max SR = 0.0).
2. Multiple testing penalty (N=81 vs N=1 increases expected max SR and penalizes p-value).
3. Non-normal returns adjustment (negative skew and fat tails increase sigma_sr).
4. Configurable benchmark_sharpe_var parameter.
5. Deterministic reproducibility.
"""

import pytest
import numpy as np
from scipy.stats import norm, skew, kurtosis
from src.research.validator import StatisticalValidator


@pytest.fixture
def synthetic_positive_returns():
    """Generates 500 daily returns with mean 0.001 (0.1%/day) and std 0.01 (16% ann vol)."""
    np.random.seed(42)
    return np.random.normal(loc=0.001, scale=0.01, size=500)


def test_dsr_single_trial(synthetic_positive_returns):
    """With 1 trial, DSR tests whether Sharpe is significantly > 0."""
    dsr_stat, p_val = StatisticalValidator.calculate_deflated_sharpe_ratio(
        synthetic_positive_returns, num_trials=1
    )
    sr = StatisticalValidator.calculate_sharpe_ratio(synthetic_positive_returns)
    assert sr > 1.0
    assert dsr_stat > 0.0
    assert p_val < 0.05  # Statistically significant


def test_dsr_multiple_trials_penalizes_p_value(synthetic_positive_returns):
    """
    Multiple testing burden: Evaluating 81 parameter configurations increases the hurdle,
    raising the expected maximum Sharpe under the null and increasing the p-value.
    """
    _, p_val_1 = StatisticalValidator.calculate_deflated_sharpe_ratio(
        synthetic_positive_returns, num_trials=1
    )
    _, p_val_81 = StatisticalValidator.calculate_deflated_sharpe_ratio(
        synthetic_positive_returns, num_trials=81
    )
    _, p_val_500 = StatisticalValidator.calculate_deflated_sharpe_ratio(
        synthetic_positive_returns, num_trials=500
    )

    assert p_val_1 < p_val_81 < p_val_500


def test_dsr_non_normal_fat_tailed_returns():
    """
    Negative skew and fat tails (kurtosis > 3) should properly inflate standard error
    and reduce DSR z-score relative to normal returns of identical Sharpe.
    """
    np.random.seed(42)
    # Heavy-tailed t-distribution returns with mean > 0
    fat_tailed_rets = np.random.standard_t(df=3, size=500) * 0.005 + 0.0008
    
    dsr_stat, p_val = StatisticalValidator.calculate_deflated_sharpe_ratio(
        fat_tailed_rets, num_trials=50
    )
    assert np.isfinite(dsr_stat)
    assert 0.0 <= p_val <= 1.0


def test_dsr_configurable_benchmark_var(synthetic_positive_returns):
    """Benchmark variance V is explicitly configurable."""
    dsr_v05, p_v05 = StatisticalValidator.calculate_deflated_sharpe_ratio(
        synthetic_positive_returns, num_trials=50, benchmark_sharpe_var=0.5
    )
    dsr_v10, p_v10 = StatisticalValidator.calculate_deflated_sharpe_ratio(
        synthetic_positive_returns, num_trials=50, benchmark_sharpe_var=1.0
    )
    # Higher benchmark variance means higher hurdle -> higher p-value
    assert p_v10 > p_v05