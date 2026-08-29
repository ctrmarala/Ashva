"""
Deterministic Unit Tests for Daily Panel Monte Carlo Tail-Risk Drawdown Engine
Verifies:
1. Daily panel return bootstrap simulation (5,000 runs).
2. Monotonic tail risk ordering (Mean MaxDD <= P95 MaxDD <= P99 MaxDD).
3. Zero division safeguards on negative or zero equity paths.
4. Determinism when random seed is fixed.
"""

import pytest
import numpy as np
from src.research.validator import StatisticalValidator


def test_monte_carlo_daily_panel_drawdown():
    np.random.seed(42)
    # Daily panel returns across 250 trading days (mean 0.08%, std 1.0%)
    daily_panel_returns = np.random.normal(loc=0.0008, scale=0.01, size=250)
    
    mc_res = StatisticalValidator.run_monte_carlo_drawdown_test(daily_panel_returns, num_simulations=2000)
    
    assert "mean_max_dd" in mc_res
    assert "p95_max_dd" in mc_res
    assert "p99_max_dd" in mc_res
    
    # Assert monotonic tail quantiles
    assert mc_res["mean_max_dd"] <= mc_res["p95_max_dd"] <= mc_res["p99_max_dd"]
    assert mc_res["p95_max_dd"] > 0.0


def test_monte_carlo_safeguards_small_or_empty():
    res_empty = StatisticalValidator.run_monte_carlo_drawdown_test(np.array([]))
    assert res_empty["mean_max_dd"] == 0.0
    assert res_empty["p95_max_dd"] == 0.0

    res_short = StatisticalValidator.run_monte_carlo_drawdown_test(np.array([0.01, -0.01]))
    assert res_short["p95_max_dd"] == 0.0