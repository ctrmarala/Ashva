"""
Unit Tests for Value at Risk (VaR) and CVaR Calculator
"""

import numpy as np
import pytest
from src.risk.var_calculator import RiskMetricsCalculator


def test_var_cvar_calculations():
    np.random.seed(42)
    # 500 return samples
    returns = np.random.normal(loc=0.0005, scale=0.015, size=500)

    # 1. Parametric VaR
    param_res = RiskMetricsCalculator.calculate_parametric_var(returns, confidence_level=0.95, portfolio_value=500000.0)
    assert "gaussian_var_pct" in param_res
    assert "cornish_fisher_var_pct" in param_res
    assert param_res["var_inr"] > 0.0

    # 2. Historical VaR and Expected Shortfall
    hist_res = RiskMetricsCalculator.calculate_historical_var_cvar(returns, confidence_level=0.95, portfolio_value=500000.0)
    assert "historical_var_pct" in hist_res
    assert "cvar_expected_shortfall_pct" in hist_res
    assert hist_res["cvar_expected_shortfall_pct"] >= hist_res["historical_var_pct"]
    assert hist_res["cvar_inr"] > 0.0
