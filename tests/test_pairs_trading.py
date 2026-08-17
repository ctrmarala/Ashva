"""
Unit Tests for Cointegration and Statistical Arbitrage Pairs Strategy
"""

import numpy as np
import pandas as pd
import pytest
from src.strategies.alpha_pairs import AlphaCointegrationPairs


def test_cointegration_and_spread_calculation():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2026-01-01 09:15", periods=n, freq="5min")

    # Generate cointegrated series: B = Random Walk, A = 1.5 * B + stationary noise
    noise = np.random.normal(0, 1.0, n)
    price_b = 1000.0 + np.cumsum(np.random.normal(0, 2.0, n))
    price_a = 1.5 * price_b + noise

    s_a = pd.Series(price_a, index=dates)
    s_b = pd.Series(price_b, index=dates)

    strat = AlphaCointegrationPairs()

    # 1. Cointegration Test
    score, p_val, is_coint = strat.test_cointegration(s_a, s_b)
    assert p_val < 0.05
    assert is_coint is True

    # 2. Spread and Z-Score
    spread, z_score, beta = strat.calculate_spread_and_zscore(s_a, s_b, lookback=40)
    assert pytest.approx(beta, 0.1) == 1.5
    assert len(z_score) == n
