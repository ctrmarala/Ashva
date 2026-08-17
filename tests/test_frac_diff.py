"""
Unit Tests for Fractional Differentiation Engine
"""

import numpy as np
import pandas as pd
import pytest
from src.features.frac_diff import get_weights_ffd, frac_diff_ffd, find_min_d_stationarity


def test_fractional_weights():
    # d = 0 gives single weight of 1.0
    w_0 = get_weights_ffd(d=0.0)
    assert len(w_0) == 1
    assert w_0[0] == 1.0

    # d = 1 gives standard first difference [-1.0, 1.0]
    w_1 = get_weights_ffd(d=1.0)
    assert len(w_1) == 2
    assert w_1[0] == -1.0
    assert w_1[1] == 1.0

    # Fractional weights decay smoothly
    w_half = get_weights_ffd(d=0.5, threshold=1e-3)
    assert len(w_half) > 5
    assert w_half[-1] == 1.0  # Most recent weight is 1.0


def test_frac_diff_ffd_execution():
    # Create trending random walk
    np.random.seed(42)
    n = 100
    prices = 1000.0 + np.cumsum(np.random.normal(loc=0.5, scale=2.0, size=n))
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    series = pd.Series(prices, index=dates, name="price")

    # Apply d = 0.4
    fd_series = frac_diff_ffd(series, d=0.4, threshold=1e-3)
    assert not fd_series.empty
    assert len(fd_series) < len(series)  # Window width dropped
    assert fd_series.name == "price_fracdiff_d0.40"


def test_find_min_d_stationarity():
    np.random.seed(42)
    n = 150
    prices = 500.0 + np.cumsum(np.random.normal(loc=0.2, scale=1.5, size=n))
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    series = pd.Series(prices, index=dates, name="close")

    result = find_min_d_stationarity(series, d_step=0.2)
    assert "optimal_d" in result
    assert 0.0 <= result["optimal_d"] <= 1.0
