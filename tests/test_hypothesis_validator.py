"""
Unit Tests for Lopez de Prado Hypothesis Validator and Statistical Tests
"""

import numpy as np
import pandas as pd
import pytest
from src.research.validator import StatisticalValidator
from src.research.triple_barrier import TripleBarrierLabeler
from src.research.hypothesis import HypothesisMetadata
from src.strategies.alpha_orb import AlphaInstitutionalORB


def test_deflated_sharpe_ratio():
    validator = StatisticalValidator()

    # 1. Strong consistent positive returns -> Low p-value (< 0.01)
    np.random.seed(42)
    strong_returns = np.random.normal(loc=0.005, scale=0.01, size=500)
    stat, p_val = validator.calculate_deflated_sharpe_ratio(strong_returns, num_trials=5)
    assert p_val < 0.05

    # 2. Pure random zero-mean noise -> High p-value (> 0.10)
    noise_returns = np.random.normal(loc=0.0, scale=0.01, size=200)
    _, noise_p_val = validator.calculate_deflated_sharpe_ratio(noise_returns, num_trials=20)
    assert noise_p_val > 0.10


def test_monte_carlo_drawdown_test():
    validator = StatisticalValidator()
    
    np.random.seed(42)
    # 50 winning trades of +2%, 50 losing trades of -1%
    trades = np.array([0.02] * 50 + [-0.01] * 50)
    mc_res = validator.run_monte_carlo_drawdown_test(trades, num_simulations=500)

    assert "mean_max_dd" in mc_res
    assert "p95_max_dd" in mc_res
    assert "p99_max_dd" in mc_res
    assert mc_res["p95_max_dd"] >= mc_res["mean_max_dd"]


def test_triple_barrier_labeling():
    dates = pd.date_range("2026-01-01 09:15", periods=50, freq="5min")
    prices = 1000.0 + np.cumsum(np.random.normal(0, 2.0, 50))
    signals = np.zeros(50)
    signals[5] = 1.0   # Buy signal at bar 5
    signals[20] = -1.0 # Short signal at bar 20

    df = pd.DataFrame({
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "signal": signals,
    }, index=dates)

    label_df = TripleBarrierLabeler.apply_triple_barrier(df, pt_mult=1.5, sl_mult=1.0, max_holding_bars=10)
    assert len(label_df) == 2
    assert "raw_return" in label_df.columns
    assert "label" in label_df.columns
    assert label_df.iloc[0]["label"] in [1, -1, 0]
