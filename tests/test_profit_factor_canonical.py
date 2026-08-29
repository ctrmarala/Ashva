"""
Deterministic Unit Tests for Canonical Profit Factor Calculation
Verifies:
1. sum(winning_trades) / abs(sum(losing_trades))
2. Zero losses edge case (returns 99.0)
3. Zero wins edge case (returns 0.0)
4. Empty trade list edge case (returns 0.0)
5. Consistency between calculate_profit_factor and calculate_trade_level_metrics
"""

import pytest
import numpy as np
import pandas as pd

from src.analytics.metrics import calculate_profit_factor, calculate_trade_level_metrics


def test_profit_factor_standard_case():
    # Wins: +1000, +2000, +500 = 3500
    # Losses: -500, -1000 = -1500
    # Expected PF: 3500 / 1500 = 2.3333...
    pnls = [1000.0, -500.0, 2000.0, -1000.0, 500.0]
    pf = calculate_profit_factor(pnls)
    assert pytest.approx(pf, 0.001) == 3500.0 / 1500.0

    metrics = calculate_trade_level_metrics(pnls)
    assert pytest.approx(metrics["net_profit_factor"], 0.01) == 2.33


def test_profit_factor_zero_losses():
    pnls = [1000.0, 2000.0, 500.0]
    pf = calculate_profit_factor(pnls)
    assert pf == 99.0


def test_profit_factor_zero_wins():
    pnls = [-500.0, -1000.0]
    pf = calculate_profit_factor(pnls)
    assert pf == 0.0


def test_profit_factor_empty():
    assert calculate_profit_factor([]) == 0.0
    assert calculate_profit_factor(np.array([])) == 0.0


def test_profit_factor_rejects_gross_to_cost_proxy():
    # Prove that PF is NOT gross_pnl / costs
    # Wins: +10000, Losses: -8000 -> PF = 10000 / 8000 = 1.25
    pnls = [10000.0, -8000.0]
    pf = calculate_profit_factor(pnls)
    assert pytest.approx(pf, 0.001) == 1.25