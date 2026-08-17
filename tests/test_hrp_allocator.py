"""
Unit Tests for Hierarchical Risk Parity (HRP) Portfolio Allocator
"""

import numpy as np
import pandas as pd
import pytest
from src.portfolio.hrp_allocator import HierarchicalRiskParityAllocator


def test_hrp_allocation():
    np.random.seed(42)
    n = 100
    
    # Create 3 assets: Asset A (low vol), Asset B (high vol), Asset C (correlated with B)
    ret_a = np.random.normal(0.001, 0.01, n)
    ret_b = np.random.normal(0.002, 0.03, n)
    ret_c = ret_b * 0.8 + np.random.normal(0, 0.005, n)

    df_rets = pd.DataFrame({
        "RELIANCE": ret_a,
        "HDFCBANK": ret_b,
        "ICICIBANK": ret_c,
    })

    allocator = HierarchicalRiskParityAllocator()
    weights = allocator.allocate(df_rets)

    assert len(weights) == 3
    assert pytest.approx(sum(weights.values()), 0.001) == 1.0
    
    # All weights must be non-negative
    for asset, w in weights.items():
        assert w > 0.0

    # Low volatility asset should receive substantial allocation
    assert weights["RELIANCE"] > 0.30
