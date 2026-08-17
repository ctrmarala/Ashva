"""
Unit Tests for Market Microstructure & Order Flow Features
"""

import numpy as np
import pandas as pd
import pytest
from src.features.microstructure import MicrostructureFeatureExtractor


@pytest.fixture
def sample_intraday_ohlcv():
    dates = pd.date_range("2026-01-01 09:15", periods=50, freq="5min")
    np.random.seed(42)
    base_price = 2500.0
    returns = np.random.normal(0, 0.002, 50)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices * (1 - 0.0005),
        "high": prices * (1 + 0.0015),
        "low": prices * (1 - 0.0015),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 50),
    }, index=dates)
    return df


def test_anchored_vwap(sample_intraday_ohlcv):
    extractor = MicrostructureFeatureExtractor()
    df_vwap = extractor.calculate_anchored_vwap(sample_intraday_ohlcv)

    assert "vwap" in df_vwap.columns
    assert "vwap_upper_1sd" in df_vwap.columns
    assert "vwap_lower_1sd" in df_vwap.columns
    assert "vwap_upper_2sd" in df_vwap.columns
    assert "vwap_lower_2sd" in df_vwap.columns

    # VWAP upper band should always be greater than VWAP
    assert (df_vwap["vwap_upper_1sd"] >= df_vwap["vwap"]).all()
    assert (df_vwap["vwap_lower_1sd"] <= df_vwap["vwap"]).all()


def test_volume_delta(sample_intraday_ohlcv):
    extractor = MicrostructureFeatureExtractor()
    df_vd = extractor.calculate_volume_delta(sample_intraday_ohlcv)

    assert "volume_delta" in df_vd.columns
    assert "cvd" in df_vd.columns
    assert "volume_surge_ratio" in df_vd.columns
    assert len(df_vd) == len(sample_intraday_ohlcv)


def test_opening_range_bands(sample_intraday_ohlcv):
    extractor = MicrostructureFeatureExtractor()
    df_orb = extractor.calculate_opening_range(sample_intraday_ohlcv, orb_start="09:15:00", orb_end="09:30:00")

    assert "orb_high" in df_orb.columns
    assert "orb_low" in df_orb.columns
    assert "orb_volume" in df_orb.columns
    
    # ORH must be greater than or equal to ORL
    valid_mask = df_orb["orb_high"].notna()
    assert (df_orb.loc[valid_mask, "orb_high"] >= df_orb.loc[valid_mask, "orb_low"]).all()


def test_hurst_exponent():
    extractor = MicrostructureFeatureExtractor()
    
    # 1. Pure deterministic trending line -> Hurst should be > 0.55
    trend = pd.Series(np.linspace(100, 200, 200))
    h_trend = extractor.calculate_hurst_exponent(trend)
    assert h_trend >= 0.55

    # 2. Mean reverting Ornstein-Uhlenbeck / stationary oscillations -> Hurst should be < 0.50
    np.random.seed(42)
    ou_series = [100.0]
    for _ in range(300):
        # Mean revert towards 100 with theta=0.5
        next_val = ou_series[-1] + 0.5 * (100.0 - ou_series[-1]) + np.random.normal(0, 1.0)
        ou_series.append(next_val)
    
    h_mr = extractor.calculate_hurst_exponent(pd.Series(ou_series))
    assert h_mr <= 0.50
