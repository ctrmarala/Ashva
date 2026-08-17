"""
Unit Tests for Machine Learning Meta-Labeling Strategy
"""

import numpy as np
import pandas as pd
import pytest
from src.strategies.alpha_orb import AlphaInstitutionalORB
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy


@pytest.fixture
def mock_multi_day_ohlcv():
    day1 = pd.date_range("2026-01-01 09:15", "2026-01-01 15:25", freq="5min")
    day2 = pd.date_range("2026-01-02 09:15", "2026-01-02 15:25", freq="5min")
    all_dates = day1.append(day2)

    np.random.seed(42)
    n = len(all_dates)
    prices = 2500.0 + np.cumsum(np.random.normal(0.5, 3.0, n))

    df = pd.DataFrame({
        "open": prices - 1.0,
        "high": prices + 3.0,
        "low": prices - 3.0,
        "close": prices,
        "volume": np.random.randint(5000, 25000, n),
    }, index=all_dates)
    return df


def test_meta_labeling_execution(mock_multi_day_ohlcv):
    primary_strat = AlphaInstitutionalORB()
    meta_strat = AlphaMetaLabeledStrategy(primary_strategy=primary_strat, parameters={"min_conviction_threshold": 0.50})

    # Fit and generate signals
    meta_strat.fit_meta_model(mock_multi_day_ohlcv)
    assert meta_strat.is_fitted is True

    signals_df = meta_strat.generate_signals(mock_multi_day_ohlcv)
    assert "signal" in signals_df.columns
    assert len(signals_df) == len(mock_multi_day_ohlcv)

    # Signals must be bounded between -1.0 and +1.0
    assert (signals_df["signal"] >= -1.0).all()
    assert (signals_df["signal"] <= 1.0).all()
