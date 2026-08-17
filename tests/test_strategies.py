"""
Unit Tests for Alpha Strategies Signal Generation
"""

import numpy as np
import pandas as pd
import pytest
from src.strategies.alpha_orb import AlphaInstitutionalORB
from src.strategies.alpha_regime import AlphaRegimeAdaptiveMR


@pytest.fixture
def mock_intraday_dataframe():
    # 2 full trading days of 5-min bars (75 bars/day = 150 bars)
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


def test_alpha_orb_signals(mock_intraday_dataframe):
    orb = AlphaInstitutionalORB()
    signals_df = orb.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    # Signal values must be in [-1.0, 0.0, 1.0]
    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})


def test_alpha_regime_signals(mock_intraday_dataframe):
    regime = AlphaRegimeAdaptiveMR()
    signals_df = regime.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})


def test_alpha_bosch_aivo_signals(mock_intraday_dataframe):
    from src.strategies.alpha_bosch_aivo import AlphaInstitutionalValueOscillations
    aivo = AlphaInstitutionalValueOscillations()
    signals_df = aivo.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})
