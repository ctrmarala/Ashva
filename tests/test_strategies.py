"""
Unit Tests for TrendSurfer Pro Alpha Strategy Signal Generation
"""

import numpy as np
import pandas as pd
import pytest
from src.strategies.alpha_trend_surfer import AlphaTrendSurfer


@pytest.fixture
def mock_intraday_dataframe():
    # 2 full trading days of 15-min bars (50 bars)
    dates = pd.date_range("2026-08-17 09:15", periods=50, freq="15min")
    prices = [1000.0 + i * 2.5 + (i % 4) * 1.5 for i in range(50)]

    df = pd.DataFrame({
        "open": [p - 1.0 for p in prices],
        "high": [p + 4.0 for p in prices],
        "low": [p - 4.0 for p in prices],
        "close": prices,
        "volume": [50000 + i * 1000 for i in range(50)],
    }, index=dates)
    return df


def test_alpha_trend_surfer_signals(mock_intraday_dataframe):
    strat = AlphaTrendSurfer()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    # Parameter search grid must be defined for DSR
    grid = strat.get_parameter_grid()
    assert "ema_period" in grid
    assert "supertrend_multiplier" in grid


def test_alpha_auction_orb_pro_signals(mock_intraday_dataframe):
    from src.strategies.alpha_orb_pro import AlphaAuctionORBPro
    orb = AlphaAuctionORBPro()
    signals_df = orb.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = orb.get_parameter_grid()
    assert "volume_mult" in grid
    assert "min_adx" in grid
