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


def test_alpha_03_vwap_reversion_signals(mock_intraday_dataframe):
    from src.strategies.alpha_03_vwap_reversion import Alpha03VWAPReversion
    strat = Alpha03VWAPReversion()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "vwap_std_multiplier" in grid
    assert "max_adx_balanced" in grid


def test_alpha_04_gap_and_go_signals(mock_intraday_dataframe):
    from src.strategies.alpha_04_gap_and_go import Alpha04GapAndGo
    strat = Alpha04GapAndGo()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "min_gap_pct" in grid
    assert "max_gap_pct" in grid
    assert "rvol_mult" in grid
    assert "min_adx" in grid
    assert "rr_ratio" in grid


def test_alpha_05_opening_drive_pullback_signals(mock_intraday_dataframe):
    from src.strategies.alpha_05_opening_drive_pullback import Alpha05OpeningDrivePullback
    strat = Alpha05OpeningDrivePullback()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "min_body_ratio" in grid
    assert "min_rvol" in grid
    assert "pullback_vol_max_ratio" in grid
    assert "target_rr" in grid


def test_alpha_06_pdh_pdl_sweep_signals(mock_intraday_dataframe):
    from src.strategies.alpha_06_pdh_pdl_sweep import Alpha06PDHPDLSweep
    strat = Alpha06PDHPDLSweep()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "max_sweep_atr_ratio" in grid
    assert "min_wick_ratio" in grid
    assert "stop_atr_buffer" in grid
    assert "target_rr" in grid


def test_alpha_07_opening_volatility_expansion_signals(mock_intraday_dataframe):
    from src.strategies.alpha_07_opening_volatility_expansion import Alpha07OpeningVolatilityExpansion
    strat = Alpha07OpeningVolatilityExpansion()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "max_compression_atr_ratio" in grid
    assert "min_rvol" in grid
    assert "target_rr" in grid
