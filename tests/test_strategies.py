"""
Unit Tests for TrendSurfer Pro Alpha Strategy Signal Generation
"""

import numpy as np
import pandas as pd
import pytest

alpha_ts = pytest.importorskip("src.strategies.alpha_trend_surfer", reason="alpha_trend_surfer archived in clean baseline")
AlphaTrendSurfer = alpha_ts.AlphaTrendSurfer


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


def test_alpha_08_opening_imbalance_signals(mock_intraday_dataframe):
    from src.strategies.alpha_08_opening_imbalance import Alpha08OpeningImbalance
    strat = Alpha08OpeningImbalance()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "max_adverse_wick_ratio" in grid
    assert "min_body_ratio" in grid
    assert "min_range_atr_ratio" in grid
    assert "min_rvol" in grid
    assert "target_rr" in grid


def test_alpha_09_opening_relative_strength_signals(mock_intraday_dataframe):
    from src.strategies.alpha_09_opening_relative_strength import Alpha09OpeningRelativeStrength
    strat = Alpha09OpeningRelativeStrength()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "min_rs_threshold" in grid
    assert "min_body_ratio" in grid
    assert "min_rvol" in grid
    assert "target_rr" in grid


def test_alpha_10_statistical_range_reversion_signals(mock_intraday_dataframe):
    from src.strategies.alpha_10_statistical_range_reversion import Alpha10StatisticalRangeReversion
    strat = Alpha10StatisticalRangeReversion()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "range_window_days" in grid
    assert "max_daily_adx" in grid
    assert "min_reward_risk_ratio" in grid
    assert "stop_buffer_atr_mult" in grid


def test_alpha_11_donchian_breakout_signals(mock_intraday_dataframe):
    from src.strategies.alpha_11_donchian_breakout import Alpha11DonchianBreakout
    strat = Alpha11DonchianBreakout()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "donchian_window_days" in grid
    assert "min_daily_adx" in grid
    assert "stop_atr_mult" in grid
    assert "target_atr_mult" in grid


def test_alpha_12_european_open_momentum_signals(mock_intraday_dataframe):
    from src.strategies.alpha_12_european_open_momentum import Alpha12EuropeanOpenMomentum
    strat = Alpha12EuropeanOpenMomentum()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "min_rvol" in grid
    assert "target_rr" in grid
    assert "max_box_atr_ratio" in grid


def test_alpha_13_htf_aligned_orb_signals(mock_intraday_dataframe):
    from src.strategies.alpha_13_htf_aligned_orb import Alpha13HTFAlignedORB
    strat = Alpha13HTFAlignedORB()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)

    unique_sigs = set(signals_df["signal"].unique())
    assert unique_sigs.issubset({-1.0, 0.0, 1.0})

    grid = strat.get_parameter_grid()
    assert "min_rvol" in grid
    assert "target_rr" in grid
    assert "max_or_atr_ratio" in grid


def test_alpha_14_gap_momentum_drift_signals(mock_intraday_dataframe):
    from src.strategies.alpha_14_gap_momentum_drift import Alpha14GapMomentumDrift
    strat = Alpha14GapMomentumDrift()
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
    assert "min_body_ratio" in grid
    assert "min_rvol" in grid
    assert "target_rr" in grid


def test_alpha_15_nr7_volatility_expansion_signals(mock_intraday_dataframe):
    from src.strategies.alpha_15_nr7_volatility_expansion import Alpha15NR7VolatilityExpansion
    strat = Alpha15NR7VolatilityExpansion()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()


def test_alpha_16_inside_day_breakout_signals(mock_intraday_dataframe):
    from src.strategies.alpha_16_inside_day_breakout import Alpha16InsideDayBreakout
    strat = Alpha16InsideDayBreakout()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()


def test_alpha_17_volume_shock_momentum_signals(mock_intraday_dataframe):
    from src.strategies.alpha_17_volume_shock_momentum import Alpha17VolumeShockMomentum
    strat = Alpha17VolumeShockMomentum()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_shock_rvol" in strat.get_parameter_grid()


def test_alpha_18_three_day_trend_orb_signals(mock_intraday_dataframe):
    from src.strategies.alpha_18_three_day_trend_orb import Alpha18ThreeDayTrendORB
    strat = Alpha18ThreeDayTrendORB()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()


def test_alpha_19_power_hour_momentum_signals(mock_intraday_dataframe):
    from src.strategies.alpha_19_power_hour_momentum import Alpha19PowerHourMomentum
    strat = Alpha19PowerHourMomentum()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()


def test_alpha_20_vwap_trend_continuation_signals(mock_intraday_dataframe):
    from src.strategies.alpha_20_vwap_trend_continuation import Alpha20VWAPTrendContinuation
    strat = Alpha20VWAPTrendContinuation()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()


def test_alpha_21_high_velocity_momentum_signals(mock_intraday_dataframe):
    from src.strategies.alpha_21_high_velocity_momentum import Alpha21HighVelocityMomentum
    strat = Alpha21HighVelocityMomentum()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()
    assert "target_rr" in strat.get_parameter_grid()


def test_alpha_22_apex_momentum_signals(mock_intraday_dataframe):
    from src.strategies.alpha_22_apex_momentum import Alpha22ApexMomentum
    strat = Alpha22ApexMomentum()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()
    assert "target_rr" in strat.get_parameter_grid()


def test_alpha_23_velocity_50_scanner_signals(mock_intraday_dataframe):
    from src.strategies.alpha_23_velocity_50_scanner import Alpha23Velocity50Scanner
    strat = Alpha23Velocity50Scanner()
    signals_df = strat.generate_signals(mock_intraday_dataframe)

    assert "signal" in signals_df.columns
    assert "stop_loss" in signals_df.columns
    assert "take_profit" in signals_df.columns
    assert "rationale" in signals_df.columns
    assert len(signals_df) == len(mock_intraday_dataframe)
    assert set(signals_df["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    assert "min_rvol" in strat.get_parameter_grid()
    assert "target_rr" in strat.get_parameter_grid()
