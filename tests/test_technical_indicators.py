"""
Unit tests for Centralized Technical Indicators Toolbox
"""

import numpy as np
import pandas as pd
import pytest
from src.features.indicators import TechnicalIndicators as TI


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2026-08-17 09:15", periods=50, freq="15min")
    prices = [1000.0 + i * 2.0 + (i % 3) * 1.5 for i in range(50)]
    return pd.DataFrame({
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 3.0 for p in prices],
        "close": prices,
        "volume": [50000 + i * 1000 for i in range(50)],
    }, index=dates)


def test_moving_averages(sample_ohlcv):
    df = TI.add_sma(sample_ohlcv, period=10)
    df = TI.add_ema(df, period=10)
    assert "sma_10" in df.columns
    assert "ema_10" in df.columns
    assert not df["sma_10"].iloc[15:].isna().any()
    assert not df["ema_10"].iloc[15:].isna().any()


def test_rsi_and_macd(sample_ohlcv):
    df = TI.add_rsi(sample_ohlcv, period=14)
    df = TI.add_macd(df, fast=12, slow=26, signal=9)
    assert "rsi_14" in df.columns
    assert "macd_line" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns
    assert df["rsi_14"].between(0, 100).all()


def test_supertrend(sample_ohlcv):
    df = TI.add_supertrend(sample_ohlcv, period=10, multiplier=3.0)
    assert "supertrend_10_3.0" in df.columns
    assert "supertrend_direction_10_3.0" in df.columns
    # Direction should be +1 or -1
    directions = set(df["supertrend_direction_10_3.0"].iloc[1:])
    assert directions.issubset({1.0, -1.0})


def test_volatility_indicators(sample_ohlcv):
    df = TI.add_atr(sample_ohlcv, period=14)
    df = TI.add_bollinger_bands(df, window=20, num_std=2.0)
    df = TI.add_keltner_channels(df, ema_period=20, atr_period=10)
    assert "atr_14" in df.columns
    assert "bb_upper_20" in df.columns
    assert "bb_lower_20" in df.columns
    assert "kc_upper_20" in df.columns
    assert (df["bb_upper_20"].iloc[20:] >= df["bb_lower_20"].iloc[20:]).all()


def test_adx_and_donchian(sample_ohlcv):
    df = TI.add_adx(sample_ohlcv, period=14)
    df = TI.add_donchian_channels(df, window=20)
    df = TI.add_stochastic_oscillator(df, k_period=14, d_period=3)
    assert "adx_14" in df.columns
    assert "donchian_upper_20" in df.columns
    assert "stoch_k_14" in df.columns
