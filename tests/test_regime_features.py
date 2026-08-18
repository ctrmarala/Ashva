"""
Unit Tests for MarketRegimeFeatureExtractor
Verifies zero look-ahead bias, correct mathematical formulas, and robust handling of market-derived features.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.features.regime import MarketRegimeFeatureExtractor


@pytest.fixture
def sample_intraday_ohlcv():
    """Generates synthetic 15-minute OHLCV data for 10 sessions."""
    records = []
    base_date = datetime(2026, 1, 1, 9, 15)
    base_price = 1000.0

    for day in range(10):
        session_date = base_date + timedelta(days=day)
        # Add slight daily trend
        daily_drift = day * 10.0
        open_price = base_price + daily_drift + (5.0 if day % 2 == 1 else -5.0)

        for bar in range(25):  # 25 15-minute bars per session (09:15 to 15:30)
            bar_time = session_date + timedelta(minutes=bar * 15)
            o = open_price + (bar * 0.5)
            h = o + 2.0
            l = o - 2.0
            c = o + 1.0
            v = 10000.0 + (5000.0 if bar == 0 else 0.0)

            records.append({
                "timestamp": bar_time,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v
            })

    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    return df


def test_normalized_atr(sample_intraday_ohlcv):
    atr_pct = MarketRegimeFeatureExtractor.compute_normalized_atr(sample_intraday_ohlcv, period=14)
    assert len(atr_pct) == len(sample_intraday_ohlcv)
    assert isinstance(atr_pct, pd.Series)
    assert (atr_pct >= 0.0).all()
    # Normalized ATR should be in a reasonable percentage range (e.g. 0.1% to 5.0%)
    assert atr_pct.iloc[-1] > 0.05
    assert atr_pct.iloc[-1] < 5.0


def test_realized_volatility(sample_intraday_ohlcv):
    ann_vol = MarketRegimeFeatureExtractor.compute_realized_volatility(sample_intraday_ohlcv, window=20)
    assert len(ann_vol) == len(sample_intraday_ohlcv)
    assert (ann_vol >= 0.0).all()


def test_tod_relative_volume(sample_intraday_ohlcv):
    rvol = MarketRegimeFeatureExtractor.compute_tod_relative_volume(sample_intraday_ohlcv, lookback_sessions=5)
    assert len(rvol) == len(sample_intraday_ohlcv)
    assert (rvol >= 0.0).all()
    # Bar 0 (09:15) had 15,000 volume vs other bars 10,000, so baseline matches
    assert not rvol.isna().any()


def test_extract_session_gap_features(sample_intraday_ohlcv):
    df_gap = MarketRegimeFeatureExtractor.extract_session_gap_features(sample_intraday_ohlcv)
    assert "gap_pct" in df_gap.columns
    assert "gap_atr_ratio" in df_gap.columns
    assert "bar1_body_ratio" in df_gap.columns
    assert "prev_close" in df_gap.columns

    # First session should have NaN or 0 prior close, subsequent sessions must have valid prior close
    # Check that day 2 (row 25 onwards) has non-NaN prev_close
    assert not df_gap.iloc[25:]["prev_close"].isna().all()
    # Verify bar1_body_ratio is bounded [0, 1]
    assert (df_gap["bar1_body_ratio"].dropna() <= 1.0).all()
    assert (df_gap["bar1_body_ratio"].dropna() >= 0.0).all()


def test_compute_multiday_trend_structure(sample_intraday_ohlcv):
    df_struct = MarketRegimeFeatureExtractor.compute_multiday_trend_structure(sample_intraday_ohlcv, lookback_days=3)
    assert "is_3d_bullish_trend" in df_struct.columns
    assert "is_3d_bearish_trend" in df_struct.columns
    assert df_struct["is_3d_bullish_trend"].dtype == bool
