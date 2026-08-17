"""
Unit Tests for Ashva DataLake (DuckDB + Parquet)
"""

import os
import shutil
import tempfile
import pandas as pd
import pytest
from src.data.data_lake import DataLake


@pytest.fixture
def temp_data_lake():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_market.duckdb")
    parquet_dir = os.path.join(temp_dir, "parquet")
    dl = DataLake(db_path=db_path, parquet_dir=parquet_dir)
    yield dl
    dl.close()
    shutil.rmtree(temp_dir)


def test_save_and_load_bars(temp_data_lake):
    dates = pd.date_range("2026-01-01 09:15", periods=5, freq="5min")
    sample_df = pd.DataFrame({
        "timestamp": dates,
        "open": [2500.0, 2510.0, 2505.0, 2520.0, 2515.0],
        "high": [2515.0, 2520.0, 2515.0, 2530.0, 2525.0],
        "low": [2495.0, 2505.0, 2500.0, 2510.0, 2510.0],
        "close": [2510.0, 2505.0, 2520.0, 2515.0, 2522.0],
        "volume": [1000, 1500, 1200, 2000, 1800]
    })

    temp_data_lake.save_bars(df=sample_df, symbol="RELIANCE", timeframe="5m")

    loaded_df = temp_data_lake.load_bars(symbol="RELIANCE", timeframe="5m")
    assert len(loaded_df) == 5
    assert list(loaded_df.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded_df.iloc[0]["open"] == 2500.0
    assert loaded_df.iloc[-1]["close"] == 2522.0

    symbols = temp_data_lake.list_symbols(timeframe="5m")
    assert "RELIANCE" in symbols


def test_date_range_filtering(temp_data_lake):
    dates = pd.date_range("2026-01-01 09:15", periods=10, freq="5min")
    sample_df = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0 + i for i in range(10)],
        "high": [105.0 + i for i in range(10)],
        "low": [95.0 + i for i in range(10)],
        "close": [102.0 + i for i in range(10)],
        "volume": [500] * 10
    })

    temp_data_lake.save_bars(df=sample_df, symbol="INFY", timeframe="5m")

    # Filter only first 3 bars
    start_t = "2026-01-01 09:15:00"
    end_t = "2026-01-01 09:25:00"
    filtered_df = temp_data_lake.load_bars("INFY", "5m", start_time=start_t, end_time=end_t)
    assert len(filtered_df) == 3
