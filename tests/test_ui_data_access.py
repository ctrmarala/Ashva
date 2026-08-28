"""
Unit tests for Ashva Observability Data Access Layer (UIDataAccess).
Tests Data Lake queries, coverage matrices, symbol detail inspection, data quality audits, and missing DB handling.
"""

import pytest
import pandas as pd
import sqlite3
import duckdb
from pathlib import Path
from src.ui.data_access import UIDataAccess


@pytest.fixture
def mock_dbs(tmp_path):
    exp_db = tmp_path / "exp.db"
    trd_db = tmp_path / "trd.db"
    duck_db = tmp_path / "market_data.duckdb"
    parquet_dir = tmp_path / "parquet"
    logs_dir = tmp_path / "logs"
    
    parquet_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Mock SQLite Experiment Ledger
    with sqlite3.connect(exp_db) as conn:
        conn.execute("""
            CREATE TABLE experiments (
                strategy_id TEXT,
                status TEXT,
                in_sample_sharpe REAL,
                cpcv_oos_sharpe REAL,
                deflated_sharpe_p_value REAL,
                net_profit_factor REAL,
                monte_carlo_95_max_dd REAL,
                trials_in_experiment INTEGER,
                timestamp TEXT
            )
        """)
        conn.execute("INSERT INTO experiments VALUES ('alpha_01', 'CAPITAL_CANDIDATE', 1.5, 1.2, 0.01, 1.8, -10.5, 100, '2026-08-28T00:00:00')")

    # 2. Mock DuckDB Data Lake
    with duckdb.connect(str(duck_db)) as conn:
        conn.execute("""
            CREATE TABLE ohlcv_bars (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                timeframe VARCHAR NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                source VARCHAR DEFAULT 'HISTORICAL',
                PRIMARY KEY (symbol, timestamp, timeframe)
            )
        """)
        # Insert 15m and 1d bars spanning > 540 days for INFY
        conn.execute("""
            INSERT INTO ohlcv_bars VALUES 
            ('INFY', TIMESTAMP '2025-01-01 09:15:00', '15m', 1500.0, 1510.0, 1495.0, 1505.0, 10000, 'HISTORICAL'),
            ('INFY', TIMESTAMP '2026-08-28 15:15:00', '15m', 1600.0, 1610.0, 1595.0, 1605.0, 20000, 'HISTORICAL'),
            ('INFY', TIMESTAMP '2026-08-28 00:00:00', '1d', 1600.0, 1610.0, 1595.0, 1605.0, 50000, 'HISTORICAL'),
            ('TCS', TIMESTAMP '2026-08-28 09:15:00', '15m', 3000.0, 3010.0, 2990.0, 3005.0, 5000, 'HISTORICAL')
        """)

    # 3. Mock Log file
    day_log = logs_dir / "2026-08-28"
    day_log.mkdir(parents=True, exist_ok=True)
    (day_log / "app.log").write_text("INFO - Ingestion completed.\n", encoding="utf-8")

    dal = UIDataAccess(
        exp_db_path=str(exp_db),
        trd_db_path=str(trd_db),
        duckdb_path=str(duck_db),
        parquet_dir=str(parquet_dir),
        logs_dir=str(logs_dir),
    )
    return dal


def test_data_overview(mock_dbs):
    dal = mock_dbs
    ov = dal.get_data_overview()
    assert ov["total_symbols"] == 2
    assert ov["total_bars"] == 4
    assert "15m" in ov["available_timeframes"]
    assert "1d" in ov["available_timeframes"]
    assert ov["earliest_timestamp"] == "2025-01-01 09:15:00"
    assert ov["latest_timestamp"] == "2026-08-28 15:15:00"


def test_coverage_matrix(mock_dbs):
    dal = mock_dbs
    df_matrix = dal.get_coverage_matrix()
    assert not df_matrix.empty
    assert "symbol" in df_matrix.columns
    assert "15m" in df_matrix.columns
    assert "1d" in df_matrix.columns
    assert "540d_Horizon" in df_matrix.columns
    
    infy_row = df_matrix[df_matrix["symbol"] == "INFY"].iloc[0]
    assert infy_row["15m"] == 2
    assert "PASS" in infy_row["540d_Horizon"]


def test_symbol_detail(mock_dbs):
    dal = mock_dbs
    detail = dal.get_symbol_detail("INFY")
    assert detail["symbol"] == "INFY"
    assert len(detail["timeframes_detail"]) == 2
    
    q_metrics = detail["quality_metrics"]
    assert q_metrics["duplicate_bars"] == 0
    assert q_metrics["invalid_ohlc_bars"] == 0
    assert q_metrics["out_of_market_hours_bars"] == 0


def test_data_quality_summary(mock_dbs):
    dal = mock_dbs
    q = dal.get_data_quality_summary()
    assert q["total_bars_audited"] == 4
    assert q["duplicate_bars"] == 0
    assert q["invalid_ohlc_bars"] == 0
    assert q["out_of_hours_intraday_bars"] == 0
    assert q["symbols_with_540d_coverage"] == 1
    assert q["quality_status"] == "CLEAN & QUALIFIED"


def test_ingestion_log_summary(mock_dbs):
    dal = mock_dbs
    df_logs = dal.get_ingestion_log_summary()
    assert not df_logs.empty
    assert "2026-08-28" in df_logs["Log Session"].values


def test_alpha_data_connection(mock_dbs):
    dal = mock_dbs
    df_conn = dal.get_alpha_data_connection()
    assert not df_conn.empty
    assert "Alpha ID" in df_conn.columns
    assert "Required Timeframe" in df_conn.columns
    assert "Data Lake Availability" in df_conn.columns


def test_missing_data_lake(tmp_path):
    dal = UIDataAccess(
        exp_db_path=str(tmp_path / "missing.db"),
        trd_db_path=str(tmp_path / "missing.db"),
        duckdb_path=str(tmp_path / "missing.duckdb"),
        parquet_dir=str(tmp_path / "missing_p/"),
        logs_dir=str(tmp_path / "missing_logs/"),
    )
    ov = dal.get_data_overview()
    assert ov["total_symbols"] == 0
    assert ov["total_bars"] == 0
    
    df_matrix = dal.get_coverage_matrix()
    assert df_matrix.empty
    
    detail = dal.get_symbol_detail("INFY")
    assert detail == {}
    
    q = dal.get_data_quality_summary()
    assert q["quality_status"] == "NO DATA"
