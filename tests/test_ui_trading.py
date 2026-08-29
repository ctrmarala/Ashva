"""
Unit tests for Tab 3 Trading Observability in Ashva UI.
Verifies Paper, Replay, and Live trading state retrieval, capital allocation,
signals, orders, fills, trade ledger, zero-signal diagnostics, and event trace drill-down.
"""

import pytest
import pandas as pd
import sqlite3
from src.ui.data_access import UIDataAccess


@pytest.fixture
def mock_trading_dal(tmp_path):
    exp_db = tmp_path / "exp.db"
    trd_db = tmp_path / "trading_ledger.db"
    duck_db = tmp_path / "ashva_market_data.duckdb"
    
    with sqlite3.connect(trd_db) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                alpha_id TEXT NOT NULL,
                alpha_version TEXT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence DOUBLE NOT NULL,
                suggested_stop_loss DOUBLE,
                suggested_take_profit DOUBLE,
                stop_dist DOUBLE,
                metadata_json TEXT,
                persisted_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT UNIQUE NOT NULL,
                signal_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                alpha_id TEXT NOT NULL,
                alpha_version TEXT,
                symbol TEXT NOT NULL,
                is_accepted INTEGER NOT NULL,
                allocated_quantity INTEGER NOT NULL,
                risk_budget DOUBLE NOT NULL,
                rejection_reason TEXT,
                competing_alphas_json TEXT,
                persisted_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                intent_id TEXT,
                decision_id TEXT,
                signal_id TEXT,
                alpha_id TEXT NOT NULL,
                alpha_version TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                status TEXT NOT NULL,
                limit_price DOUBLE,
                stop_price DOUBLE,
                product_type TEXT NOT NULL,
                is_reduce_only INTEGER,
                reject_reason TEXT,
                broker_order_id TEXT,
                mode TEXT NOT NULL,
                tag TEXT,
                created_at TEXT NOT NULL,
                broker_ack_at TEXT,
                persisted_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fills_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fill_id TEXT UNIQUE NOT NULL,
                order_id TEXT NOT NULL,
                decision_id TEXT,
                signal_id TEXT,
                alpha_id TEXT NOT NULL,
                alpha_version TEXT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                fill_price DOUBLE NOT NULL,
                quantity INTEGER NOT NULL,
                commission DOUBLE,
                slippage DOUBLE,
                latency_ms DOUBLE,
                is_stop_loss INTEGER,
                cost_breakdown_json TEXT,
                persisted_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_ledger (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alpha_id TEXT NOT NULL,
                alpha_version TEXT,
                signal_id TEXT,
                decision_id TEXT,
                order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                gross_pnl REAL,
                net_pnl REAL,
                total_costs REAL,
                mfe_pct REAL,
                mae_pct REAL,
                holding_period_bars INTEGER,
                exit_reason TEXT,
                cost_breakdown_json TEXT,
                mode TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equity REAL,
                cash REAL,
                unrealized_pnl REAL,
                open_positions_count INTEGER,
                mode TEXT
            )
        """)

        # Insert sample mock records
        conn.execute("INSERT INTO signals_log VALUES (1, 'sig_1', '2026-08-28 10:00:00', 'alpha_test', 'v1.0.0', 'INFY', 'BUY', 0.85, 1830.0, 1890.0, 20.0, '{}', '2026-08-28 10:00:00')")
        conn.execute("INSERT INTO decisions_log VALUES (1, 'dec_1', 'sig_1', '2026-08-28 10:00:01', 'alpha_test', 'v1.0.0', 'INFY', 1, 50, 2500.0, '', '[]', '2026-08-28 10:00:01')")
        conn.execute("INSERT INTO orders_log VALUES (1, 'ord_1', 'int_1', 'dec_1', 'sig_1', 'alpha_test', 'v1.0.0', 'INFY', 'BUY', 'LIMIT', 50, 'FILLED', 1850.0, 0.0, 'INTRADAY', 0, '', 'b_1', 'REPLAY', 'TAG', '2026-08-28 10:00:02', '2026-08-28 10:00:03', '2026-08-28 10:00:03')")
        conn.execute("INSERT INTO fills_log VALUES (1, 'fill_1', 'ord_1', 'dec_1', 'sig_1', 'alpha_test', 'v1.0.0', '2026-08-28 10:00:05', 'INFY', 'BUY', 1850.2, 50, 20.0, 10.0, 150.0, 0, '{}', '2026-08-28 10:00:05')")
        conn.execute("INSERT INTO trade_ledger VALUES (4, 'alpha_test', 'v1.0.0', 'sig_1', 'dec_1', 'ord_1', 'KOTAKBANK', 'BUY', 50, '2026-08-28 10:00:00', '2026-08-28 11:30:00', 1800.0, 1840.0, 2000.0, 1920.0, 80.0, 2.5, -0.4, 6, 'TARGET', '{\"brokerage\": 40.0, \"stt\": 25.0, \"turnover_charges\": 7.0, \"gst\": 8.0, \"total\": 80.0}', 'REPLAY')")
        conn.execute("INSERT INTO portfolio_snapshots VALUES (1, '2026-08-28 15:30:00', 501920.0, 501920.0, 0.0, 0, 'REPLAY')")

    return UIDataAccess(
        exp_db_path=str(exp_db),
        trd_db_path=str(trd_db),
        duckdb_path=str(duck_db),
    )


def test_trading_portfolio_summary_replay(mock_trading_dal):
    summary = mock_trading_dal.get_trading_portfolio_summary(mode="REPLAY")
    assert summary["mode"] == "REPLAY"
    assert summary["initial_capital"] == 500000.0
    assert summary["current_equity"] >= 500000.0
    assert summary["cash"] > 0
    assert "open_positions" in summary


def test_trading_portfolio_summary_paper(mock_trading_dal):
    summary = mock_trading_dal.get_trading_portfolio_summary(mode="PAPER")
    assert summary["mode"] == "PAPER"
    assert summary["initial_capital"] == 500000.0
    assert summary["current_equity"] == 500000.0
    assert summary["open_positions"] == 0


def test_active_trading_alphas(mock_trading_dal):
    active_alphas = mock_trading_dal.get_active_trading_alphas()
    assert isinstance(active_alphas, list)


def test_alpha_symbol_evaluation_matrix(mock_trading_dal):
    df_matrix = mock_trading_dal.get_alpha_symbol_evaluation_matrix()
    assert isinstance(df_matrix, pd.DataFrame)


def test_trading_signals_and_decisions(mock_trading_dal):
    df_sig = mock_trading_dal.get_trading_signals(mode="REPLAY", limit=50)
    assert not df_sig.empty
    expected_cols = ["timestamp", "signal_id", "alpha_id", "symbol", "direction", "confidence", "decision_status"]
    for col in expected_cols:
        assert col in df_sig.columns


def test_trading_orders_and_fills(mock_trading_dal):
    df_ord = mock_trading_dal.get_trading_orders(mode="REPLAY", limit=50)
    assert not df_ord.empty
    assert "order_id" in df_ord.columns
    assert "symbol" in df_ord.columns

    df_fills = mock_trading_dal.get_trading_fills(mode="REPLAY", limit=50)
    assert not df_fills.empty
    assert "fill_id" in df_fills.columns
    assert "fill_price" in df_fills.columns


def test_closed_trades_ledger(mock_trading_dal):
    df_trades = mock_trading_dal.get_closed_trades(mode="REPLAY", limit=50)
    assert not df_trades.empty
    assert "trade_id" in df_trades.columns
    assert "net_pnl" in df_trades.columns
    assert "gross_pnl" in df_trades.columns
    assert "exit_reason" in df_trades.columns


def test_replay_summary_and_breakdown(mock_trading_dal):
    summary = mock_trading_dal.get_replay_summary()
    assert summary["replay_status"] in ["COMPLETED", "READY"]
    assert summary["total_trades"] >= 1
    assert summary["net_pnl"] > 0
    assert summary["win_rate"] == 100.0

    df_alpha = mock_trading_dal.get_replay_alpha_breakdown()
    assert not df_alpha.empty
    assert "alpha_id" in df_alpha.columns
    assert "trades_count" in df_alpha.columns
    assert "total_net_pnl" in df_alpha.columns


def test_capital_allocation_breakdown(mock_trading_dal):
    alloc = mock_trading_dal.get_capital_allocation_breakdown(mode="PAPER")
    assert alloc["initial_capital"] == 500000.0
    assert alloc["max_risk_per_trade_pct"] == 0.0050
    assert isinstance(alloc["per_alpha_table"], list)


def test_event_trace_drill_down(mock_trading_dal):
    # Test trace for existing Trade #4 (KOTAKBANK BUY)
    trace = mock_trading_dal.get_event_trace("4")
    assert trace["trade_id"] == 4
    assert trace["symbol"] == "KOTAKBANK"
    assert trace["side"] == "BUY"
    assert trace["pnl_details"]["net_pnl"] > 0
    assert "signal_details" in trace
    assert "allocator_decision" in trace
    assert "order_details" in trace

    # Test trace for non-existent trade
    missing_trace = mock_trading_dal.get_event_trace("999999")
    assert missing_trace.get("status") == "NOT FOUND"
