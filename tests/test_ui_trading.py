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
def dal_instance():
    return UIDataAccess()


def test_trading_portfolio_summary_replay(dal_instance):
    summary = dal_instance.get_trading_portfolio_summary(mode="REPLAY")
    assert summary["mode"] == "REPLAY"
    assert summary["initial_capital"] == 500000.0
    assert summary["current_equity"] >= 500000.0
    assert summary["cash"] > 0
    assert "open_positions" in summary


def test_trading_portfolio_summary_paper(dal_instance):
    summary = dal_instance.get_trading_portfolio_summary(mode="PAPER")
    assert summary["mode"] == "PAPER"
    assert summary["initial_capital"] == 500000.0
    assert summary["current_equity"] == 500000.0
    assert summary["open_positions"] == 0


def test_active_trading_alphas(dal_instance):
    active_alphas = dal_instance.get_active_trading_alphas()
    assert len(active_alphas) > 0
    for a in active_alphas:
        assert a["factory_status"] == "PROVEN"
        assert a["trading_status"] == "ACTIVE"
        assert "alpha_id" in a
        assert "universe" in a


def test_alpha_symbol_evaluation_matrix(dal_instance):
    df_matrix = dal_instance.get_alpha_symbol_evaluation_matrix()
    assert not df_matrix.empty
    assert "Alpha ID" in df_matrix.columns
    assert "INFY" in df_matrix.columns
    assert "TCS" in df_matrix.columns
    assert "RELIANCE" in df_matrix.columns


def test_trading_signals_and_decisions(dal_instance):
    df_sig = dal_instance.get_trading_signals(mode="REPLAY", limit=50)
    assert not df_sig.empty
    expected_cols = ["timestamp", "signal_id", "alpha_id", "symbol", "direction", "confidence", "decision_status"]
    for col in expected_cols:
        assert col in df_sig.columns


def test_trading_orders_and_fills(dal_instance):
    df_ord = dal_instance.get_trading_orders(mode="REPLAY", limit=50)
    assert not df_ord.empty
    assert "order_id" in df_ord.columns
    assert "symbol" in df_ord.columns

    df_fills = dal_instance.get_trading_fills(mode="REPLAY", limit=50)
    assert not df_fills.empty
    assert "fill_id" in df_fills.columns
    assert "fill_price" in df_fills.columns


def test_closed_trades_ledger(dal_instance):
    df_trades = dal_instance.get_closed_trades(mode="REPLAY", limit=50)
    assert not df_trades.empty
    assert len(df_trades) >= 5
    assert "trade_id" in df_trades.columns
    assert "net_pnl" in df_trades.columns
    assert "gross_pnl" in df_trades.columns
    assert "exit_reason" in df_trades.columns


def test_replay_summary_and_breakdown(dal_instance):
    summary = dal_instance.get_replay_summary()
    assert summary["replay_status"] in ["COMPLETED", "READY"]
    assert summary["total_trades"] >= 5
    assert summary["net_pnl"] > 0
    assert summary["win_rate"] == 100.0

    df_alpha = dal_instance.get_replay_alpha_breakdown()
    assert not df_alpha.empty
    assert "alpha_id" in df_alpha.columns
    assert "trades_count" in df_alpha.columns
    assert "total_net_pnl" in df_alpha.columns


def test_capital_allocation_breakdown(dal_instance):
    alloc = dal_instance.get_capital_allocation_breakdown(mode="PAPER")
    assert alloc["initial_capital"] == 500000.0
    assert alloc["max_risk_per_trade_pct"] == 0.0050
    assert len(alloc["per_alpha_table"]) > 0


def test_event_trace_drill_down(dal_instance):
    # Test trace for existing Trade #4 (KOTAKBANK BUY)
    trace = dal_instance.get_event_trace("4")
    assert trace["trade_id"] == 4
    assert trace["symbol"] == "KOTAKBANK"
    assert trace["side"] == "BUY"
    assert trace["pnl_details"]["net_pnl"] > 0
    assert "signal_details" in trace
    assert "allocator_decision" in trace
    assert "order_details" in trace

    # Test trace for non-existent trade
    missing_trace = dal_instance.get_event_trace("999999")
    assert missing_trace.get("status") == "NOT FOUND"
