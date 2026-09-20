"""
Unit Tests for Unified Multi-Alpha Shared-Capital Portfolio Engine
"""

import pytest
import pandas as pd
from datetime import datetime
from src.analytics.unified_portfolio_engine import UnifiedPortfolioEngine
from src.analytics.indian_costs import IndianCostModel, Segment


def test_unified_portfolio_engine_concurrency_limit():
    engine = UnifiedPortfolioEngine(
        initial_capital=500000.0,
        max_concurrent_positions=2,  # Max 2 concurrent positions
        capital_per_trade_pct=0.50,  # 250k each
    )

    candidate_trades = [
        {
            "alpha_id": "alpha_101",
            "symbol": "RELIANCE",
            "side": "BUY",
            "entry_time": "2026-05-01 09:30:00",
            "exit_time": "2026-05-01 10:30:00",
            "entry_price": 2500.0,
            "exit_price": 2550.0,
            "exit_reason": "TARGET",
        },
        {
            "alpha_id": "alpha_102",
            "symbol": "TCS",
            "side": "BUY",
            "entry_time": "2026-05-01 09:35:00",
            "exit_time": "2026-05-01 11:00:00",
            "entry_price": 3500.0,
            "exit_price": 3570.0,
            "exit_reason": "TARGET",
        },
        {
            "alpha_id": "alpha_103",
            "symbol": "INFY",
            "side": "BUY",
            "entry_time": "2026-05-01 09:40:00",  # Both slots full at 09:40!
            "exit_time": "2026-05-01 10:00:00",
            "entry_price": 1500.0,
            "exit_price": 1530.0,
            "exit_reason": "TARGET",
        },
    ]

    res = engine.run_portfolio_simulation(candidate_trades)
    assert res["total_candidates"] == 3
    assert res["total_executed_trades"] == 2
    assert res["total_rejected_signals"] == 1
    assert res["rejected_signals"][0].reason == "CONCURRENCY_LIMIT"
    assert res["rejected_signals"][0].alpha_id == "alpha_103"
    assert res["net_pnl"] > 0


def test_unified_portfolio_consecutive_loss_lockout():
    engine = UnifiedPortfolioEngine(
        initial_capital=500000.0,
        max_concurrent_positions=4,
        capital_per_trade_pct=0.25,
        max_consecutive_losses=1,  # 1 loss locks out the strategy
    )

    candidate_trades = [
        {
            "alpha_id": "alpha_loser",
            "symbol": "RELIANCE",
            "side": "BUY",
            "entry_time": "2026-05-01 09:30:00",
            "exit_time": "2026-05-01 10:00:00",
            "entry_price": 2500.0,
            "exit_price": 2400.0,  # Loss
            "exit_reason": "STOP_LOSS",
        },
        {
            "alpha_id": "alpha_loser",
            "symbol": "TCS",
            "side": "BUY",
            "entry_time": "2026-05-01 10:30:00",  # Triggers after the 1st loss
            "exit_time": "2026-05-01 11:30:00",
            "entry_price": 3500.0,
            "exit_price": 3600.0,
            "exit_reason": "TARGET",
        },
    ]

    res = engine.run_portfolio_simulation(candidate_trades)
    assert res["total_executed_trades"] == 1
    assert res["total_rejected_signals"] == 1
    assert res["rejected_signals"][0].reason == "CONSECUTIVE_LOSS_LOCKOUT"


def test_unified_portfolio_daily_limits():
    engine = UnifiedPortfolioEngine(
        initial_capital=500000.0,
        max_concurrent_positions=4,
        max_daily_trades_per_strategy=1,
    )

    candidate_trades = [
        {
            "alpha_id": "alpha_active",
            "symbol": "RELIANCE",
            "side": "BUY",
            "entry_time": "2026-05-01 09:30:00",
            "exit_time": "2026-05-01 10:00:00",
            "entry_price": 2500.0,
            "exit_price": 2550.0,
            "exit_reason": "TARGET",
        },
        {
            "alpha_id": "alpha_active",
            "symbol": "TCS",
            "side": "BUY",
            "entry_time": "2026-05-01 11:00:00",  # Same day trade 2 -> rejected
            "exit_time": "2026-05-01 12:00:00",
            "entry_price": 3500.0,
            "exit_price": 3550.0,
            "exit_reason": "TARGET",
        },
        {
            "alpha_id": "alpha_active",
            "symbol": "INFY",
            "side": "BUY",
            "entry_time": "2026-05-02 09:30:00",  # Next day trade 1 -> accepted
            "exit_time": "2026-05-02 10:00:00",
            "entry_price": 1500.0,
            "exit_price": 1550.0,
            "exit_reason": "TARGET",
        },
    ]

    res = engine.run_portfolio_simulation(candidate_trades)
    assert res["total_executed_trades"] == 2
    assert res["total_rejected_signals"] == 1
    assert res["rejected_signals"][0].reason == "STRATEGY_DAILY_LIMIT"
