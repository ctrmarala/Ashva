"""
Unit Tests for Institutional Backtest Engine
"""

import numpy as np
import pandas as pd
import pytest
from src.backtest.engine import BacktestEngine
from src.analytics.indian_costs import Segment


def test_backtest_engine_execution():
    dates = pd.date_range("2026-01-01 09:15", periods=30, freq="5min")
    prices = [2500.0 + i * 5.0 for i in range(30)]  # Uptrend from 2500 to 2645
    
    # Enter at bar 2, exit at bar 10
    signals = np.zeros(30)
    signals[2:10] = 1.0

    df = pd.DataFrame({
        "open": prices,
        "high": [p + 2.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
        "signal": signals,
    }, index=dates)

    engine = BacktestEngine(initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
    res = engine.run(df, symbol="TEST_STRAT_SYM", strategy_id="TEST_STRAT")

    assert res.total_trades == 1
    assert res.winning_trades == 1
    assert res.total_net_pnl > 0
    assert res.final_equity > res.initial_capital
    assert res.total_brokerage_paid > 0
    assert res.total_stt_paid > 0
    assert len(res.trade_list) == 1
    
    trade = res.trade_list[0]
    assert trade.side == "LONG"
    # Strict next-bar execution: Signal at Bar 2 Close -> Entry filled at Bar 3 Open (prices[3])
    assert trade.entry_price == prices[3]
    # Strict next-bar exit: Signal returned to 0 at Bar 10 Close -> Exit filled at Bar 11 Open (prices[11])
    assert trade.exit_price == prices[11]


def test_risk_based_position_sizing():
    """Verify that position size scales inversely with stop distance under risk-based sizing."""
    dates = pd.date_range("2026-01-01 09:15", periods=10, freq="5min")
    prices = [1000.0] * 10
    signals = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # 0.5% of Rs 500,000 = Rs 2,500 risk budget
    # Stop distance = Rs 25 (1000 - 975) -> Target Qty = 2500 / 25 = 100 shares
    stop_losses = [975.0] * 10
    take_profits = [1050.0] * 10

    df = pd.DataFrame({
        "open": prices, "high": [p + 5.0 for p in prices], "low": [p - 5.0 for p in prices], "close": prices,
        "signal": signals, "stop_loss": stop_losses, "take_profit": take_profits
    }, index=dates)

    engine = BacktestEngine(initial_capital=500000.0)
    res = engine.run(df, symbol="TEST_STOCK", risk_per_trade_pct=0.005)

    assert len(res.trade_list) == 1
    # Rs 2,500 risk / Rs 25 stop distance = exactly 100 shares
    assert res.trade_list[0].quantity == 100
