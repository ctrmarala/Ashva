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
    res = engine.run(df, symbol="RELIANCE", strategy_id="TEST_STRAT")

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
