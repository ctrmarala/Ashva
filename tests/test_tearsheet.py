"""
Unit Tests for Quant Tearsheet Generator
"""

import os
from pathlib import Path
import pandas as pd
import pytest
from src.backtest.engine import BacktestResult
from src.analytics.tearsheet import QuantTearsheetGenerator


def test_tearsheet_html_generation():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    equity_curve = pd.Series([500000.0, 502000.0, 505000.0, 503000.0, 508000.0, 507000.0, 510000.0, 512000.0, 511000.0, 515000.0], index=dates)

    result = BacktestResult(
        symbol="RELIANCE",
        strategy_id="ALPHA_TEST",
        initial_capital=500000.0,
        final_equity=515000.0,
        total_net_pnl=15000.0,
        net_roi_pct=3.0,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate_pct=100.0,
        gross_profit_factor=20.0,
        net_profit_factor=15.0,
        sharpe_ratio=2.5,
        sortino_ratio=3.2,
        max_drawdown_pct=0.8,
        max_drawdown_duration_bars=2,
        total_brokerage_paid=40.0,
        total_stt_paid=250.0,
        total_taxes_paid=1000.0,
        equity_curve=equity_curve,
        trade_list=[],
    )

    gen = QuantTearsheetGenerator()
    path = gen.generate_html_tearsheet(result)

    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "ASHVA QUANTITATIVE TEARSHEET" in content
    assert "RELIANCE" in content
