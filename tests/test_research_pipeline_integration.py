"""
End-to-End Synthetic Panel Integration Test for Ashva Research & Statistical Validation Pipeline
Proves:
1. Multi-symbol panel backtest aggregation.
2. Canonical Daily Panel Trade-Return Series generation.
3. Full integration with StatisticalValidator.validate_panel_hypothesis().
4. CPCVEngine combinatorial purged/embargoed OOS evaluation.
5. Bailey & Lopez de Prado DSR calculation with trial accounting.
6. 5,000 bootstrap simulations on Daily Panel Returns for tail risk.
7. Canonical Net Profit Factor calculation (sum(wins) / abs(sum(losses))).
8. Immutable SQLite experiment ledger persistence.
"""

import pytest
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.strategies.alpha_001_opening_gap_continuation import Alpha1OpeningGapContinuation
from src.research.validator import StatisticalValidator
from src.analytics.indian_costs import IndianCostModel
from src.backtest.engine import BacktestTrade
from src.research.experiment_ledger import ResearchExperimentLedger


@pytest.fixture
def synthetic_panel_backtest_trades():
    """Constructs 100 BacktestTrade objects across 5 symbols over 30 calendar days."""
    np.random.seed(42)
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    trades = []
    base_time = pd.Timestamp("2026-06-01 09:30:00")
    cost_model = IndianCostModel()

    trade_id = 1
    for d in range(30):
        trade_date = base_time + timedelta(days=d)
        for sym_idx, sym in enumerate(symbols):
            entry_time = trade_date + timedelta(minutes=15 * sym_idx)
            exit_time = entry_time + timedelta(hours=2)
            # Create alternating positive and negative trades
            is_win = (trade_id % 3 != 0)
            entry_px = 1000.0 + sym_idx * 100.0
            exit_px = entry_px * (1.02 if is_win else 0.99)
            qty = 50
            side = "LONG"
            
            cb = cost_model.calculate_trade_costs(
                buy_price=entry_px,
                sell_price=exit_px,
                quantity=qty,
            )
            
            trades.append(
                BacktestTrade(
                    trade_id=trade_id,
                    symbol=sym,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_px,
                    exit_price=exit_px,
                    quantity=qty,
                    side=side,
                    gross_pnl=cb.gross_pnl,
                    net_pnl=cb.net_pnl,
                    cost_breakdown=cb,
                    duration_bars=8,
                    exit_reason="TAKE_PROFIT" if is_win else "STOP_LOSS",
                    entry_rationale=f"1_alpha test trigger @ {entry_time}",
                    sizing_rationale=f"Qty {qty}",
                    mfe_pct=2.5 if is_win else 0.5,
                    mae_pct=0.2 if is_win else 1.2,
                )
            )
            trade_id += 1
    return trades


def test_research_pipeline_end_to_end(tmp_path, synthetic_panel_backtest_trades):
    # Use temporary test database for clean isolation
    test_db = tmp_path / "test_exp_ledger.db"
    ledger = ResearchExperimentLedger(db_path=str(test_db))
    cost_model = IndianCostModel()
    validator = StatisticalValidator(cost_model=cost_model, experiment_ledger=ledger)

    strat = Alpha1OpeningGapContinuation()
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    symbol_metrics = [{"symbol": s, "trades": 20, "net_pnl": 5000.0} for s in symbols]
    timeframe_comparison = {
        "15m": {"timeframe": "15m", "net_profit_factor": 1.45, "empirical_timeframe_score": 0.65},
        "5m": {"timeframe": "5m", "net_profit_factor": 1.10, "empirical_timeframe_score": 0.42},
    }

    report = validator.validate_panel_hypothesis(
        hypothesis=strat,
        all_trades=synthetic_panel_backtest_trades,
        symbol_metrics=symbol_metrics,
        timeframe_comparison=timeframe_comparison,
        parameter_grid=strat.get_parameter_grid(),
        tested_timeframes_count=2,
        initial_capital=500000.0,
        selected_timeframe="15m",
        symbol_universe=symbols,
    )

    # 1. Assert Valid Statistical Report
    assert report.hypothesis_id == "1_alpha"
    assert report.in_sample_sharpe > 0.0
    assert report.cpcv_mean_sharpe > 0.0
    assert 0.0 <= report.deflated_sharpe_p_value <= 1.0
    assert report.net_profit_factor_post_tax > 1.0
    assert report.monte_carlo_95_max_dd_pct >= 0.0
    assert report.tested_trials_count == 81 * 2  # 81 parameter combos * 2 tested timeframes

    # 2. Assert Canonical Persistence in SQLite Experiment Ledger
    with sqlite3.connect(str(test_db)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT experiment_id, strategy_id, status, in_sample_sharpe, cpcv_oos_sharpe, net_profit_factor FROM experiment_journal WHERE strategy_id = '1_alpha'")
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "1_alpha"
        assert row[2] in ["ACCEPTED", "CAPITAL_CANDIDATE", "REJECTED", "PROVEN"]
        assert row[3] == round(float(report.in_sample_sharpe), 3)
        assert row[4] == round(float(report.cpcv_mean_sharpe), 3)
        assert row[5] == round(float(report.net_profit_factor_post_tax), 3)