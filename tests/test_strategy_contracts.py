"""
Ashva Quantitative Strategy Contract & Lifecycle Test Suite
Prevents premature 1-bar signal-pulse exits and verifies that all alphas
strictly adhere to the BacktestEngine position lifecycle contract.

CI / DEV Gate: Blocks any alpha with a lifecycle defect from entering research matrices.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.run_hypothesis_lab import STRATEGY_MAP
from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine


@pytest.fixture(scope="module")
def sample_bars():
    lake = DataLake(read_only=True)
    df = lake.load_bars("RELIANCE", "15m", max_lookback_days=180)
    if df.empty:
        pytest.skip("RELIANCE 15m bars not available in DataLake")
    return df


@pytest.mark.parametrize("strat_id,strat_tuple", list(STRATEGY_MAP.items()))
def test_strategy_lifecycle_contract(strat_id, strat_tuple, sample_bars):
    """
    Validates that each strategy maintains active position state across intraday bars
    and does not inadvertently exit after 1 bar due to signal=0 pulse defects.
    """
    strat_name, strat_cls = strat_tuple
    strat = strat_cls()
    sig_df = strat.generate_signals(sample_bars)

    assert "signal" in sig_df.columns, f"{strat_id}: Missing 'signal' column in output DataFrame"
    assert "stop_loss" in sig_df.columns, f"{strat_id}: Missing 'stop_loss' column"
    assert "take_profit" in sig_df.columns, f"{strat_id}: Missing 'take_profit' column"

    cost_model = IndianCostModel()
    eng = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
    res = eng.run(sig_df, symbol="RELIANCE", strategy_id=strat_id, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)

    # If the strategy generated trades, assert that it does not suffer from 100% 1-bar pulse exits
    if res.total_trades > 0:
        one_bar_signal_exits = sum(
            1 for t in res.trade_list if t.duration_bars <= 1 and t.exit_reason == "SIGNAL"
        )
        pulse_exit_pct = (one_bar_signal_exits / res.total_trades) * 100.0

        # Hard Gate: No intraday strategy should have > 50% unintended 1-bar SIGNAL exits
        assert pulse_exit_pct < 50.0, (
            f"🚨 STRATEGY CONTRACT VIOLATION in {strat_id} ({strat_name}):\n"
            f"{pulse_exit_pct:.1f}% of trades ({one_bar_signal_exits}/{res.total_trades}) exited after exactly 1 bar on 'SIGNAL'.\n"
            f"The strategy is emitting 1-bar signal pulses instead of maintaining 'curr_state' across bars until SL/TP/EOD."
        )
