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

from src.strategies.registry import get_all_strategies
STRATEGY_MAP = {k: v for k, v in get_all_strategies(reload=True).items() if not k.startswith("Alpha")}
from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.research.alpha_linter import AlphaLinter


@pytest.fixture(scope="module")
def sample_bars():
    lake = DataLake(read_only=True)
    df = lake.load_bars("RELIANCE", "15m", max_lookback_days=180)
    if df.empty:
        pytest.skip("RELIANCE 15m bars not available in DataLake")
    return df


strategy_items = list(STRATEGY_MAP.items())

if not strategy_items:
    def test_strategy_lifecycle_contract_empty():
        # Clean baseline state: no active strategies loaded
        assert len(strategy_items) == 0
else:
    @pytest.mark.parametrize("strat_id,strat_tuple", strategy_items)
    def test_strategy_alpha_linter_compliance(strat_id, strat_tuple):
        """
        Validates that every strategy strictly satisfies AlphaLinter static & runtime rules:
        - No hardcoded tickers
        - Non-empty parameter grid
        - Metadata completeness
        - Zero lookahead perturbation invariance
        - Output column contracts (signal, stop_loss, take_profit)
        """
        strat_cls = strat_tuple[1] if isinstance(strat_tuple, tuple) else strat_tuple
        strat = strat_cls()
        violations = AlphaLinter.lint_strategy_instance(strat)
        assert len(violations) == 0, f"AlphaLinter violations in {strat_id}: {violations}"

    @pytest.mark.parametrize("strat_id,strat_tuple", strategy_items)
    def test_strategy_lifecycle_contract(strat_id, strat_tuple, sample_bars):
        """
        Validates that each strategy maintains active position state across intraday bars
        and does not inadvertently exit after 1 bar due to signal=0 pulse defects.
        """
        strat_cls = strat_tuple[1] if isinstance(strat_tuple, tuple) else strat_tuple
        strat_name = getattr(strat_cls, "__name__", strat_id)
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


def test_signal_timing_next_bar_open_fill():
    """
    Engine Verification Test: Proves that when a signal is emitted at the close of Bar t (e.g. 09:30),
    BacktestEngine enters the position strictly at Bar t+1 Open (09:45) with zero lookahead at Bar t.
    """
    timestamps = [
        pd.Timestamp("2026-08-28 09:15:00"),
        pd.Timestamp("2026-08-28 09:30:00"), # Signal fires at close of this bar
        pd.Timestamp("2026-08-28 09:45:00"), # Entry MUST fill at Open of this bar
        pd.Timestamp("2026-08-28 10:00:00"),
        pd.Timestamp("2026-08-28 15:15:00"),
    ]
    df = pd.DataFrame({
        "open": [100.0, 102.0, 108.0, 110.0, 112.0],
        "high": [103.0, 105.0, 110.0, 112.0, 114.0],
        "low": [99.0, 101.0, 107.0, 109.0, 110.0],
        "close": [102.0, 104.0, 109.0, 111.0, 113.0],
        "volume": [1000, 1000, 1000, 1000, 1000],
        "signal": [0.0, 1.0, 1.0, 1.0, 0.0], # Signal triggers on bar 1 (09:30)
        "stop_loss": [0.0, 95.0, 95.0, 95.0, 0.0],
        "take_profit": [0.0, 120.0, 120.0, 120.0, 0.0],
    }, index=pd.DatetimeIndex(timestamps))

    cost_model = IndianCostModel()
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
    res = engine.run(df, symbol="RELIANCE", strategy_id="test_timing")

    assert res.total_trades == 1
    trade = res.trade_list[0]
    # Signal emitted at 09:30 -> Fill must be 09:45 at Open price 108.0
    assert trade.entry_time == pd.Timestamp("2026-08-28 09:45:00")
    assert trade.entry_price == 108.0
