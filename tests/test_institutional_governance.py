"""
Unit tests for Institutional Risk Governance, Broker Reconciliation,
Research Experiment Registry, and Slippage Stress Matrix.
"""

from datetime import datetime, time
import pandas as pd
import pytest

from src.risk.risk_manager import RiskManager
from src.core.events import OrderEvent, OrderSide, OrderType
from src.execution.reconciliation import BrokerReconciliationEngine
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord
from src.analytics.trade_explainability import TradeExplainabilityEngine, TradeExplanationRecord
from src.backtest.engine import BacktestEngine


class MockBrokerGateway:
    def __init__(self):
        self.positions = [{"symbol": "RELIANCE", "quantity": 100, "side": "LONG"}]
        self.orders = [{"order_id": "ORD_01", "status": "OPEN"}]
        self.placed_orders = []

    def get_positions(self):
        return self.positions

    def get_order_book(self):
        return self.orders

    def cancel_all_orders(self):
        self.orders = []

    def place_order(self, order):
        self.placed_orders.append(order)


def test_rms_allows_exits_when_max_positions_reached():
    """Verify that exit orders are NEVER blocked even when max open positions are reached."""
    rms = RiskManager(max_open_positions=4)
    order = OrderEvent(
        symbol="RELIANCE",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=100,
        strategy_id="TEST",
    )
    
    # 4 positions already open -> New entry would be rejected
    market_time = datetime(2026, 8, 17, 10, 0, 0)
    is_approved_entry, reason_entry = rms.validate_order(order, current_equity=500000.0, current_price=1300.0, open_positions_count=4, is_exit=False, current_time=market_time)
    assert not is_approved_entry
    assert "Max open positions limit" in reason_entry

    # But an EXIT order MUST be approved even past cutoffs or with full positions!
    is_approved_exit, reason_exit = rms.validate_order(order, current_equity=500000.0, current_price=1300.0, open_positions_count=4, is_exit=True, current_time=market_time)
    assert is_approved_exit
    assert reason_exit is None


def test_rms_active_kill_switch_liquidation():
    """Verify that kill switch flattens positions actively at the broker."""
    rms = RiskManager()
    broker = MockBrokerGateway()
    
    event = rms.trigger_kill_switch(broker_gateway=broker, reason="TEST_CIRCUIT_BREAKER")
    assert rms.kill_switch_active
    assert len(broker.placed_orders) == 1
    assert broker.placed_orders[0].side == OrderSide.SELL
    assert broker.placed_orders[0].quantity == 100
    assert broker.placed_orders[0].tag == "EMERGENCY_FLATTEN"


def test_broker_reconciliation_engine():
    """Verify detection of position and order discrepancies."""
    broker = MockBrokerGateway()
    reconciler = BrokerReconciliationEngine(broker_gateway=broker)
    
    # 1. Perfectly synced
    is_synced, msgs = reconciler.reconcile_positions({"RELIANCE": {"quantity": 100}})
    assert is_synced
    assert len(msgs) == 0

    # 2. Phantom position divergence
    is_synced_diverged, msgs_diverged = reconciler.reconcile_positions({"RELIANCE": {"quantity": 50}})
    assert not is_synced_diverged
    assert len(msgs_diverged) == 1
    assert "POSITION DIVERGENCE" in msgs_diverged[0]


def test_experiment_ledger_trial_counting(tmp_path):
    """Verify cumulative trial count increments properly for multiple testing."""
    db_file = str(tmp_path / "test_exp_ledger.db")
    ledger = ResearchExperimentLedger(db_path=db_file)
    
    rec1 = ExperimentRecord(
        experiment_id="EXP_001",
        strategy_id="ORB",
        symbol_universe="NIFTY_TOP5",
        timeframe="15m",
        parameters_json="{}",
        in_sample_sharpe=1.8,
        cpcv_oos_sharpe=1.4,
        deflated_sharpe_p_value=0.01,
        net_profit_factor=1.45,
        monte_carlo_95_max_dd=8.5,
        total_trials_cumulative=1,
        git_commit_sha="2393f38",
        status="ACCEPTED",
        rejection_reasons_json="[]",
    )
    n1 = ledger.log_experiment(rec1)
    assert n1 == 1

    rec2 = ExperimentRecord(
        experiment_id="EXP_002",
        strategy_id="VOL_SQUEEZE",
        symbol_universe="NIFTY_TOP5",
        timeframe="15m",
        parameters_json="{}",
        in_sample_sharpe=1.2,
        cpcv_oos_sharpe=0.4,
        deflated_sharpe_p_value=0.12,
        net_profit_factor=0.95,
        monte_carlo_95_max_dd=16.5,
        total_trials_cumulative=2,
        git_commit_sha="2393f38",
        status="REJECTED",
        rejection_reasons_json="[\"DSR Failed\"]",
    )
    n2 = ledger.log_experiment(rec2)
    assert n2 == 2


def test_slippage_stress_matrix():
    """Verify that BacktestEngine produces a 5-tier slippage matrix."""
    engine = BacktestEngine(initial_capital=500000.0)
    df = pd.DataFrame({
        "close": [100.0, 102.0, 105.0, 103.0, 108.0],
        "signal": [0, 1, 1, 0, 0],
    }, index=pd.date_range("2026-08-17 09:15", periods=5, freq="15min"))

    matrix_df = engine.run_slippage_stress_matrix(df, symbol="TEST_SYM")
    assert len(matrix_df) == 5
    assert list(matrix_df["Scenario"]) == ["Optimistic", "Base", "Conservative", "Stress", "Extreme"]
    assert matrix_df.iloc[0]["Slippage_Bps"] == 1.0
    assert matrix_df.iloc[4]["Slippage_Bps"] == 20.0
