"""
Ashva Async Trading Ledger Test Suite
Tests non-blocking event queueing, batch persistence, and trade drilldown lineage queries.
"""

from datetime import datetime
import pytest

from src.core.events import (
    SignalEvent, SignalType, DecisionEvent, OrderEvent, FillEvent,
    OrderSide, OrderType, OrderStatus, ProductType,
)
from src.trading.ledger import TradingLedger


def test_async_ledger_persistence_and_drilldown(tmp_path):
    """Verify asynchronous event queueing and relational trade drilldown queries."""
    db_file = str(tmp_path / "test_ledger.db")
    ledger = TradingLedger(db_path=db_file)

    now = datetime(2026, 7, 1, 9, 30)

    # 1. Log Signal
    sig = SignalEvent(
        symbol="INFY",
        timestamp=now,
        strategy_id="ALPHA_67",
        alpha_version="1.0.0",
        signal_type=SignalType.LONG,
        confidence=0.88,
        suggested_stop_loss=1480.0,
        suggested_take_profit=1550.0,
        signal_id="SIG_TEST_001",
    )
    ledger.log_signal(sig)

    # 2. Log Decision
    dec = DecisionEvent(
        decision_id="DEC_TEST_001",
        signal_id="SIG_TEST_001",
        timestamp=now,
        alpha_id="ALPHA_67",
        alpha_version="1.0.0",
        symbol="INFY",
        is_accepted=True,
        allocated_quantity=100,
        risk_budget=2000.0,
        competing_alphas=["ALPHA_67", "ALPHA_52"],
    )
    ledger.log_decision(dec)

    # 3. Log Order
    ord = OrderEvent(
        symbol="INFY",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        order_id="ORD_TEST_001",
        intent_id="INT_TEST_001",
        decision_id="DEC_TEST_001",
        signal_id="SIG_TEST_001",
        strategy_id="ALPHA_67",
        alpha_version="1.0.0",
        status=OrderStatus.FILLED,
        timestamp=now,
    )
    ledger.log_order(ord)

    # 4. Log Fill
    fill = FillEvent(
        order_id="ORD_TEST_001",
        decision_id="DEC_TEST_001",
        signal_id="SIG_TEST_001",
        strategy_id="ALPHA_67",
        alpha_version="1.0.0",
        symbol="INFY",
        timestamp=now,
        side=OrderSide.BUY,
        fill_price=1500.0,
        quantity=100,
        commission=20.0,
        slippage=0.5,
        latency_ms=12.0,
    )
    ledger.log_fill(fill)

    # 5. Log Closed Trade
    trade = {
        "strategy_id": "ALPHA_67",
        "alpha_version": "1.0.0",
        "signal_id": "SIG_TEST_001",
        "decision_id": "DEC_TEST_001",
        "order_id": "ORD_TEST_001",
        "symbol": "INFY",
        "side": "BUY",
        "quantity": 100,
        "entry_time": now,
        "exit_time": datetime(2026, 7, 1, 15, 15),
        "entry_price": 1500.0,
        "exit_price": 1530.0,
        "gross_pnl": 3000.0,
        "net_pnl": 2850.0,
        "slippage_paid": 50.0,
        "total_costs": 150.0,
        "mfe": 4000.0,
        "mae": -500.0,
        "mfe_pct": 2.67,
        "mae_pct": -0.33,
        "holding_period_bars": 23,
        "exit_reason": "EOD_SQUAREOFF",
        "cost_breakdown": {"brokerage": 40.0, "stt": 50.0},
        "mode": "REPLAY",
    }
    ledger.log_closed_trade(trade)

    # Flush queue to SQLite
    ledger.flush()

    # Query trades
    trades = ledger.get_trades(alpha_id="ALPHA_67")
    assert len(trades) == 1
    assert trades[0]["net_pnl"] == 2850.0

    # Query full trade drilldown lineage
    drilldown = ledger.get_trade_drilldown(trade_id=1)
    assert drilldown is not None
    assert drilldown["trade"]["alpha_id"] == "ALPHA_67"
    assert drilldown["decision"]["is_accepted"] == 1
    assert drilldown["signal"]["confidence"] == 0.88
    assert drilldown["order"]["order_id"] == "ORD_TEST_001"

    ledger.shutdown()
