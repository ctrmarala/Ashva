"""
Ashva Trading Engine & Replay Verification Test Suite
Tests OrderManager 10-state lifecycle, PositionManager continuous MTM & MFE/MAE,
LiveRiskManager hierarchical rules, MultiAlphaAllocator competing allocations, and end-to-end Replay parity.
"""

from datetime import datetime, time, timedelta
import pytest
import pandas as pd
import numpy as np

from src.core.events import (
    MarketEvent, SignalEvent, SignalType, OrderIntent,
    OrderEvent, FillEvent, OrderSide, OrderType, OrderStatus, ProductType,
)
from src.trading.contract import QualifiedAlphaContract
from src.trading.manifest import TradingManifest
from src.trading.allocator import MultiAlphaAllocator
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.portfolio_state import PortfolioState
from src.trading.live_rms import LiveRiskManager, SafetyState
from src.trading.ledger import TradingLedger
from src.execution.replay_adapter import ReplayExecutionAdapter
from src.market_data.provider import MarketDataProvider
from src.trading.engine import TradingEngine


def test_order_manager_lifecycle():
    """Verify 10-state lifecycle transitions in OrderManager."""
    om = OrderManager()
    intent = OrderIntent(
        strategy_id="ALPHA_TEST",
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=50,
        intent_id="INT_001",
    )
    order = OrderEvent(
        symbol="TCS",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50,
        order_id="ORD_001",
        intent_id="INT_001",
        strategy_id="ALPHA_TEST",
        status=OrderStatus.SUBMITTED,
    )
    om.on_order_submitted(intent, order)
    assert "ORD_001" in om.active_orders

    om.on_order_acknowledged("ORD_001", broker_order_id="BORD_999")
    assert om.active_orders["ORD_001"].status == OrderStatus.ACKNOWLEDGED
    assert om.active_orders["ORD_001"].broker_order_id == "BORD_999"

    fill = FillEvent(
        order_id="ORD_001",
        symbol="TCS",
        timestamp=datetime(2026, 7, 1, 9, 30),
        side=OrderSide.BUY,
        fill_price=3500.0,
        quantity=50,
        strategy_id="ALPHA_TEST",
    )
    om.on_fill(fill)
    assert "ORD_001" not in om.active_orders
    assert "ORD_001" in om.completed_orders
    assert om.completed_orders["ORD_001"].status == OrderStatus.FILLED


def test_position_manager_mfe_mae_and_closing():
    """Verify PositionManager continuous MFE/MAE and trade record on close."""
    pm = PositionManager()
    t0 = datetime(2026, 7, 1, 9, 30)

    # Entry fill
    fill_in = FillEvent(
        order_id="ORD_001",
        symbol="TCS",
        timestamp=t0,
        side=OrderSide.BUY,
        fill_price=3500.0,
        quantity=10,
        strategy_id="ALPHA_54",
        alpha_version="1.0.0",
        signal_id="SIG_001",
        decision_id="DEC_001",
    )
    res = pm.on_fill(fill_in)
    assert res is None
    pos = pm.get_position("TCS")
    assert pos is not None
    assert pos.quantity == 10

    # Bar 1: Price goes higher (Favorable)
    mkt1 = MarketEvent(
        symbol="TCS",
        timestamp=t0 + timedelta(minutes=15),
        timeframe="15m",
        open=3510.0,
        high=3550.0,
        low=3505.0,
        close=3540.0,
        volume=1000,
    )
    pm.on_market_event(mkt1)
    assert pos.mfe == 500.0  # (3550 - 3500) * 10
    assert pos.unrealized_pnl == 400.0

    # Bar 2: Price dips lower (Adverse)
    mkt2 = MarketEvent(
        symbol="TCS",
        timestamp=t0 + timedelta(minutes=30),
        timeframe="15m",
        open=3540.0,
        high=3545.0,
        low=3480.0,
        close=3490.0,
        volume=1000,
    )
    pm.on_market_event(mkt2)
    assert pos.mfe == 500.0
    assert pos.mae == -200.0  # (3480 - 3500) * 10

    # Exit fill
    fill_out = FillEvent(
        order_id="ORD_002",
        symbol="TCS",
        timestamp=t0 + timedelta(minutes=45),
        side=OrderSide.SELL,
        fill_price=3530.0,
        quantity=10,
        strategy_id="ALPHA_54",
    )
    closed = pm.on_fill(fill_out)
    assert closed is not None
    assert closed["gross_pnl"] == 300.0
    assert closed["mfe"] == 500.0
    assert closed["mae"] == -200.0
    assert closed["strategy_id"] == "ALPHA_54"
    assert pm.get_position("TCS") is None


def test_multi_alpha_allocator_competing_signals():
    """Verify MultiAlphaAllocator picks highest priority alpha and records decisions."""
    allocator = MultiAlphaAllocator()
    pm = PositionManager()
    portfolio = PortfolioState(initial_capital=500000.0)

    contract_a67 = QualifiedAlphaContract(
        alpha_id="A67_MOMENTUM",
        strategy_class=None,
        universe=["INFY"],
        priority_score=2.0,
        risk_per_trade_pct=0.01,
        stop_loss_pct=0.01,
    )
    contract_a52 = QualifiedAlphaContract(
        alpha_id="A52_REVERSION",
        strategy_class=None,
        universe=["INFY"],
        priority_score=1.0,
        risk_per_trade_pct=0.01,
        stop_loss_pct=0.01,
    )
    contracts_map = {"A67_MOMENTUM": contract_a67, "A52_REVERSION": contract_a52}

    now = datetime(2026, 7, 1, 9, 45)
    sig_a67 = SignalEvent(
        symbol="INFY",
        timestamp=now,
        strategy_id="A67_MOMENTUM",
        signal_type=SignalType.LONG,
        confidence=0.9,
    )
    sig_a52 = SignalEvent(
        symbol="INFY",
        timestamp=now,
        strategy_id="A52_REVERSION",
        signal_type=SignalType.LONG,
        confidence=0.8,
    )

    intents, decisions = allocator.allocate(
        candidate_signals=[sig_a67, sig_a52],
        contracts_map=contracts_map,
        current_prices={"INFY": 1500.0},
        position_manager=pm,
        portfolio_state=portfolio,
    )

    # Only A67 should generate an order intent
    assert len(intents) == 1
    assert intents[0].strategy_id == "A67_MOMENTUM"

    # Both candidates must have decision events recorded
    assert len(decisions) == 2
    a67_dec = next(d for d in decisions if d.alpha_id == "A67_MOMENTUM")
    a52_dec = next(d for d in decisions if d.alpha_id == "A52_REVERSION")

    assert a67_dec.is_accepted is True
    assert a52_dec.is_accepted is False
    assert "LOWER_PRIORITY_SCORE" in a52_dec.rejection_reason


def test_live_risk_manager_hierarchical_controls():
    """Verify LiveRiskManager hierarchical kill-switches (global, alpha, symbol)."""
    rms = LiveRiskManager()
    pm = PositionManager()
    portfolio = PortfolioState(initial_capital=500000.0)

    now = datetime(2026, 7, 1, 10, 0)
    intent_infy = OrderIntent(strategy_id="A67", symbol="INFY", side=OrderSide.BUY, quantity=10, timestamp=now)

    # 1. Normal state -> approved
    approved, reason = rms.validate_order(intent_infy, 1500.0, pm, portfolio)
    assert approved is True

    # 2. Disable symbol INFY
    rms.disable_symbol("INFY", reason="Manual Quarantine")
    approved, reason = rms.validate_order(intent_infy, 1500.0, pm, portfolio)
    assert approved is False
    assert "quarantined/disabled" in reason

    # 3. Re-enable symbol, disable Alpha A67
    rms.enable_symbol("INFY")
    rms.disable_alpha("A67", reason="Alpha Underperformance")
    approved, reason = rms.validate_order(intent_infy, 1500.0, pm, portfolio)
    assert approved is False
    assert "disabled by risk policy" in reason

    # 4. Global Emergency Kill Switch
    rms.enable_alpha("A67")
    rms.trigger_kill_switch("Circuit Breaker")
    assert rms.kill_switch_active is True
    approved, reason = rms.validate_order(intent_infy, 1500.0, pm, portfolio)
    assert approved is False
    assert "HALTED" in reason

    # 5. Exit orders must NEVER be blocked even in emergency HALT
    exit_intent = OrderIntent(
        strategy_id="A67",
        symbol="INFY",
        side=OrderSide.SELL,
        quantity=10,
        is_reduce_only=True,
        timestamp=now,
    )
    approved, reason = rms.validate_order(exit_intent, 1500.0, pm, portfolio)
    assert approved is True
