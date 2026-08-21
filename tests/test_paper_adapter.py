"""
Ashva Paper Execution Adapter Unit Test Suite
Tests PaperExecutionAdapter order submission, barrier management, slippage, and fill processing.
"""

from datetime import datetime
import pytest

from src.core.events import (
    OrderIntent, OrderSide, OrderType, ProductType, MarketEvent,
)
from src.execution.paper_adapter import PaperExecutionAdapter


def test_paper_adapter_order_lifecycle():
    """Verify PaperExecutionAdapter order submission and fill generation."""
    adapter = PaperExecutionAdapter(base_slippage_bps=0.0, simulated_latency_ms=10.0)
    
    intent = OrderIntent(
        strategy_id="ALPHA_04",
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=25,
        order_type=OrderType.MARKET,
        timestamp=datetime(2026, 7, 1, 9, 30),
    )
    order_event = adapter.submit_order(intent)
    assert order_event.order_id.startswith("PAP_ORD_")
    assert len(adapter.get_open_orders()) == 1

    # Process market bar
    mkt = MarketEvent(
        symbol="TCS",
        timestamp=datetime(2026, 7, 1, 9, 45),
        timeframe="15m",
        open=3500.0,
        high=3520.0,
        low=3490.0,
        close=3510.0,
        volume=5000,
    )
    fills = adapter.process_market_event(mkt)
    assert len(fills) == 1
    assert fills[0].symbol == "TCS"
    assert fills[0].quantity == 25
    assert fills[0].fill_price == 3510.0  # close price when slippage is 0
    assert len(adapter.get_open_orders()) == 0


def test_paper_adapter_barrier_execution():
    """Verify PaperExecutionAdapter intrabar SL barrier execution."""
    adapter = PaperExecutionAdapter(base_slippage_bps=0.0)
    
    adapter.register_barriers(
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=20,
        entry_price=3500.0,
        strategy_id="ALPHA_04",
        stop_loss=3450.0,
        take_profit=3600.0,
    )

    # Bar 1: Price does not hit barrier
    mkt1 = MarketEvent(
        symbol="TCS",
        timestamp=datetime(2026, 7, 1, 10, 0),
        timeframe="15m",
        open=3500.0,
        high=3520.0,
        low=3470.0,
        close=3480.0,
        volume=1000,
    )
    fills1 = adapter.process_market_event(mkt1)
    assert len(fills1) == 0

    # Bar 2: Price hits stop loss at 3450
    mkt2 = MarketEvent(
        symbol="TCS",
        timestamp=datetime(2026, 7, 1, 10, 15),
        timeframe="15m",
        open=3480.0,
        high=3485.0,
        low=3440.0,
        close=3445.0,
        volume=1000,
    )
    fills2 = adapter.process_market_event(mkt2)
    assert len(fills2) == 1
    assert fills2[0].is_stop_loss is True
    assert fills2[0].fill_price == 3450.0
    assert fills2[0].side == OrderSide.SELL
