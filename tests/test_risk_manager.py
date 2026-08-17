"""
Unit Tests for Real-Time Risk Management System (RMS)
"""

from datetime import datetime
import pytest
from src.core.events import OrderEvent, OrderSide, OrderType, ProductType
from src.risk.risk_manager import RiskManager


@pytest.fixture
def sample_order():
    return OrderEvent(
        order_id="ORD_001",
        symbol="RELIANCE",
        timestamp=datetime.now(),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50,
        product_type=ProductType.INTRADAY,
    )


def test_order_validation_approved(sample_order):
    rms = RiskManager(max_daily_loss_pct=1.5, max_open_positions=4)
    rms.set_starting_equity(500000.0)

    is_valid, reason = rms.validate_order(
        order=sample_order,
        current_equity=500000.0,
        current_price=2500.0,
        open_positions_count=1,
        current_time=datetime(2026, 1, 1, 10, 30, 0),
    )
    assert is_valid is True
    assert reason is None


def test_daily_loss_circuit_breaker(sample_order):
    rms = RiskManager(max_daily_loss_pct=1.5)
    rms.set_starting_equity(500000.0)

    # Simulate equity dropping from 5,00,000 to 4,90,000 (Loss = 10,000 = 2.0% > 1.5%)
    is_valid, reason = rms.validate_order(
        order=sample_order,
        current_equity=490000.0,
        current_price=2500.0,
        open_positions_count=1,
        current_time=datetime(2026, 1, 1, 11, 0, 0),
    )
    assert is_valid is False
    assert "Daily loss" in reason
    assert rms.trading_halted_for_day is True


def test_kill_switch(sample_order):
    rms = RiskManager()
    rms.trigger_kill_switch(reason="MANUAL_TEST")

    is_valid, reason = rms.validate_order(
        order=sample_order,
        current_equity=500000.0,
        current_price=2500.0,
        open_positions_count=0,
    )
    assert is_valid is False
    assert "kill-switch is active" in reason


def test_time_restrictions(sample_order):
    rms = RiskManager(intraday_entry_cutoff="15:00:00", intraday_square_off="15:15:00")
    
    # 15:05 PM (Past entry cutoff) -> Should reject
    is_valid, reason = rms.validate_order(
        order=sample_order,
        current_equity=500000.0,
        current_price=2500.0,
        open_positions_count=0,
        current_time=datetime(2026, 1, 1, 15, 5, 0),
    )
    assert is_valid is False
    assert "Past entry cutoff time" in reason
