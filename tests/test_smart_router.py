"""
Unit Tests for Smart Order Router (TWAP and Iceberg)
"""

from datetime import datetime
import pytest
from src.core.events import OrderEvent, OrderSide, OrderType
from src.execution.smart_router import SmartOrderRouter


def test_twap_slicing():
    parent_order = OrderEvent(
        order_id="ORD_PARENT_1",
        symbol="RELIANCE",
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=300,
        limit_price=2500.0,
    )

    slices = SmartOrderRouter.slice_twap(parent_order, total_duration_minutes=30, num_slices=6)
    assert len(slices) == 6
    assert sum(s["quantity"] for s in slices) == 300
    assert slices[0]["quantity"] == 50
    assert slices[-1]["scheduled_time"] > slices[0]["scheduled_time"]


def test_iceberg_slicing():
    parent_order = OrderEvent(
        order_id="ORD_ICE_1",
        symbol="HDFCBANK",
        timestamp=datetime.now(),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=250,
    )

    clips = SmartOrderRouter.slice_iceberg(parent_order, visible_clip_size=100)
    assert len(clips) == 3
    assert sum(c["quantity"] for c in clips) == 250
    assert clips[0]["quantity"] == 100
    assert clips[1]["quantity"] == 100
    assert clips[2]["quantity"] == 50
