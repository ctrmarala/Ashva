"""
Unit Tests for Ashva Async Event Bus
"""

import asyncio
from datetime import datetime
import pytest
from src.core.events import EventType, TickEvent, RiskEvent
from src.core.event_bus import AsyncEventBus


@pytest.mark.asyncio
async def test_async_event_bus_pub_sub():
    bus = AsyncEventBus()
    received_events = []

    async def on_tick(event: TickEvent):
        received_events.append(event)

    bus.subscribe(EventType.TICK, on_tick)
    await bus.start()

    sample_tick = TickEvent(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        last_price=2500.0,
        bid_price=2499.5,
        ask_price=2500.5,
        volume=100
    )

    await bus.publish(sample_tick)
    await asyncio.sleep(0.05)  # Allow async queue to drain
    await bus.stop()

    assert len(received_events) == 1
    assert received_events[0].symbol == "RELIANCE"
    assert received_events[0].last_price == 2500.0


def test_sync_event_bus_dispatch():
    bus = AsyncEventBus()
    received_sync = []

    def on_risk(event: RiskEvent):
        received_sync.append(event)

    bus.subscribe_sync(EventType.RISK_BREACH, on_risk)

    sample_risk = RiskEvent(
        timestamp=datetime.now(),
        severity="CRITICAL",
        rule_name="DAILY_LOSS_LIMIT",
        message="Loss exceeded 1.5%",
        action_taken="HALT_STRATEGY"
    )

    bus.publish_nowait(sample_risk)
    assert len(received_sync) == 1
    assert received_sync[0].rule_name == "DAILY_LOSS_LIMIT"
