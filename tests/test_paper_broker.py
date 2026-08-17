"""
Unit Tests for Institutional Paper Trading Simulator
"""

from datetime import datetime
import pytest
from src.core.events import OrderEvent, OrderSide, OrderType, ProductType
from src.execution.paper_broker import PaperBroker


def test_paper_broker_trade_lifecycle():
    broker = PaperBroker(initial_capital=500000.0, slippage_bps=0.0)

    # 1. Submit BUY Order (100 shares @ 2500 = 2,50,000)
    buy_order = OrderEvent(
        order_id="ORD_BUY_1",
        symbol="RELIANCE",
        timestamp=datetime.now(),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
    )

    fill_buy = broker.submit_order(buy_order, current_price=2500.0)
    assert fill_buy.fill_price == 2500.0
    assert "RELIANCE" in broker.open_positions
    assert broker.cash == 250000.0
    assert broker.equity == 500000.0

    # 2. Mark to Market price appreciation to 2550 (+5,000 gross)
    broker.update_market_price("RELIANCE", 2550.0)
    assert broker.equity == 505000.0

    # 3. Submit SELL Order to Close
    sell_order = OrderEvent(
        order_id="ORD_SELL_1",
        symbol="RELIANCE",
        timestamp=datetime.now(),
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=100,
    )

    fill_sell = broker.submit_order(sell_order, current_price=2550.0)
    assert fill_sell.fill_price == 2550.0
    assert "RELIANCE" not in broker.open_positions
    assert broker.equity > 504000.0  # Profitable net of Indian taxes & brokerage
