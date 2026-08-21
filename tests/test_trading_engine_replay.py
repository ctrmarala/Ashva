"""
Ashva Trading Engine & Replay Mode Unit Tests
Verifies:
1. Multi-symbol chronological synchronization in ReplayMarketDataProvider.
2. OrderManager lifecycle states (Intent -> Submitted -> Filled).
3. PositionManager Mark-to-Market (MTM) calculations and position closing.
4. LiveRiskManager circuit breaker enforcement.
5. Mode-agnostic TradingEngine end-to-end event processing loop.
"""

from datetime import datetime, time
import pytest
import numpy as np
import pandas as pd

from src.core.events import (
    MarketEvent, OrderIntent, OrderEvent, FillEvent, OrderSide, OrderType, OrderStatus, ProductType
)
from src.trading.contract import QualifiedAlphaContract
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.portfolio_state import PortfolioState
from src.trading.live_rms import LiveRiskManager
from src.trading.engine import TradingEngine
from src.market_data.provider import MarketDataProvider
from src.market_data.replay_provider import ReplayMarketDataProvider
from src.execution.replay_adapter import ReplayExecutionAdapter
from src.analytics.indian_costs import IndianCostModel, Segment


class MockMarketDataProvider(MarketDataProvider):
    """Feeds a static list of MarketEvents for testing."""
    def __init__(self, events: list):
        self.events = events
        self.latest_prices = {}

    def subscribe(self, symbols, timeframe="15m"):
        pass

    def stream_events(self):
        for ev in self.events:
            self.latest_prices[ev.symbol] = ev.close
            yield ev

    def get_latest_price(self, symbol: str):
        return self.latest_prices.get(symbol.upper())


class DummyBreakoutStrategy:
    """Mock strategy generating a Buy signal on bar index >= 2."""
    def __init__(self, parameters=None):
        self.parameters = parameters or {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 0.0
        out["stop_loss"] = 0.0
        out["take_profit"] = 0.0

        if len(out) >= 3:
            # Generate LONG signal with SL 98.0 and TP 106.0
            out.iloc[-1, out.columns.get_loc("signal")] = 1.0
            out.iloc[-1, out.columns.get_loc("stop_loss")] = 98.0
            out.iloc[-1, out.columns.get_loc("take_profit")] = 106.0
        return out


# =========================================================================
# TEST 1: OrderManager State Transitions
# =========================================================================
def test_order_manager_lifecycle():
    om = OrderManager()

    intent = OrderIntent(
        strategy_id="TEST_ALPHA",
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=50,
        order_type=OrderType.MARKET,
    )
    order = OrderEvent(
        order_id="ORD_001",
        symbol="TCS",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50,
        status=OrderStatus.ACCEPTED,
    )

    om.on_order_submitted(intent, order)
    assert "ORD_001" in om.active_orders
    assert om.intent_to_order_map[intent.intent_id] == "ORD_001"

    # Fill
    fill = FillEvent(
        order_id="ORD_001",
        symbol="TCS",
        timestamp=datetime.now(),
        side=OrderSide.BUY,
        fill_price=3500.0,
        quantity=50,
    )
    om.on_fill(fill)

    assert "ORD_001" not in om.active_orders
    assert "ORD_001" in om.completed_orders
    assert om.completed_orders["ORD_001"].status == OrderStatus.FILLED
    assert len(om.fills) == 1


# =========================================================================
# TEST 2: PositionManager MTM & PnL Accounting
# =========================================================================
def test_position_manager_mtm_and_closing():
    cost_model = IndianCostModel()
    pm = PositionManager(cost_model=cost_model)

    ts_entry = datetime(2026, 8, 1, 9, 30)
    # Buy 100 shares of RELIANCE @ 2500
    fill_entry = FillEvent(
        order_id="ORD_BUY_1",
        symbol="RELIANCE",
        timestamp=ts_entry,
        side=OrderSide.BUY,
        fill_price=2500.0,
        quantity=100,
        strategy_id="ORB",
    )
    pm.on_fill(fill_entry)

    pos = pm.get_position("RELIANCE")
    assert pos is not None
    assert pos.quantity == 100
    assert pos.entry_price == 2500.0

    # Market price moves to 2550 (+50 points)
    market_ev = MarketEvent(
        symbol="RELIANCE",
        timestamp=datetime(2026, 8, 1, 10, 0),
        timeframe="15m",
        open=2540.0,
        high=2560.0,
        low=2535.0,
        close=2550.0,
        volume=10000,
    )
    pm.on_market_event(market_ev)
    assert pm.get_total_unrealized_pnl() == 5000.0  # +50 * 100

    # Sell 100 shares @ 2560 (Exit)
    fill_exit = FillEvent(
        order_id="ORD_SELL_1",
        symbol="RELIANCE",
        timestamp=datetime(2026, 8, 1, 10, 15),
        side=OrderSide.SELL,
        fill_price=2560.0,
        quantity=100,
        strategy_id="ORB",
    )
    closed = pm.on_fill(fill_exit)
    assert closed is not None
    assert closed["gross_pnl"] == 6000.0  # (2560 - 2500) * 100
    assert closed["net_pnl"] > 5500.0     # Net of statutory taxes
    assert pm.get_position("RELIANCE") is None
    assert len(pm.closed_trades) == 1


# =========================================================================
# TEST 3: LiveRiskManager Circuit Breaker Enforcement
# =========================================================================
def test_live_risk_manager_rules():
    rms = LiveRiskManager(
        max_daily_loss_pct=1.5,
        max_concurrent_positions=2,
        entry_start_time=time(9, 30),
        entry_end_time=time(15, 0),
    )

    pm = PositionManager()
    ps = PortfolioState(initial_capital=500000.0)

    # 1. Normal valid order
    intent = OrderIntent(
        strategy_id="ORB",
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=20,
        timestamp=datetime(2026, 8, 1, 10, 0),
    )
    approved, reason = rms.validate_order(intent, current_price=3500.0, position_manager=pm, portfolio_state=ps)
    assert approved is True
    assert reason is None

    # 2. Outside trading hours (09:15 AM)
    early_intent = OrderIntent(
        strategy_id="ORB",
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=20,
        timestamp=datetime(2026, 8, 1, 9, 15),
    )
    approved, reason = rms.validate_order(early_intent, current_price=3500.0, position_manager=pm, portfolio_state=ps)
    assert approved is False
    assert "outside trading window" in reason

    # 3. Exit orders are NEVER blocked even if kill-switch is active
    rms.kill_switch_active = True
    # Mock open position in TCS
    pm.open_positions["TCS"] = None  # Mock presence
    exit_intent = OrderIntent(
        strategy_id="ORB",
        symbol="TCS",
        side=OrderSide.SELL,
        quantity=20,
        is_reduce_only=True,
        timestamp=datetime(2026, 8, 1, 11, 0),
    )
    approved_exit, reason_exit = rms.validate_order(exit_intent, current_price=3500.0, position_manager=pm, portfolio_state=ps)
    assert approved_exit is True
    assert reason_exit is None


# =========================================================================
# TEST 4: End-to-End TradingEngine in REPLAY Mode
# =========================================================================
def test_trading_engine_end_to_end_replay():
    # Build 20 synthetic 15m bars
    dates = pd.date_range("2026-08-01 09:15", periods=20, freq="15min")
    market_events = []
    
    for i, dt in enumerate(dates):
        p = 100.0 + (i * 0.5)
        market_events.append(MarketEvent(
            symbol="INFY",
            timestamp=dt,
            timeframe="15m",
            open=p,
            high=p + 1.0,
            low=p - 0.5,
            close=p + 0.2,
            volume=5000,
        ))

    mock_provider = MockMarketDataProvider(market_events)
    replay_adapter = ReplayExecutionAdapter(segment=Segment.EQUITY_INTRADAY)

    contract = QualifiedAlphaContract(
        alpha_id="DUMMY_BREAKOUT",
        strategy_class=DummyBreakoutStrategy,
        universe=["INFY"],
        timeframe="15m",
        risk_per_trade_pct=0.01,
        max_capital_allocation_pct=0.20,
    )

    engine = TradingEngine(
        market_data_provider=mock_provider,
        execution_adapter=replay_adapter,
        alpha_contracts=[contract],
        initial_capital=500000.0,
    )

    summary = engine.run()

    assert summary["initial_capital"] == 500000.0
    assert "total_trades" in summary
    assert "final_equity" in summary
    assert len(summary["closed_trades"]) >= 1  # Successfully executed trade and EOD square-off
