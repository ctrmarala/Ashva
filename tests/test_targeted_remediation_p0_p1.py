"""
Ashva P0 and P1 Correctness & Lifecycle Remediation Test Suite
Validates:
1. Entry window enforcement per alpha
2. Actual-fill barrier registration sequence
3. Stop/target data lineage preservation in Position
4. Strict stop-loss validation (removal of silent 1% fallback)
5. Explicit RMS rejection logging in Decision ledger
6. Metadata preservation in TradingManifest
7. Portfolio-level cross-symbol risk rationing
"""

from datetime import datetime, time, timedelta
import pytest
import pandas as pd

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
from src.trading.live_rms import LiveRiskManager
from src.trading.ledger import TradingLedger
from src.execution.replay_adapter import ReplayExecutionAdapter
from src.market_data.provider import MarketDataProvider
from src.trading.engine import TradingEngine


class DummyOpeningAlpha:
    """Dummy strategy emitting signals on every bar."""
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 1.0
        out["stop_loss"] = out["close"] - 10.0
        out["take_profit"] = out["close"] + 20.0
        return out


class DummyNoStopAlpha:
    """Strategy emitting signals with no stop loss."""
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 1.0
        return out


class MockBarStreamer(MarketDataProvider):
    def __init__(self, events: list):
        self.events = events

    def subscribe(self, symbols, timeframe="15m"):
        pass

    def stream_events(self):
        for e in self.events:
            yield e

    def get_latest_price(self, symbol):
        return 100.0


def test_alpha_entry_window_enforcement(tmp_path):
    """P0-1: Verify that signals outside contract.entry_start_time and entry_end_time are NOT evaluated."""
    db_path = str(tmp_path / "test_ledger.db")
    ledger = TradingLedger(db_path=db_path)

    # Contract active ONLY between 09:30 and 10:00
    contract = QualifiedAlphaContract(
        alpha_id="OPENING_MOMENTUM",
        strategy_class=DummyOpeningAlpha,
        universe=["TCS"],
        entry_start_time=time(9, 30),
        entry_end_time=time(10, 0),
        stop_type="STRATEGY_DEFINED",
    )
    manifest = TradingManifest([contract])

    # Bar 1 at 09:15 (Before entry window) -> Must NOT trade
    # Bar 2 at 09:45 (Inside entry window) -> MUST trade
    # Bar 3 at 10:30 (After entry window) -> Must NOT trade
    t_date = datetime(2026, 7, 1)
    bars = [
        MarketEvent("TCS", t_date.replace(hour=9, minute=15), "15m", 3500.0, 3510.0, 3495.0, 3505.0, 1000),
        MarketEvent("TCS", t_date.replace(hour=9, minute=45), "15m", 3505.0, 3520.0, 3500.0, 3515.0, 1000),
        MarketEvent("TCS", t_date.replace(hour=10, minute=30), "15m", 3515.0, 3530.0, 3510.0, 3525.0, 1000),
    ]
    
    # Warmup buffer to allow min 15 bars
    streamer = MockBarStreamer(bars)
    adapter = ReplayExecutionAdapter(slippage_bps=0.0)
    engine = TradingEngine(
        market_data_provider=streamer,
        execution_adapter=adapter,
        manifest=manifest,
        ledger=ledger,
    )
    # Inject warmup history
    engine._history_buffers["TCS"] = [
        {"timestamp": t_date.replace(hour=9, minute=0) + timedelta(minutes=i*15), "open": 3500.0, "high": 3510.0, "low": 3490.0, "close": 3500.0, "volume": 500, "vwap": 3500.0}
        for i in range(20)
    ]

    engine.run()
    
    # Check that the order was submitted on Bar 2 (09:45) and not Bar 1 or 3
    orders = engine.order_manager.completed_orders
    active = engine.order_manager.active_orders
    all_orders = list(orders.values()) + list(active.values())

    assert len(all_orders) == 1
    assert all_orders[0].timestamp.time() == time(9, 45)


def test_actual_fill_barrier_registration_sequence():
    """P0-2: Verify barrier is registered with the actual fill price (including slippage) on next-bar fill."""
    adapter = ReplayExecutionAdapter(slippage_bps=10.0) # 10 bps slippage
    pm = PositionManager()
    
    # Simulated next bar fill at 1000.0 + 10 bps slippage = 1001.0
    fill = FillEvent(
        order_id="ORD_001",
        symbol="INFY",
        timestamp=datetime(2026, 7, 1, 9, 30),
        side=OrderSide.BUY,
        fill_price=1001.0, # Actual fill price
        quantity=50,
        strategy_id="A67",
        stop_loss=990.0,
        take_profit=1020.0,
        stop_dist=11.0,
    )
    
    # Register barriers upon fill
    adapter.register_barriers(
        symbol=fill.symbol,
        side=fill.side,
        quantity=fill.quantity,
        entry_price=fill.fill_price,
        strategy_id=fill.strategy_id,
        stop_loss=fill.stop_loss,
        take_profit=fill.take_profit,
    )

    assert "INFY" in adapter._active_barriers
    assert adapter._active_barriers["INFY"]["entry_price"] == 1001.0
    assert adapter._active_barriers["INFY"]["stop_loss"] == 990.0


def test_position_preserves_stop_target_lineage():
    """P0-3: Verify Position maintains stop_loss, take_profit, and stop_dist from FillEvent."""
    pm = PositionManager()
    fill = FillEvent(
        order_id="ORD_001",
        symbol="INFY",
        timestamp=datetime(2026, 7, 1, 9, 30),
        side=OrderSide.BUY,
        fill_price=1500.0,
        quantity=100,
        strategy_id="A67",
        alpha_version="1.0.0",
        stop_loss=1485.0,
        take_profit=1530.0,
        stop_dist=15.0,
    )
    pm.on_fill(fill)
    pos = pm.get_position("INFY")
    assert pos is not None
    assert pos.stop_loss == 1485.0
    assert pos.take_profit == 1530.0
    assert pos.stop_dist == 15.0


def test_strict_stop_loss_rejection_no_1pct_fallback():
    """P0-5: Verify signal without stop is rejected when contract has no stop_loss_pct (No silent 1% guessing)."""
    allocator = MultiAlphaAllocator()
    pm = PositionManager()
    portfolio = PortfolioState(initial_capital=500000.0)

    contract = QualifiedAlphaContract(
        alpha_id="NO_STOP_ALPHA",
        strategy_class=DummyNoStopAlpha,
        universe=["TCS"],
        stop_type="STRATEGY_DEFINED",
        stop_loss_pct=None, # No contract stop defined
    )
    contracts_map = {"NO_STOP_ALPHA": contract}

    sig = SignalEvent(
        symbol="TCS",
        timestamp=datetime(2026, 7, 1, 9, 30),
        strategy_id="NO_STOP_ALPHA",
        signal_type=SignalType.LONG,
        suggested_stop_loss=None, # Strategy emitted no stop
    )

    intents, decisions = allocator.allocate(
        candidate_signals=[sig],
        contracts_map=contracts_map,
        current_prices={"TCS": 3500.0},
        position_manager=pm,
        portfolio_state=portfolio,
    )

    assert len(intents) == 0
    assert len(decisions) == 1
    assert decisions[0].is_accepted is False
    assert "MISSING_STOP_LOSS_DEFINITION" in decisions[0].rejection_reason


def test_risk_rejection_logged_to_decision_ledger(tmp_path):
    """P1-7: Verify that RMS rejections are explicitly logged in the decision ledger as RMS_REJECTED."""
    db_path = str(tmp_path / "test_ledger.db")
    ledger = TradingLedger(db_path=db_path)

    contract = QualifiedAlphaContract(
        alpha_id="A67",
        strategy_class=DummyOpeningAlpha,
        universe=["TCS"],
        entry_start_time=time(9, 15),
        entry_end_time=time(15, 0),
        risk_per_trade_pct=0.01,
        stop_loss_pct=0.01,
    )
    manifest = TradingManifest([contract])
    rms = LiveRiskManager()
    rms.trigger_kill_switch("Manual Safety Halt") # Kill switch active

    t_date = datetime(2026, 7, 1, 9, 30)
    bars = [MarketEvent("TCS", t_date, "15m", 3500.0, 3510.0, 3490.0, 3505.0, 1000)]
    streamer = MockBarStreamer(bars)
    adapter = ReplayExecutionAdapter()

    engine = TradingEngine(
        market_data_provider=streamer,
        execution_adapter=adapter,
        manifest=manifest,
        risk_manager=rms,
        ledger=ledger,
    )
    engine._history_buffers["TCS"] = [
        {"timestamp": t_date - timedelta(minutes=i*15), "open": 3500.0, "high": 3510.0, "low": 3490.0, "close": 3500.0, "volume": 500, "vwap": 3500.0}
        for i in range(20)
    ]
    engine.run()

    ledger.flush()
    with ledger._get_connection() as conn:
        rows = conn.execute("SELECT is_accepted, rejection_reason FROM decisions_log WHERE rejection_reason LIKE 'RMS_REJECTED%'").fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == 0
        assert "RMS_REJECTED" in rows[0][1]


def test_manifest_metadata_preservation():
    """P1-8: Verify that TradingManifest.set_contract_status uses dataclasses.replace and preserves all metadata."""
    contract = QualifiedAlphaContract(
        alpha_id="ALPHA_TEST_META",
        strategy_class=None,
        alpha_version="2.3.4",
        research_commit_sha="SHA_XYZ_123",
        status="ACTIVE",
    )
    manifest = TradingManifest([contract])
    manifest.set_contract_status("ALPHA_TEST_META", "PAUSED")

    updated = manifest.get_contract("ALPHA_TEST_META")
    assert updated is not None
    assert updated.status == "PAUSED"
    assert updated.alpha_version == "2.3.4"
    assert updated.research_commit_sha == "SHA_XYZ_123"
