"""
Ashva Core Event Definitions
Typed, immutable dataclasses for event-driven processing across Data, Strategy, Risk, and Execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List


class EventType(str, Enum):
    TICK = "TICK"
    BAR = "BAR"
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER_INTENT = "ORDER_INTENT"
    ORDER = "ORDER"
    FILL = "FILL"
    POSITION_UPDATE = "POSITION_UPDATE"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    RISK_BREACH = "RISK_BREACH"
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"


class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    SL_MARKET = "SL_MARKET"


class OrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"


class ProductType(str, Enum):
    INTRADAY = "INTRADAY"   # MIS
    DELIVERY = "DELIVERY"   # CNC
    MARGIN = "MARGIN"       # NRML / F&O


@dataclass(frozen=True)
class TickEvent:
    symbol: str
    timestamp: datetime
    last_price: float
    bid_price: float
    ask_price: float
    volume: int
    open_interest: int = 0
    event_type: EventType = EventType.TICK


@dataclass(frozen=True)
class BarEvent:
    symbol: str
    timestamp: datetime
    timeframe: str          # e.g., '1m', '5m', '15m', '1d'
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    event_type: EventType = EventType.BAR


@dataclass(frozen=True)
class MarketEvent:
    """Standardized market update dispatched to TradingEngine."""
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    event_type: EventType = EventType.MARKET


@dataclass(frozen=True)
class SignalEvent:
    symbol: str
    timestamp: datetime
    strategy_id: str
    signal_type: SignalType
    confidence: float = 1.0              # 0.0 to 1.0 confidence/meta-label score
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    stop_dist: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = EventType.SIGNAL


@dataclass(frozen=True)
class OrderIntent:
    """Requested trading intent produced by Strategy/RMS prior to adapter dispatch."""
    strategy_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    product_type: ProductType = ProductType.INTRADAY
    is_reduce_only: bool = False
    tag: str = "AshvaAlgo"
    timestamp: datetime = field(default_factory=datetime.now)
    intent_id: str = field(default_factory=lambda: f"INT_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}")
    event_type: EventType = EventType.ORDER_INTENT


@dataclass(frozen=True)
class OrderEvent:
    """Lifecycle order state dispatched by Execution Adapter."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    order_id: str = field(default_factory=lambda: f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}")
    status: OrderStatus = OrderStatus.SUBMITTED
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    product_type: ProductType = ProductType.INTRADAY
    strategy_id: str = ""
    is_reduce_only: bool = False
    reject_reason: Optional[str] = None
    tag: str = "AshvaAlgo"
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: EventType = EventType.ORDER


@dataclass(frozen=True)
class FillEvent:
    """Execution fill confirmation returned by broker or replay adapter."""
    order_id: str
    symbol: str
    timestamp: datetime
    side: OrderSide
    fill_price: float
    quantity: int
    commission: float = 0.0
    slippage: float = 0.0
    cost_breakdown: Optional[Dict[str, Any]] = None
    strategy_id: str = ""
    is_stop_loss: bool = False
    event_type: EventType = EventType.FILL


@dataclass(frozen=True)
class PositionUpdateEvent:
    symbol: str
    timestamp: datetime
    quantity: int
    side: str
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    event_type: EventType = EventType.POSITION_UPDATE


@dataclass(frozen=True)
class PortfolioUpdateEvent:
    timestamp: datetime
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    total_equity: float
    open_positions_count: int
    drawdown_pct: float
    event_type: EventType = EventType.PORTFOLIO_UPDATE


@dataclass(frozen=True)
class RiskEvent:
    timestamp: datetime
    severity: str           # "WARNING", "CRITICAL", "EMERGENCY_HALT"
    rule_name: str
    message: str
    action_taken: str       # "REJECT_ORDER", "HALT_STRATEGY", "FLATTEN_ALL_POSITIONS"
    event_type: EventType = EventType.RISK_BREACH
