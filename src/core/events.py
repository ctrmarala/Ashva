"""
Ashva Core Event Definitions
Typed, immutable dataclasses for event-driven processing across Data, Strategy, Allocation, Risk, Execution, and Ledger.
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
    DECISION = "DECISION"
    ORDER_INTENT = "ORDER_INTENT"
    ORDER = "ORDER"
    FILL = "FILL"
    POSITION_UPDATE = "POSITION_UPDATE"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    RISK_BREACH = "RISK_BREACH"
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
    SYSTEM_EVENT = "SYSTEM_EVENT"


class TradingMode(str, Enum):
    REPLAY = "REPLAY"
    PAPER = "PAPER"
    LIVE = "LIVE"


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
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    ACCEPTED = "ACKNOWLEDGED"          # Alias for backwards compatibility
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


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
    bid: Optional[float] = None
    ask: Optional[float] = None
    event_type: EventType = EventType.MARKET


@dataclass(frozen=True)
class SignalEvent:
    """Quantitative signal produced by a Qualified Alpha."""
    symbol: str
    timestamp: datetime
    strategy_id: str                     # alpha_id
    signal_type: SignalType
    confidence: float = 1.0              # 0.0 to 1.0 confidence/meta-label score
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    stop_dist: Optional[float] = None
    alpha_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}")
    event_type: EventType = EventType.SIGNAL


@dataclass(frozen=True)
class DecisionEvent:
    """Allocation decision record for multi-alpha signal evaluation."""
    decision_id: str
    signal_id: str
    timestamp: datetime
    alpha_id: str
    alpha_version: str
    symbol: str
    is_accepted: bool
    allocated_quantity: int = 0
    risk_budget: float = 0.0
    rejection_reason: Optional[str] = None
    competing_alphas: List[str] = field(default_factory=list)
    event_type: EventType = EventType.DECISION


@dataclass(frozen=True)
class OrderIntent:
    """Requested trading intent produced by MultiAlphaAllocator / Risk."""
    strategy_id: str                     # alpha_id
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    product_type: ProductType = ProductType.INTRADAY
    is_reduce_only: bool = False
    alpha_version: str = "1.0.0"
    signal_id: str = ""
    decision_id: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
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
    intent_id: str = ""
    decision_id: str = ""
    signal_id: str = ""
    strategy_id: str = ""
    alpha_version: str = "1.0.0"
    status: OrderStatus = OrderStatus.SUBMITTED
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_dist: Optional[float] = None
    product_type: ProductType = ProductType.INTRADAY
    is_reduce_only: bool = False
    reject_reason: Optional[str] = None
    broker_order_id: Optional[str] = None
    mode: TradingMode = TradingMode.REPLAY
    tag: str = "AshvaAlgo"
    timestamp: datetime = field(default_factory=datetime.now)
    broker_ack_timestamp: Optional[datetime] = None
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
    latency_ms: float = 0.0
    cost_breakdown: Optional[Dict[str, Any]] = None
    strategy_id: str = ""
    alpha_version: str = "1.0.0"
    signal_id: str = ""
    decision_id: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_dist: Optional[float] = None
    is_stop_loss: bool = False
    fill_id: str = field(default_factory=lambda: f"FILL_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}")
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
    strategy_id: str = ""
    alpha_version: str = "1.0.0"
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
    daily_loss_pct: float = 0.0
    mode: TradingMode = TradingMode.REPLAY
    event_type: EventType = EventType.PORTFOLIO_UPDATE


@dataclass(frozen=True)
class RiskEvent:
    timestamp: datetime
    severity: str           # "WARNING", "CRITICAL", "EMERGENCY_HALT"
    rule_name: str
    message: str
    action_taken: str       # "REJECT_ORDER", "HALT_STRATEGY", "FLATTEN_ALL_POSITIONS"
    alpha_id: Optional[str] = None
    symbol: Optional[str] = None
    event_type: EventType = EventType.RISK_BREACH


@dataclass(frozen=True)
class SystemEvent:
    timestamp: datetime
    event_name: str
    severity: str           # "INFO", "WARNING", "ERROR", "CRITICAL"
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = EventType.SYSTEM_EVENT
