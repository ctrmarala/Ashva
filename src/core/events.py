"""
Ashva Core Event Definitions
Typed, immutable dataclasses for event-driven processing across Data, Strategy, Risk, and Execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


class EventType(str, Enum):
    TICK = "TICK"
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
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
class SignalEvent:
    symbol: str
    timestamp: datetime
    strategy_id: str
    signal_type: SignalType
    confidence: float = 1.0              # 0.0 to 1.0 confidence/meta-label score
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = EventType.SIGNAL


@dataclass(frozen=True)
class OrderEvent:
    order_id: str
    symbol: str
    timestamp: datetime
    side: OrderSide
    order_type: OrderType
    quantity: int
    product_type: ProductType = ProductType.INTRADAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tag: str = "AshvaAlgo"
    event_type: EventType = EventType.ORDER


@dataclass(frozen=True)
class FillEvent:
    order_id: str
    symbol: str
    timestamp: datetime
    side: OrderSide
    fill_price: float
    quantity: int
    commission: float = 0.0
    slippage: float = 0.0
    cost_breakdown: Optional[Dict[str, Any]] = None
    event_type: EventType = EventType.FILL


@dataclass(frozen=True)
class RiskEvent:
    timestamp: datetime
    severity: str           # "WARNING", "CRITICAL", "EMERGENCY_HALT"
    rule_name: str
    message: str
    action_taken: str       # "REJECT_ORDER", "HALT_STRATEGY", "FLATTEN_ALL_POSITIONS"
    event_type: EventType = EventType.RISK_BREACH
