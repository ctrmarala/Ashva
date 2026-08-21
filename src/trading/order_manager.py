"""
Ashva Production Order Manager
Maintains strict state tracking and lifecycle transitions for all algorithm orders:
Intent -> Submitted -> Accepted -> Filled / Partially Filled / Rejected / Cancelled.
"""

from datetime import datetime
import logging
from typing import Dict, List, Optional
from src.core.events import OrderIntent, OrderEvent, FillEvent, OrderStatus

logger = logging.getLogger("Ashva.OrderManager")


class OrderManager:
    """
    Central order lifecycle state manager.
    """

    def __init__(self):
        self.active_orders: Dict[str, OrderEvent] = {}
        self.completed_orders: Dict[str, OrderEvent] = {}
        self.intent_to_order_map: Dict[str, str] = {}  # intent_id -> order_id
        self.fills: List[FillEvent] = []

    def on_order_submitted(self, intent: OrderIntent, order: OrderEvent):
        """Records initial order submission."""
        self.active_orders[order.order_id] = order
        self.intent_to_order_map[intent.intent_id] = order.order_id

    def on_order_rejected(self, order_id: str, reason: str):
        """Handles order rejection by broker or RMS."""
        if order_id in self.active_orders:
            ord = self.active_orders.pop(order_id)
            updated_ord = OrderEvent(
                order_id=ord.order_id,
                symbol=ord.symbol,
                side=ord.side,
                order_type=ord.order_type,
                quantity=ord.quantity,
                status=OrderStatus.REJECTED,
                limit_price=ord.limit_price,
                stop_price=ord.stop_price,
                product_type=ord.product_type,
                strategy_id=ord.strategy_id,
                is_reduce_only=ord.is_reduce_only,
                reject_reason=reason,
                tag=ord.tag,
                timestamp=datetime.now(),
            )
            self.completed_orders[order_id] = updated_ord

    def on_fill(self, fill: FillEvent):
        """Processes execution fill and updates order state."""
        self.fills.append(fill)
        order_id = fill.order_id

        if order_id in self.active_orders:
            ord = self.active_orders.pop(order_id)
            completed_ord = OrderEvent(
                order_id=ord.order_id,
                symbol=ord.symbol,
                side=ord.side,
                order_type=ord.order_type,
                quantity=ord.quantity,
                status=OrderStatus.FILLED,
                limit_price=ord.limit_price,
                stop_price=ord.stop_price,
                product_type=ord.product_type,
                strategy_id=ord.strategy_id,
                is_reduce_only=ord.is_reduce_only,
                tag=ord.tag,
                timestamp=fill.timestamp,
            )
            self.completed_orders[order_id] = completed_ord

    def get_order(self, order_id: str) -> Optional[OrderEvent]:
        return self.active_orders.get(order_id) or self.completed_orders.get(order_id)

    def get_active_orders_for_symbol(self, symbol: str) -> List[OrderEvent]:
        s_clean = symbol.upper()
        return [o for o in self.active_orders.values() if o.symbol == s_clean]
