"""
Ashva Execution Adapter Abstract Interface
Defines the contract for order submission, cancellation, and fill event generation
across Replay, Paper, and Live broker adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from src.core.events import OrderIntent, OrderEvent, FillEvent, MarketEvent


class ExecutionAdapter(ABC):
    """
    Abstract interface for broker execution environments.
    """

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> OrderEvent:
        """Translates an OrderIntent into an OrderEvent and submits it."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an active pending order."""
        pass

    @abstractmethod
    def process_market_event(self, event: MarketEvent) -> List[FillEvent]:
        """
        Evaluates active orders / stops against incoming market updates
        and produces execution fills.
        """
        pass

    @abstractmethod
    def get_open_orders(self) -> List[OrderEvent]:
        """Returns currently active open orders."""
        pass
