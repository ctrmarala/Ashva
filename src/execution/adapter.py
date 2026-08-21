"""
Ashva Execution Adapter Base Interface
Mode-agnostic abstraction defining the contract between TradingEngine and execution destinations:
ReplayExecutionAdapter, PaperExecutionAdapter, and BrokerExecutionAdapter.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.core.events import OrderIntent, OrderEvent, FillEvent, MarketEvent, OrderSide


class ExecutionAdapter(ABC):
    """
    Standardized execution interface.
    """

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> OrderEvent:
        """Submits an order intent to the execution destination."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Requests cancellation of an open order."""
        pass

    @abstractmethod
    def process_market_event(self, market_event: MarketEvent) -> List[FillEvent]:
        """
        Evaluates active orders / barriers against incoming market data.
        Returns any generated execution fills.
        """
        pass

    @abstractmethod
    def get_open_orders(self) -> List[OrderEvent]:
        """Returns currently active open orders."""
        pass

    def register_barriers(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: float,
        strategy_id: str,
        alpha_version: str = "1.0.0",
        signal_id: str = "",
        decision_id: str = "",
        order_id: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Optional hook for adapters simulating intrabar stop-loss / take-profit barriers."""
        pass

    def clear_barriers(self, symbol: str):
        """Optional hook to clear active barriers upon position close."""
        pass
