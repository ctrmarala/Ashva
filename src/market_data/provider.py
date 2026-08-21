"""
Ashva Market Data Provider Abstract Interface
Defines the contract for all streaming market data providers (Replay, Paper, Live).
"""

from abc import ABC, abstractmethod
from typing import List, Generator, Optional, Dict, Any
from src.core.events import MarketEvent


class MarketDataProvider(ABC):
    """
    Abstract interface for streaming market data events into TradingEngine.
    """

    @abstractmethod
    def subscribe(self, symbols: List[str], timeframe: str = "15m"):
        """Subscribes provider to a set of symbols and bar timeframe."""
        pass

    @abstractmethod
    def stream_events(self) -> Generator[MarketEvent, None, None]:
        """Streams chronological MarketEvents into the engine event loop."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Returns the most recent price for a symbol."""
        pass

    def get_warmup_bars(self, symbol: str, count: int = 800) -> List[Dict[str, Any]]:
        """Returns historical warmup bars prior to streaming to prime rolling indicators."""
        return []
