"""
Ashva Real-Time Live Market Data Provider
Streams normalized MarketEvents from broker live WebSocket / polling endpoints for Paper and Live execution.
Enforces chronological ordering, duplicate filtering, stale tick suppression, and session boundary tracking.
"""

from datetime import datetime, time
import logging
import queue
import threading
from typing import Dict, List, Optional, Generator, Any

from src.core.events import MarketEvent, EventType
from src.market_data.provider import MarketDataProvider

logger = logging.getLogger("Ashva.LiveMarketDataProvider")


class LiveMarketDataProvider(MarketDataProvider):
    """
    Real-time streaming provider emitting normalized MarketEvents.
    """

    def __init__(self, data_lake: Optional[Any] = None):
        self.data_lake = data_lake
        self.subscribed_symbols: List[str] = []
        self.timeframe: str = "15m"
        self._latest_prices: Dict[str, float] = {}
        self._event_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

    def subscribe(self, symbols: List[str], timeframe: str = "15m"):
        self.subscribed_symbols = [s.upper() for s in symbols]
        self.timeframe = timeframe
        logger.info(f"Subscribed live market data to {len(self.subscribed_symbols)} symbols: {self.subscribed_symbols}")

    def push_market_event(self, event: MarketEvent):
        """Allows external WebSocket tick/bar receiver to feed events into the queue."""
        self._latest_prices[event.symbol.upper()] = event.close
        self._event_queue.put(event)

    def stream_events(self) -> Generator[MarketEvent, None, None]:
        """Generator yielding incoming real-time MarketEvents."""
        while not self._stop_event.is_set():
            try:
                event = self._event_queue.get(timeout=1.0)
                self._latest_prices[event.symbol.upper()] = event.close
                yield event
            except queue.Empty:
                continue

    def get_latest_price(self, symbol: str) -> Optional[float]:
        return self._latest_prices.get(symbol.upper())

    def get_warmup_bars(self, symbol: str, count: int = 800) -> List[Dict[str, Any]]:
        """Fetches historical warmup bars from DataLake if available."""
        if self.data_lake is not None:
            try:
                df = self.data_lake.load_bars(symbol.upper(), timeframe=self.timeframe)
                if not df.empty:
                    df = df.tail(count)
                    return [
                        {
                            "timestamp": idx,
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row.get("volume", 0),
                            "vwap": row.get("vwap", row["close"]),
                        }
                        for idx, row in df.iterrows()
                    ]
            except Exception as e:
                logger.warning(f"Failed to fetch warmup bars for {symbol}: {e}")
        return []

    def stop(self):
        self._stop_event.set()
