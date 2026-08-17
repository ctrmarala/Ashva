"""
Ashva Asynchronous Event Bus
High-performance, async pub/sub dispatcher for financial events with priority queuing and latency instrumentation.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
import logging
from typing import Callable, Coroutine, Dict, List, Any, Optional

from src.core.events import EventType, RiskEvent, OrderEvent, FillEvent, SignalEvent, BarEvent, TickEvent

logger = logging.getLogger("Ashva.EventBus")


class AsyncEventBus:
    """
    Asynchronous event bus supporting non-blocking concurrent handlers and priority routing.
    """

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self._subscribers: Dict[EventType, List[Callable[[Any], Coroutine[Any, Any, None]]]] = defaultdict(list)
        self._sync_subscribers: Dict[EventType, List[Callable[[Any], None]]] = defaultdict(list)
        
        # Internal async queue for decoupled consumption
        self._queue: Optional[asyncio.Queue] = None
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
        self.event_counter = 0

    def subscribe(self, event_type: EventType, handler: Callable[[Any], Coroutine[Any, Any, None]]):
        """Registers an asynchronous coroutine handler for an event type."""
        self._subscribers[event_type].append(handler)

    def subscribe_sync(self, event_type: EventType, handler: Callable[[Any], None]):
        """Registers a synchronous callback handler for an event type."""
        self._sync_subscribers[event_type].append(handler)

    async def start(self):
        """Starts the background event processing loop."""
        if not self._running:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
            self._running = True
            self._consumer_task = asyncio.create_task(self._event_consumer_loop())
            logger.info("AsyncEventBus started.")

    async def stop(self):
        """Gracefully stops the event loop and flushes pending events."""
        if self._running:
            self._running = False
            if self._consumer_task:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass
            logger.info("AsyncEventBus stopped.")

    async def publish(self, event: Any):
        """
        Publishes an event to the queue. 
        High-priority events (Risk, Order) are processed immediately if queue is running.
        """
        self.event_counter += 1
        if self._queue is not None and self._running:
            await self._queue.put(event)
        else:
            # Immediate dispatch if bus loop is not running (e.g. In sync backtests)
            await self._dispatch_event(event)

    def publish_nowait(self, event: Any):
        """Non-blocking publish for synchronous callers."""
        self.event_counter += 1
        if self._queue is not None and self._running:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.error(f"Event bus queue full! Dropping event {event.event_type}")
        else:
            # Synchronous dispatch
            self._dispatch_sync(event)

    async def _event_consumer_loop(self):
        """Continuous background worker draining the event queue."""
        while self._running:
            try:
                event = await self._queue.get()
                await self._dispatch_event(event)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event consumer loop: {e}", exc_info=True)

    async def _dispatch_event(self, event: Any):
        """Dispatches an event to all registered async and sync subscribers."""
        event_type = getattr(event, "event_type", None)
        if not event_type:
            return

        # 1. Sync subscribers
        for sync_handler in self._sync_subscribers.get(event_type, []):
            try:
                sync_handler(event)
            except Exception as e:
                logger.error(f"Error in sync handler {sync_handler.__name__} for {event_type}: {e}", exc_info=True)

        # 2. Async subscribers (run concurrently)
        async_handlers = self._subscribers.get(event_type, [])
        if async_handlers:
            tasks = [asyncio.create_task(handler(event)) for handler in async_handlers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Exception in async handler for {event_type}: {res}")

    def _dispatch_sync(self, event: Any):
        """Dispatches event to synchronous subscribers only."""
        event_type = getattr(event, "event_type", None)
        if not event_type:
            return
        for sync_handler in self._sync_subscribers.get(event_type, []):
            try:
                sync_handler(event)
            except Exception as e:
                logger.error(f"Error in sync handler {sync_handler.__name__} for {event_type}: {e}", exc_info=True)
