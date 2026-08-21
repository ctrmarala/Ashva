"""
Ashva Asynchronous Structured Trading Ledger & WAL
Non-blocking event-driven journal persisting complete trading lifecycle records to SQLite WAL.
Guarantees sub-millisecond execution critical path by offloading persistence to a dedicated background queue.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import queue
import sqlite3
import threading
import time
from typing import Dict, List, Any, Optional, Tuple

from src.core.events import (
    SignalEvent, DecisionEvent, OrderEvent, FillEvent,
    PositionUpdateEvent, PortfolioUpdateEvent, RiskEvent, SystemEvent,
    TradingMode,
)

logger = logging.getLogger("Ashva.TradingLedger")


class TradingLedger:
    """
    Append-Only Structured Trading Ledger with asynchronous background writer.
    """

    def __init__(self, db_path: str = "data_lake/trading_ledger.db", max_queue_size: int = 10000):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_queue_size = max_queue_size
        
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._dropped_events_count = 0

        self._init_db()
        self._start_worker()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # 1. Signals Log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    alpha_id TEXT NOT NULL,
                    alpha_version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence DOUBLE NOT NULL,
                    suggested_stop_loss DOUBLE,
                    suggested_take_profit DOUBLE,
                    stop_dist DOUBLE,
                    metadata_json TEXT NOT NULL,
                    persisted_at TEXT NOT NULL
                );
            """)
            # 2. Decisions Log (Multi-Alpha Allocations)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT UNIQUE NOT NULL,
                    signal_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    alpha_id TEXT NOT NULL,
                    alpha_version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    is_accepted INTEGER NOT NULL,
                    allocated_quantity INTEGER NOT NULL,
                    risk_budget DOUBLE NOT NULL,
                    rejection_reason TEXT,
                    competing_alphas_json TEXT NOT NULL,
                    persisted_at TEXT NOT NULL
                );
            """)
            # 3. Orders Log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    intent_id TEXT,
                    decision_id TEXT,
                    signal_id TEXT,
                    alpha_id TEXT NOT NULL,
                    alpha_version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    limit_price DOUBLE,
                    stop_price DOUBLE,
                    product_type TEXT NOT NULL,
                    is_reduce_only INTEGER NOT NULL,
                    reject_reason TEXT,
                    broker_order_id TEXT,
                    mode TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    broker_ack_at TEXT,
                    persisted_at TEXT NOT NULL
                );
            """)
            # 4. Fills Log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fills_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fill_id TEXT UNIQUE NOT NULL,
                    order_id TEXT NOT NULL,
                    decision_id TEXT,
                    signal_id TEXT,
                    alpha_id TEXT NOT NULL,
                    alpha_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    fill_price DOUBLE NOT NULL,
                    quantity INTEGER NOT NULL,
                    commission DOUBLE NOT NULL,
                    slippage DOUBLE NOT NULL,
                    latency_ms DOUBLE NOT NULL,
                    is_stop_loss INTEGER NOT NULL,
                    cost_breakdown_json TEXT,
                    persisted_at TEXT NOT NULL
                );
            """)
            # 5. Authoritative Trade Ledger (Closed Trades with Full Attribution & MFE/MAE)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_ledger (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alpha_id TEXT NOT NULL,
                    alpha_version TEXT NOT NULL,
                    signal_id TEXT,
                    decision_id TEXT,
                    order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    exit_price DOUBLE NOT NULL,
                    gross_pnl DOUBLE NOT NULL,
                    net_pnl DOUBLE NOT NULL,
                    slippage_paid DOUBLE NOT NULL,
                    total_costs DOUBLE NOT NULL,
                    mfe DOUBLE NOT NULL,
                    mae DOUBLE NOT NULL,
                    mfe_pct DOUBLE NOT NULL,
                    mae_pct DOUBLE NOT NULL,
                    holding_period_bars INTEGER NOT NULL,
                    exit_reason TEXT NOT NULL,
                    cost_breakdown_json TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    persisted_at TEXT NOT NULL
                );
            """)
            # 6. Portfolio Snapshots Log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cash DOUBLE NOT NULL,
                    realized_pnl DOUBLE NOT NULL,
                    unrealized_pnl DOUBLE NOT NULL,
                    total_equity DOUBLE NOT NULL,
                    open_positions_count INTEGER NOT NULL,
                    drawdown_pct DOUBLE NOT NULL,
                    daily_loss_pct DOUBLE NOT NULL,
                    mode TEXT NOT NULL,
                    persisted_at TEXT NOT NULL
                );
            """)
            # 7. System & Risk Events Log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_events_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    persisted_at TEXT NOT NULL
                );
            """)
            # Indices for rapid querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_sym ON signals_log (symbol, timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_alpha ON orders_log (alpha_id, created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_alpha ON trade_ledger (alpha_id, exit_time);")

    def _start_worker(self):
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._writer_loop, daemon=True, name="AshvaLedgerWriter")
        self._worker_thread.start()

    def _writer_loop(self):
        """Dedicated background thread draining event queue into SQLite WAL."""
        batch = []
        last_flush_time = time.time()

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                # Wait up to 100ms for incoming item
                item = self._queue.get(timeout=0.1)
                batch.append(item)
                self._queue.task_done()
            except queue.Empty:
                pass

            now = time.time()
            if batch and (len(batch) >= 20 or (now - last_flush_time) >= 0.1 or self._queue.empty() or self._stop_event.is_set()):
                self._flush_batch(batch)
                batch.clear()
                last_flush_time = now

        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[Tuple[str, Any]]):
        """Flushes a batch of event records inside a single SQLite transaction."""
        persisted_at = datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                for event_type, record in batch:
                    if event_type == "SIGNAL":
                        sig: SignalEvent = record
                        conn.execute("""
                            INSERT OR REPLACE INTO signals_log (
                                signal_id, timestamp, alpha_id, alpha_version, symbol, signal_type,
                                confidence, suggested_stop_loss, suggested_take_profit, stop_dist,
                                metadata_json, persisted_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            sig.signal_id, sig.timestamp.isoformat(), sig.strategy_id,
                            sig.alpha_version, sig.symbol.upper(), sig.signal_type.value,
                            sig.confidence, sig.suggested_stop_loss, sig.suggested_take_profit,
                            sig.stop_dist, json.dumps(sig.metadata, default=str), persisted_at,
                        ))

                    elif event_type == "DECISION":
                        dec: DecisionEvent = record
                        conn.execute("""
                            INSERT OR REPLACE INTO decisions_log (
                                decision_id, signal_id, timestamp, alpha_id, alpha_version, symbol,
                                is_accepted, allocated_quantity, risk_budget, rejection_reason,
                                competing_alphas_json, persisted_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            dec.decision_id, dec.signal_id, dec.timestamp.isoformat(), dec.alpha_id,
                            dec.alpha_version, dec.symbol.upper(), 1 if dec.is_accepted else 0,
                            dec.allocated_quantity, dec.risk_budget, dec.rejection_reason,
                            json.dumps(dec.competing_alphas), persisted_at,
                        ))

                    elif event_type == "ORDER":
                        ord: OrderEvent = record
                        conn.execute("""
                            INSERT OR REPLACE INTO orders_log (
                                order_id, intent_id, decision_id, signal_id, alpha_id, alpha_version,
                                symbol, side, order_type, quantity, status, limit_price, stop_price,
                                product_type, is_reduce_only, reject_reason, broker_order_id, mode,
                                tag, created_at, broker_ack_at, persisted_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            ord.order_id, ord.intent_id, ord.decision_id, ord.signal_id,
                            ord.strategy_id, ord.alpha_version, ord.symbol.upper(), ord.side.value,
                            ord.order_type.value, ord.quantity, ord.status.value, ord.limit_price,
                            ord.stop_price, ord.product_type.value, 1 if ord.is_reduce_only else 0,
                            ord.reject_reason, ord.broker_order_id, ord.mode.value if hasattr(ord.mode, "value") else str(ord.mode),
                            ord.tag, ord.timestamp.isoformat(),
                            ord.broker_ack_timestamp.isoformat() if ord.broker_ack_timestamp else None,
                            persisted_at,
                        ))

                    elif event_type == "FILL":
                        f: FillEvent = record
                        conn.execute("""
                            INSERT OR REPLACE INTO fills_log (
                                fill_id, order_id, decision_id, signal_id, alpha_id, alpha_version,
                                timestamp, symbol, side, fill_price, quantity, commission, slippage,
                                latency_ms, is_stop_loss, cost_breakdown_json, persisted_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            f.fill_id, f.order_id, f.decision_id, f.signal_id, f.strategy_id,
                            f.alpha_version, f.timestamp.isoformat(), f.symbol.upper(), f.side.value,
                            f.fill_price, f.quantity, f.commission, f.slippage, f.latency_ms,
                            1 if f.is_stop_loss else 0, json.dumps(f.cost_breakdown or {}, default=str),
                            persisted_at,
                        ))

                    elif event_type == "TRADE":
                        t: Dict[str, Any] = record
                        conn.execute("""
                            INSERT INTO trade_ledger (
                                alpha_id, alpha_version, signal_id, decision_id, order_id, symbol,
                                side, quantity, entry_time, exit_time, entry_price, exit_price,
                                gross_pnl, net_pnl, slippage_paid, total_costs, mfe, mae, mfe_pct,
                                mae_pct, holding_period_bars, exit_reason, cost_breakdown_json, mode,
                                persisted_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            t.get("strategy_id", "UNKNOWN"), t.get("alpha_version", "1.0.0"),
                            t.get("signal_id"), t.get("decision_id"), t.get("order_id"),
                            t.get("symbol", "").upper(), t.get("side", ""), t.get("quantity", 0),
                            t.get("entry_time", "").isoformat() if isinstance(t.get("entry_time"), datetime) else str(t.get("entry_time")),
                            t.get("exit_time", "").isoformat() if isinstance(t.get("exit_time"), datetime) else str(t.get("exit_time")),
                            t.get("entry_price", 0.0), t.get("exit_price", 0.0),
                            t.get("gross_pnl", 0.0), t.get("net_pnl", 0.0),
                            t.get("slippage_paid", 0.0), t.get("total_costs", 0.0),
                            t.get("mfe", 0.0), t.get("mae", 0.0), t.get("mfe_pct", 0.0),
                            t.get("mae_pct", 0.0), t.get("holding_period_bars", 0),
                            t.get("exit_reason", "NORMAL"),
                            json.dumps(t.get("cost_breakdown", {}), default=str),
                            t.get("mode", "REPLAY"), persisted_at,
                        ))

                    elif event_type == "PORTFOLIO":
                        p: PortfolioUpdateEvent = record
                        conn.execute("""
                            INSERT INTO portfolio_snapshots (
                                timestamp, cash, realized_pnl, unrealized_pnl, total_equity,
                                open_positions_count, drawdown_pct, daily_loss_pct, mode, persisted_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (
                            p.timestamp.isoformat(), p.cash, p.realized_pnl, p.unrealized_pnl,
                            p.total_equity, p.open_positions_count, p.drawdown_pct, p.daily_loss_pct,
                            p.mode.value if hasattr(p.mode, "value") else str(p.mode), persisted_at,
                        ))

                    elif event_type == "SYSTEM":
                        sys_ev: SystemEvent = record
                        conn.execute("""
                            INSERT INTO system_events_log (
                                timestamp, event_type, severity, component, message, details_json, persisted_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?);
                        """, (
                            sys_ev.timestamp.isoformat(), sys_ev.event_name, sys_ev.severity,
                            sys_ev.component, sys_ev.message, json.dumps(sys_ev.details, default=str),
                            persisted_at,
                        ))
        except Exception as e:
            logger.error(f"Failed to flush batch to trading ledger: {e}")

    def log_signal(self, signal: SignalEvent):
        """Asynchronously queues a signal record."""
        self._enqueue("SIGNAL", signal)

    def log_decision(self, decision: DecisionEvent):
        """Asynchronously queues an allocation decision record."""
        self._enqueue("DECISION", decision)

    def log_order(self, order: OrderEvent):
        """Asynchronously queues an order lifecycle record."""
        self._enqueue("ORDER", order)

    def log_fill(self, fill: FillEvent):
        """Asynchronously queues an execution fill record."""
        self._enqueue("FILL", fill)

    def log_closed_trade(self, trade_record: Dict[str, Any]):
        """Asynchronously queues a completed trade record."""
        self._enqueue("TRADE", trade_record)

    def log_portfolio_snapshot(self, snapshot: PortfolioUpdateEvent):
        """Asynchronously queues a portfolio update record."""
        self._enqueue("PORTFOLIO", snapshot)

    def log_system_event(self, event: SystemEvent):
        """Asynchronously queues a system/risk event."""
        self._enqueue("SYSTEM", event)

    def _enqueue(self, event_type: str, item: Any):
        try:
            self._queue.put_nowait((event_type, item))
        except queue.Full:
            self._dropped_events_count += 1
            logger.critical(f"TradingLedger queue overflow! Dropped events: {self._dropped_events_count}")

    def flush(self, timeout: float = 5.0):
        """Waits for all pending records in the queue to be persisted."""
        start_t = time.time()
        while not self._queue.empty() and (time.time() - start_t) < timeout:
            time.sleep(0.05)

    def shutdown(self):
        """Gracefully flushes remaining items and stops background writer."""
        self.flush()
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    # -------------------------------------------------------------
    # Query APIs for Dashboard, Reconciliation & Alpha Analytics
    # -------------------------------------------------------------
    def get_trades(self, alpha_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Queries trade ledger with full attribution."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            if alpha_id:
                rows = conn.execute(
                    "SELECT * FROM trade_ledger WHERE alpha_id = ? ORDER BY trade_id DESC LIMIT ?",
                    (alpha_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trade_ledger ORDER BY trade_id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_trade_drilldown(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Returns full contextual drilldown answering: 'Why did Ashva do this trade?'
        Lineage: Trade -> Fill -> Order -> Decision -> Signal.
        """
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            trade_row = conn.execute("SELECT * FROM trade_ledger WHERE trade_id = ?", (trade_id,)).fetchone()
            if not trade_row:
                return None
            
            trade = dict(trade_row)
            decision_id = trade.get("decision_id")
            signal_id = trade.get("signal_id")
            order_id = trade.get("order_id")

            decision = None
            if decision_id:
                d_row = conn.execute("SELECT * FROM decisions_log WHERE decision_id = ?", (decision_id,)).fetchone()
                if d_row:
                    decision = dict(d_row)

            signal = None
            if signal_id:
                s_row = conn.execute("SELECT * FROM signals_log WHERE signal_id = ?", (signal_id,)).fetchone()
                if s_row:
                    signal = dict(s_row)

            order = None
            if order_id:
                o_row = conn.execute("SELECT * FROM orders_log WHERE order_id = ?", (order_id,)).fetchone()
                if o_row:
                    order = dict(o_row)

            return {
                "trade": trade,
                "decision": decision,
                "signal": signal,
                "order": order,
            }
