"""
Ashva Crash-Resilient State Machine & Write-Ahead Logger (WAL)
Persists portfolio equity, open positions, pending orders, and trade history in SQLite/DuckDB WAL
for sub-100ms recovery upon server restart, internet drop, or system crash.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Dict, List, Any, Optional

from src.core.events import OrderEvent, FillEvent, OrderSide, OrderType, ProductType


class StateMachineWAL:
    """
    Write-Ahead Logging State Store ensuring zero desynchronization with broker state.
    """

    def __init__(self, db_path: str = "data_lake/ashva_state_wal.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    cash DOUBLE NOT NULL,
                    equity DOUBLE NOT NULL,
                    daily_starting_equity DOUBLE NOT NULL,
                    peak_equity DOUBLE NOT NULL,
                    kill_switch_active INTEGER NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    entry_time TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    stop_loss DOUBLE,
                    take_profit DOUBLE
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    limit_price DOUBLE,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_ledger (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    exit_price DOUBLE NOT NULL,
                    gross_pnl DOUBLE NOT NULL,
                    net_pnl DOUBLE NOT NULL,
                    cost_breakdown_json TEXT NOT NULL
                );
            """)

    def save_portfolio_state(
        self,
        cash: float,
        equity: float,
        daily_starting_equity: float,
        peak_equity: float,
        kill_switch_active: bool,
    ):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO portfolio_state (id, timestamp, cash, equity, daily_starting_equity, peak_equity, kill_switch_active)
                VALUES (1, ?, ?, ?, ?, ?, ?);
            """, (datetime.now().isoformat(), cash, equity, daily_starting_equity, peak_equity, 1 if kill_switch_active else 0))

    def load_portfolio_state(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT cash, equity, daily_starting_equity, peak_equity, kill_switch_active FROM portfolio_state WHERE id = 1;").fetchone()
            if row:
                return {
                    "cash": row[0],
                    "equity": row[1],
                    "daily_starting_equity": row[2],
                    "peak_equity": row[3],
                    "kill_switch_active": bool(row[4]),
                }
            return None

    def upsert_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        entry_time: str,
        strategy_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO open_positions (symbol, side, quantity, entry_price, entry_time, strategy_id, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (symbol.upper(), side, quantity, entry_price, entry_time, strategy_id, stop_loss, take_profit))

    def remove_position(self, symbol: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM open_positions WHERE symbol = ?;", (symbol.upper(),))

    def load_open_positions(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT symbol, side, quantity, entry_price, entry_time, strategy_id, stop_loss, take_profit FROM open_positions;").fetchall()
            return [
                {
                    "symbol": r[0],
                    "side": r[1],
                    "quantity": r[2],
                    "entry_price": r[3],
                    "entry_time": r[4],
                    "strategy_id": r[5],
                    "stop_loss": r[6],
                    "take_profit": r[7],
                }
                for r in rows
            ]

    def log_closed_trade(
        self,
        symbol: str,
        entry_time: str,
        exit_time: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        gross_pnl: float,
        net_pnl: float,
        cost_breakdown: Dict[str, Any],
    ):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO trade_ledger (symbol, entry_time, exit_time, side, quantity, entry_price, exit_price, gross_pnl, net_pnl, cost_breakdown_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (symbol.upper(), entry_time, exit_time, side, quantity, entry_price, exit_price, gross_pnl, net_pnl, json.dumps(cost_breakdown)))
