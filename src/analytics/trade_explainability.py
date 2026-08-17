"""
Ashva Institutional Trade Explainability & Rationale Ledger
Provides complete transparency into the quantitative decision lifecycle for every executed order.
Answers: Why did we enter? Why this size? Why did we exit? What was the post-trade MFE/MAE?
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Dict, List, Any, Optional


@dataclass
class TradeExplanationRecord:
    trade_id: str
    symbol: str
    entry_time: str
    exit_time: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    
    # Quantitative Decision Attribution
    strategy_id: str
    entry_rationale: str       # e.g., "ORB Breakout above 9:30 AM Range (Rs 1305.50) + VWAP Support + CVD Delta > 1.8 sigma"
    sizing_rationale: str      # e.g., "0.40% Portfolio Risk (Rs 2000) / Stop Dist Rs 14.50 = 138 shares"
    exit_rationale: str        # e.g., "Target 2.5R Achieved @ Rs 1341.75" or "Stop Loss Hit @ Rs 1291.00"
    market_regime: str         # e.g., "HIGH_VOLATILITY_TRENDING"
    
    # Excursion Diagnostics
    mfe_pct: float             # Maximum Favorable Excursion (% gain reached before exit)
    mae_pct: float             # Maximum Adverse Excursion (% drawdown experienced during trade)
    slippage_cost_inr: float   # Execution slippage relative to theoretical signal price
    taxes_and_fees_inr: float  # Angel One & Indian statutory fees


class TradeExplainabilityEngine:
    """
    Audit and explainability repository for institutional fund compliance and strategy refinement.
    """

    def __init__(self, db_path: str = "data_lake/trade_explainability.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_explanations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    exit_price DOUBLE NOT NULL,
                    gross_pnl DOUBLE NOT NULL,
                    net_pnl DOUBLE NOT NULL,
                    strategy_id TEXT NOT NULL,
                    entry_rationale TEXT NOT NULL,
                    sizing_rationale TEXT NOT NULL,
                    exit_rationale TEXT NOT NULL,
                    market_regime TEXT NOT NULL,
                    mfe_pct DOUBLE NOT NULL,
                    mae_pct DOUBLE NOT NULL,
                    slippage_cost_inr DOUBLE NOT NULL,
                    taxes_and_fees_inr DOUBLE NOT NULL
                );
            """)

    def log_explanation(self, record: TradeExplanationRecord):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trade_explanations (
                    trade_id, symbol, entry_time, exit_time, side, quantity,
                    entry_price, exit_price, gross_pnl, net_pnl, strategy_id,
                    entry_rationale, sizing_rationale, exit_rationale, market_regime,
                    mfe_pct, mae_pct, slippage_cost_inr, taxes_and_fees_inr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.trade_id, record.symbol, record.entry_time, record.exit_time,
                record.side, record.quantity, record.entry_price, record.exit_price,
                record.gross_pnl, record.net_pnl, record.strategy_id,
                record.entry_rationale, record.sizing_rationale, record.exit_rationale,
                record.market_regime, record.mfe_pct, record.mae_pct,
                record.slippage_cost_inr, record.taxes_and_fees_inr,
            ))

    def get_explanations_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trade_explanations WHERE symbol = ? ORDER BY id DESC", (symbol,)).fetchall()
            return [dict(r) for r in rows]
