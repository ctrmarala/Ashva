"""
Ashva Production Portfolio State Tracker
Maintains real-time account balances, realized PnL, cash reserves, continuous Mark-to-Market equity,
peak equity watermarks, and intraday drawdowns.
"""

from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.analytics.metrics import calculate_daily_mtm_sharpe, calculate_daily_mtm_sortino, calculate_max_drawdown_pct


class PortfolioState:
    """
    Central financial state tracker for TradingEngine.
    """

    def __init__(self, initial_capital: float = 500000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.daily_starting_equity = initial_capital
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        self.equity_history: List[Dict[str, Any]] = []

    def update_mtm(self, unrealized_pnl: float, timestamp: datetime):
        """Updates continuous MTM equity and peak watermark."""
        self.unrealized_pnl = unrealized_pnl
        self.current_equity = self.cash + unrealized_pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)

        self.equity_history.append({
            "timestamp": timestamp,
            "cash": round(self.cash, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "equity": round(self.current_equity, 2),
        })

    def on_trade_closed(self, net_pnl: float, timestamp: datetime):
        """Releases trade PnL back into cash reserves upon exit."""
        self.cash += net_pnl
        self.realized_pnl += net_pnl
        self.current_equity = self.cash + self.unrealized_pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)

    def get_drawdown_pct(self) -> float:
        """Computes current drawdown from all-time peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity * 100.0)

    def get_daily_loss_pct(self) -> float:
        """Computes current loss from daily starting equity."""
        if self.daily_starting_equity <= 0:
            return 0.0
        return max(0.0, (self.daily_starting_equity - self.current_equity) / self.daily_starting_equity * 100.0)

    def get_summary(self) -> Dict[str, Any]:
        """Calculates comprehensive performance metrics across simulation."""
        df_eq = pd.DataFrame(self.equity_history)
        if not df_eq.empty:
            df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
            eq_series = df_eq.set_index("timestamp")["equity"]
            daily_sharpe = calculate_daily_mtm_sharpe(eq_series)
            daily_sortino = calculate_daily_mtm_sortino(eq_series)
            max_dd = calculate_max_drawdown_pct(eq_series)
        else:
            daily_sharpe = 0.0
            daily_sortino = 0.0
            max_dd = 0.0

        total_net_pnl = self.current_equity - self.initial_capital
        roi_pct = (total_net_pnl / self.initial_capital) * 100.0

        return {
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.current_equity, 2),
            "cash": round(self.cash, 2),
            "total_net_pnl": round(total_net_pnl, 2),
            "roi_pct": round(roi_pct, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "daily_sharpe": round(daily_sharpe, 2),
            "daily_sortino": round(daily_sortino, 2),
            "max_drawdown_pct": round(max_dd, 2),
        }
