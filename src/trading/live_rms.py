"""
Ashva Institutional Live Risk Manager (Live RMS)
Enforces strict fund-level circuit breakers, intraday loss cutoffs, position limits,
sector limits, and emergency kill-switches.
CRITICAL MANDATE: Exit and position-reducing orders are NEVER blocked by limits or circuit breakers.
"""

from datetime import datetime, time
import logging
from typing import Dict, List, Tuple, Any, Optional

from src.core.events import OrderIntent, OrderSide
from src.trading.position_manager import PositionManager
from src.trading.portfolio_state import PortfolioState

logger = logging.getLogger("Ashva.LiveRMS")

SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "TECHM": "IT", "WIPRO": "IT",
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "INDUSINDBK": "BANKING", "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE",
    "SHRIRAMFIN": "FINANCE", "HDFCLIFE": "FINANCE", "SBILIFE": "FINANCE",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "TMPV": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "RELIANCE": "ENERGY", "NTPC": "ENERGY", "ONGC": "ENERGY", "POWERGRID": "ENERGY",
    "COALINDIA": "ENERGY", "BPCL": "ENERGY",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA", "DIVISLAB": "PHARMA",
    "APOLLOHOSP": "PHARMA",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "GRASIM": "MATERIALS", "ULTRACEMCO": "MATERIALS",
    "PIDILITIND": "MATERIALS",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "TATACONSUM": "FMCG", "TITAN": "CONSUMER", "TRENT": "RETAIL",
    "LT": "INFRA", "ADANIENT": "INFRA", "ADANIPORTS": "INFRA", "BEL": "DEFENSE",
}


class LiveRiskManager:
    """
    Real-time risk manager validating all OrderIntents before adapter submission.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 1.5,
        max_portfolio_drawdown_pct: float = 5.0,
        max_concurrent_positions: int = 5,
        max_positions_per_sector: int = 2,
        entry_start_time: time = time(9, 15),
        entry_end_time: time = time(15, 0),
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_positions_per_sector = max_positions_per_sector
        self.entry_start_time = entry_start_time
        self.entry_end_time = entry_end_time
        
        self.kill_switch_active = False

    def validate_order(
        self,
        intent: OrderIntent,
        current_price: float,
        position_manager: PositionManager,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates an order against institutional risk constraints.
        Returns (is_approved, reject_reason).
        """
        # RULE 1: Exit / Position-Reducing Orders are NEVER BLOCKED
        sym = intent.symbol.upper()
        existing_pos = position_manager.get_position(sym)
        is_closing_order = (
            intent.is_reduce_only
            or (existing_pos is not None and (
                (intent.side == OrderSide.SELL and existing_pos.side == OrderSide.BUY)
                or (intent.side == OrderSide.BUY and existing_pos.side == OrderSide.SELL)
            ))
        )

        if is_closing_order:
            return True, None

        # -------------------------------------------------------------
        # ENTRY RISK CHECKS
        # -------------------------------------------------------------
        if self.kill_switch_active:
            return False, "RMS: Kill-switch active. All new entries blocked."

        # 1. Daily Loss Circuit Breaker
        if portfolio_state.get_daily_loss_pct() >= self.max_daily_loss_pct:
            return False, f"RMS: Daily loss cutoff ({self.max_daily_loss_pct}%) breached."

        # 2. Portfolio Max Drawdown Circuit Breaker
        if portfolio_state.get_drawdown_pct() >= self.max_portfolio_drawdown_pct:
            return False, f"RMS: Portfolio drawdown cutoff ({self.max_portfolio_drawdown_pct}%) breached."

        # 3. Trading Hours Window Check
        order_time = intent.timestamp.time() if hasattr(intent.timestamp, "time") else None
        if order_time is not None:
            if order_time < self.entry_start_time or order_time > self.entry_end_time:
                return False, f"RMS: Order outside trading window ({self.entry_start_time} - {self.entry_end_time})."

        # 4. Duplicate Position Check
        if existing_pos is not None:
            return False, f"RMS: Open position already exists in {sym}."

        # 5. Max Concurrent Positions Cap
        open_count = len(position_manager.open_positions)
        if open_count >= self.max_concurrent_positions:
            return False, f"RMS: Max concurrent positions ({self.max_concurrent_positions}) reached."

        # 6. Sector Concentration Limit
        sector = SECTOR_MAP.get(sym, "OTHER")
        sector_count = sum(1 for p in position_manager.open_positions.values() if SECTOR_MAP.get(p.symbol, "OTHER") == sector)
        if sector_count >= self.max_positions_per_sector:
            return False, f"RMS: Max positions for sector '{sector}' ({self.max_positions_per_sector}) reached."

        # 7. Maximum Capital Cap per Trade (Max 20% of MTM equity)
        max_capital = portfolio_state.current_equity * 0.20
        order_value = current_price * intent.quantity
        if order_value > (max_capital * 1.05):  # 5% buffer
            return False, f"RMS: Order value (Rs {order_value:.2f}) exceeds max allocation (Rs {max_capital:.2f})."

        return True, None
