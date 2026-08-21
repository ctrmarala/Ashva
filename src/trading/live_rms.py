"""
Ashva Institutional Live Risk Manager (Live RMS)
Enforces hierarchical kill-switches, automatic safety state transitions, fund-level circuit breakers,
intraday loss cutoffs, position caps, and sector limits.
CRITICAL MANDATE: Exit and position-reducing orders are NEVER blocked by limits, circuit breakers, or safety halts.
"""

from datetime import datetime, time, timedelta
from enum import Enum
import logging
from typing import Dict, List, Tuple, Any, Optional, Set

from src.core.events import OrderIntent, OrderSide, RiskEvent
from src.trading.position_manager import PositionManager
from src.trading.portfolio_state import PortfolioState

logger = logging.getLogger("Ashva.LiveRMS")


class SafetyState(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    HALTED = "HALTED"
    CLOSING = "CLOSING"
    EMERGENCY = "EMERGENCY"


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
    Real-time hierarchical risk manager validating all OrderIntents.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 1.5,
        max_portfolio_drawdown_pct: float = 5.0,
        max_concurrent_positions: int = 5,
        max_positions_per_sector: int = 2,
        max_single_position_capital_pct: float = 0.20,
        entry_start_time: time = time(9, 15),
        entry_end_time: time = time(15, 0),
        max_stale_data_seconds: float = 180.0,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_positions_per_sector = max_positions_per_sector
        self.max_single_position_capital_pct = max_single_position_capital_pct
        self.entry_start_time = entry_start_time
        self.entry_end_time = entry_end_time
        self.max_stale_data_seconds = max_stale_data_seconds

        # Hierarchical Controls
        self.safety_state = SafetyState.ACTIVE
        self.disabled_alphas: Set[str] = set()
        self.disabled_symbols: Set[str] = set()
        self.disabled_strategy_groups: Set[str] = set()
        self.risk_events: List[RiskEvent] = []

    @property
    def kill_switch_active(self) -> bool:
        return self.safety_state in [SafetyState.HALTED, SafetyState.EMERGENCY, SafetyState.CLOSING]

    def trigger_kill_switch(self, reason: str = "MANUAL_TRIGGER", severity: str = "CRITICAL"):
        """Activates global emergency kill-switch."""
        self.safety_state = SafetyState.HALTED
        ev = RiskEvent(
            timestamp=datetime.now(),
            severity=severity,
            rule_name="GLOBAL_KILL_SWITCH",
            message=f"Global Kill Switch Activated: {reason}",
            action_taken="HALT_ALL_ENTRIES",
        )
        self.risk_events.append(ev)
        logger.critical(f"EMERGENCY: Global Kill Switch Activated: {reason}")
        return ev

    def reset_kill_switch(self, reason: str = "MANUAL_RESET"):
        """Resets safety state to ACTIVE."""
        self.safety_state = SafetyState.ACTIVE
        logger.info(f"RMS: Safety state reset to ACTIVE. Reason: {reason}")

    def disable_alpha(self, alpha_id: str, reason: str = "MANUAL_DISABLE"):
        """Disables entries for a specific alpha."""
        self.disabled_alphas.add(alpha_id)
        logger.warning(f"RMS: Alpha {alpha_id} disabled. Reason: {reason}")

    def enable_alpha(self, alpha_id: str):
        """Re-enables a disabled alpha."""
        self.disabled_alphas.discard(alpha_id)
        logger.info(f"RMS: Alpha {alpha_id} re-enabled.")

    def disable_symbol(self, symbol: str, reason: str = "MANUAL_DISABLE"):
        """Disables entries for a specific symbol."""
        self.disabled_symbols.add(symbol.upper())
        logger.warning(f"RMS: Symbol {symbol} disabled. Reason: {reason}")

    def enable_symbol(self, symbol: str):
        """Re-enables a disabled symbol."""
        self.disabled_symbols.discard(symbol.upper())
        logger.info(f"RMS: Symbol {symbol} re-enabled.")

    def validate_order(
        self,
        intent: OrderIntent,
        current_price: float,
        position_manager: PositionManager,
        portfolio_state: PortfolioState,
        last_market_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates an order against institutional risk constraints.
        Returns (is_approved, reject_reason).
        """
        sym = intent.symbol.upper()
        existing_pos = position_manager.get_position(sym)
        
        # RULE 1: Exit / Position-Reducing Orders are NEVER BLOCKED
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
        # 1. Safety State Check
        if self.safety_state in [SafetyState.HALTED, SafetyState.EMERGENCY, SafetyState.CLOSING]:
            return False, f"RMS: Safety state is {self.safety_state.value}. All new entries blocked."

        if self.safety_state == SafetyState.PAUSED:
            return False, "RMS: Safety state is PAUSED. Entries temporarily suspended."

        # 2. Hierarchical Target Disables
        if intent.strategy_id in self.disabled_alphas:
            return False, f"RMS: Alpha {intent.strategy_id} is disabled by risk policy."

        if sym in self.disabled_symbols:
            return False, f"RMS: Symbol {sym} is quarantined/disabled by risk policy."

        # 3. Daily Loss Circuit Breaker
        if portfolio_state.get_daily_loss_pct() >= self.max_daily_loss_pct:
            return False, f"RMS: Daily loss cutoff ({self.max_daily_loss_pct}%) breached."

        # 4. Portfolio Max Drawdown Circuit Breaker
        if portfolio_state.get_drawdown_pct() >= self.max_portfolio_drawdown_pct:
            return False, f"RMS: Portfolio drawdown cutoff ({self.max_portfolio_drawdown_pct}%) breached."

        # 5. Stale Market Data Check
        if last_market_time is not None and hasattr(intent.timestamp, "timestamp"):
            data_age = (intent.timestamp - last_market_time).total_seconds()
            if data_age > self.max_stale_data_seconds:
                return False, f"RMS: Stale market data detected (Age={data_age:.1f}s > {self.max_stale_data_seconds}s)."

        # 6. Trading Hours Window Check
        order_time = intent.timestamp.time() if hasattr(intent.timestamp, "time") else None
        if order_time is not None:
            if order_time < self.entry_start_time or order_time > self.entry_end_time:
                return False, f"RMS: Order outside permitted entry window ({self.entry_start_time.strftime('%H:%M')} - {self.entry_end_time.strftime('%H:%M')})."

        # 7. Max Concurrent Positions Cap
        open_positions = position_manager.get_all_positions()
        if len(open_positions) >= self.max_concurrent_positions:
            return False, f"RMS: Max concurrent positions limit ({self.max_concurrent_positions}) reached."

        # 8. Sector Exposure Cap
        target_sector = SECTOR_MAP.get(sym, "UNKNOWN")
        if target_sector != "UNKNOWN":
            sector_count = sum(1 for p in open_positions if SECTOR_MAP.get(p.symbol, "UNKNOWN") == target_sector)
            if sector_count >= self.max_positions_per_sector:
                return False, f"RMS: Sector exposure limit for '{target_sector}' ({self.max_positions_per_sector} positions) reached."

        # 9. Single Position Capital Allocation Cap
        order_val = current_price * intent.quantity
        max_allowed_val = portfolio_state.current_equity * self.max_single_position_capital_pct
        if order_val > (max_allowed_val + 1e-4):
            return False, f"RMS: Order value (Rs {order_val:,.2f}) exceeds {self.max_single_position_capital_pct*100}% capital cap (Rs {max_allowed_val:,.2f})."

        return True, None
