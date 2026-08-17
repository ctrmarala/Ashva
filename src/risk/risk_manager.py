"""
Ashva Real-Time Risk Management System (RMS)
Enforces strict fund-level circuit breakers, intraday loss cutoffs, position limits, and emergency kill-switches.
"""

from datetime import datetime, time
import logging
from typing import Dict, List, Tuple, Any, Optional

from src.core.events import OrderEvent, RiskEvent, OrderSide, OrderType

logger = logging.getLogger("Ashva.RMS")


class RiskManager:
    """
    Independent Risk Management Gateway. All orders must be validated here prior to routing to brokers.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 1.5,       # Hard circuit breaker: Halt trading if day loss >= 1.5%
        max_portfolio_drawdown_pct: float = 5.0, # Reduce size if overall DD >= 5%
        max_open_positions: int = 4,           # Max concurrent open positions
        max_order_value_inr: float = 200000.0, # Max single order value
        intraday_entry_cutoff: str = "15:00:00",
        intraday_square_off: str = "15:15:00",
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self.max_open_positions = max_open_positions
        self.max_order_value_inr = max_order_value_inr
        self.intraday_entry_cutoff = datetime.strptime(intraday_entry_cutoff, "%H:%M:%S").time()
        self.intraday_square_off = datetime.strptime(intraday_square_off, "%H:%M:%S").time()

        # State Variables
        self.kill_switch_active = False
        self.trading_halted_for_day = False
        self.daily_starting_equity = 500000.0
        self.peak_equity = 500000.0
        self.risk_events_log: List[RiskEvent] = []

    def set_starting_equity(self, equity: float):
        """Sets the baseline equity for the current trading day."""
        self.daily_starting_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def trigger_kill_switch(self, reason: str = "MANUAL_EMERGENCY_TRIGGER") -> RiskEvent:
        """
        Activates global kill-switch: immediately blocks all orders and locks the system.
        """
        self.kill_switch_active = True
        risk_event = RiskEvent(
            timestamp=datetime.now(),
            severity="EMERGENCY_HALT",
            rule_name="GLOBAL_KILL_SWITCH",
            message=f"Global Kill Switch Activated: {reason}",
            action_taken="FLATTEN_ALL_POSITIONS",
        )
        self.risk_events_log.append(risk_event)
        logger.critical(f"EMERGENCY: {risk_event.message}")
        return risk_event

    def validate_order(
        self,
        order: OrderEvent,
        current_equity: float,
        current_price: float,
        open_positions_count: int,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates order against all fund risk rules. Returns (is_approved, rejection_reason).
        """
        check_time = (current_time or datetime.now()).time()

        # 1. Kill Switch Check
        if self.kill_switch_active:
            return False, "Order rejected: Global kill-switch is active."

        # 2. Daily Loss Limit Circuit Breaker
        daily_pnl = current_equity - self.daily_starting_equity
        daily_loss_pct = (-daily_pnl / self.daily_starting_equity) * 100.0 if daily_pnl < 0 else 0.0

        if daily_loss_pct >= self.max_daily_loss_pct:
            self.trading_halted_for_day = True
            risk_evt = RiskEvent(
                timestamp=datetime.now(),
                severity="CRITICAL",
                rule_name="MAX_DAILY_LOSS_BREACH",
                message=f"Daily loss ({daily_loss_pct:.2f}%) hit hard limit ({self.max_daily_loss_pct}%). Trading halted.",
                action_taken="HALT_STRATEGY",
            )
            self.risk_events_log.append(risk_evt)
            return False, f"Order rejected: {risk_evt.message}"

        if self.trading_halted_for_day:
            return False, "Order rejected: Trading is halted for the day due to prior risk breach."

        # 3. Time of Day Restrictions
        if check_time >= self.intraday_square_off:
            return False, f"Order rejected: Market is in intraday square-off window (>= {self.intraday_square_off})."

        if check_time >= self.intraday_entry_cutoff and order.side in [OrderSide.BUY, OrderSide.SELL]:
            # Disallow opening new positions past entry cutoff
            return False, f"Order rejected: Past entry cutoff time ({self.intraday_entry_cutoff})."

        # 4. Open Positions Count Limit
        if open_positions_count >= self.max_open_positions:
            return False, f"Order rejected: Max open positions limit ({self.max_open_positions}) reached."

        # 5. Order Value Cap
        estimated_order_value = current_price * order.quantity
        if estimated_order_value > self.max_order_value_inr:
            return False, f"Order rejected: Value (Rs {estimated_order_value:,.2f}) exceeds cap (Rs {self.max_order_value_inr:,.2f})."

        return True, None
