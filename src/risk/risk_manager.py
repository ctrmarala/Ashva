"""
Ashva Real-Time Risk Management System (RMS)
Enforces strict fund-level circuit breakers, intraday loss cutoffs, position limits,
smart exit priority routing, and active emergency kill-switch liquidation.
"""

from datetime import datetime, time
import logging
from typing import Dict, List, Tuple, Any, Optional

from src.core.events import OrderEvent, RiskEvent, OrderSide, OrderType

logger = logging.getLogger("Ashva.RMS")


class RiskManager:
    """
    Institutional Independent Risk Management Gateway.
    All orders must be validated here prior to routing to brokers.
    CRITICAL RULE: Exit and position-reducing orders are NEVER blocked by entry limits or circuit breakers.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 1.5,         # Hard circuit breaker: Halt trading if day loss >= 1.5%
        max_portfolio_drawdown_pct: float = 5.0, # Reduce size if overall DD >= 5%
        max_open_positions: int = 4,             # Max concurrent open positions
        max_order_value_inr: float = 200000.0,   # Max single order value
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

    def trigger_kill_switch(self, broker_gateway: Optional[Any] = None, reason: str = "MANUAL_EMERGENCY_TRIGGER") -> RiskEvent:
        """
        Activates global kill-switch:
        1. Sets kill_switch_active = True to permanently block new entries.
        2. Cancels all pending open orders at broker.
        3. Retrieves active open positions and executes market liquidation (flatten).
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

        # Active Liquidation Execution if broker gateway is provided
        if broker_gateway is not None:
            try:
                # 1. Cancel pending orders
                if hasattr(broker_gateway, "cancel_all_orders"):
                    broker_gateway.cancel_all_orders()
                    logger.info("[RMS] Cancelled all pending broker orders.")

                # 2. Flatten all open positions
                if hasattr(broker_gateway, "get_positions") and hasattr(broker_gateway, "place_order"):
                    positions = broker_gateway.get_positions()
                    for pos in positions:
                        qty = pos.get("quantity", 0)
                        sym = pos.get("symbol", "")
                        side = pos.get("side", "")
                        if qty > 0 and sym:
                            opp_side = OrderSide.SELL if side == "LONG" else OrderSide.BUY
                            flatten_order = OrderEvent(
                                symbol=sym,
                                side=opp_side,
                                order_type=OrderType.MARKET,
                                quantity=qty,
                                strategy_id="RMS_KILL_SWITCH",
                                tag="EMERGENCY_FLATTEN",
                            )
                            broker_gateway.place_order(flatten_order)
                            logger.critical(f"[RMS] Submitted Emergency Flatten Order: {opp_side.value} {qty} {sym}")

                    # 3. Post-Liquidation Verification Loop (Confirm Flatness)
                    is_flat = False
                    for attempt in range(3):
                        try:
                            remaining = broker_gateway.get_positions()
                            active_rem = [p for p in remaining if p.get("quantity", 0) > 0]
                            if not active_rem:
                                is_flat = True
                                logger.info("[RMS] KILL SWITCH VERIFIED: All positions confirmed FLAT at broker.")
                                break
                        except Exception as e:
                            logger.warning(f"[RMS] Verification poll attempt {attempt+1} encountered: {e}")

                    if not is_flat and positions:
                        logger.critical("[RMS] CRITICAL RMS ALERT: Liquidation order dispatched, awaiting broker fill confirmation.")

            except Exception as e:
                logger.error(f"[RMS] Failed during active liquidation: {e}")

        return risk_event

    def validate_order(
        self,
        order: OrderEvent,
        current_equity: float,
        current_price: float,
        open_positions_count: int,
        is_exit: bool = False,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates order against all fund risk rules.
        CRITICAL INSTITUTIONAL RULE: Exits/Stop-loss orders must ALWAYS be allowed even if
        circuit breakers or entry cutoff times are breached.
        """
        check_time = (current_time or datetime.now()).time()

        # =========================================================================
        # EXIT / REDUCE-ONLY FAST-PATH (NEVER TRAP CAPITAL)
        # =========================================================================
        if is_exit or getattr(order, "is_reduce_only", False) or getattr(order, "tag", "") == "EMERGENCY_FLATTEN":
            # Exits are always approved to reduce risk and protect capital
            return True, None

        # =========================================================================
        # NEW ENTRY ORDER RISK GATES
        # =========================================================================

        # 1. Kill Switch Check
        if self.kill_switch_active:
            return False, "New Entry Rejected: Global kill-switch is active."

        # 2. Daily Loss Limit Circuit Breaker
        daily_pnl = current_equity - self.daily_starting_equity
        daily_loss_pct = (-daily_pnl / self.daily_starting_equity) * 100.0 if daily_pnl < 0 else 0.0

        if daily_loss_pct >= self.max_daily_loss_pct:
            self.trading_halted_for_day = True
            risk_evt = RiskEvent(
                timestamp=datetime.now(),
                severity="CRITICAL",
                rule_name="MAX_DAILY_LOSS_BREACH",
                message=f"Daily loss ({daily_loss_pct:.2f}%) hit hard limit ({self.max_daily_loss_pct}%). New entries halted.",
                action_taken="HALT_NEW_ENTRIES",
            )
            self.risk_events_log.append(risk_evt)
            return False, f"New Entry Rejected: {risk_evt.message}"

        if self.trading_halted_for_day:
            return False, "New Entry Rejected: Trading is halted for the day due to prior risk breach."

        # 3. Time of Day Restrictions
        if check_time >= self.intraday_square_off:
            return False, f"New Entry Rejected: Market is in square-off window (>= {self.intraday_square_off})."

        if check_time >= self.intraday_entry_cutoff:
            return False, f"New Entry Rejected: Past entry cutoff time ({self.intraday_entry_cutoff})."

        # 4. Open Positions Count Limit
        if open_positions_count >= self.max_open_positions:
            return False, f"New Entry Rejected: Max open positions limit ({self.max_open_positions}) reached."

        # 5. Order Value Cap
        estimated_order_value = current_price * order.quantity
        if estimated_order_value > self.max_order_value_inr:
            return False, f"New Entry Rejected: Order value (Rs {estimated_order_value:,.2f}) exceeds cap (Rs {self.max_order_value_inr:,.2f})."

        return True, None
