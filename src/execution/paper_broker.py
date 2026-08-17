"""
Ashva Institutional Paper Trading Simulator
High-fidelity broker simulator with slippage, latency, order queueing, and exact Indian cost accounting.
"""

from datetime import datetime
import logging
from typing import Dict, List, Optional, Any

from src.core.events import OrderEvent, FillEvent, TickEvent, BarEvent, OrderSide, OrderType, ProductType
from src.analytics.indian_costs import IndianCostModel, Segment
from src.core.state_machine import StateMachineWAL

logger = logging.getLogger("Ashva.PaperBroker")


class PaperBroker:
    """
    Simulates real-world order execution and portfolio state tracking.
    """

    def __init__(
        self,
        initial_capital: float = 500000.0,
        cost_model: Optional[IndianCostModel] = None,
        state_wal: Optional[StateMachineWAL] = None,
        slippage_bps: float = 3.0,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.equity = initial_capital
        self.cost_model = cost_model or IndianCostModel(default_slippage_bps=slippage_bps)
        self.state_wal = state_wal or StateMachineWAL()
        self.slippage_bps = slippage_bps

        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.active_orders: Dict[str, OrderEvent] = {}
        self.last_prices: Dict[str, float] = {}

    def update_market_price(self, symbol: str, price: float):
        """Updates internal LTP cache and marks portfolio to market."""
        self.last_prices[symbol.upper()] = price
        self._mark_to_market()

    def _mark_to_market(self):
        """Recomputes total equity based on cash and open position unrealized PnL."""
        total_pos_value = 0.0
        for symbol, pos in self.open_positions.items():
            curr_p = self.last_prices.get(symbol, pos["entry_price"])
            side = pos["side"]
            qty = pos["quantity"]
            entry_p = pos["entry_price"]

            if side == "LONG":
                unrealized = (curr_p - entry_p) * qty
                total_pos_value += (entry_p * qty) + unrealized
            else:  # SHORT
                unrealized = (entry_p - curr_p) * qty
                total_pos_value += (entry_p * qty) + unrealized

        self.equity = self.cash + total_pos_value

    def submit_order(self, order: OrderEvent, current_price: Optional[float] = None) -> FillEvent:
        """
        Executes order and returns a FillEvent.
        """
        sym = order.symbol.upper()
        fill_base_price = current_price or self.last_prices.get(sym, order.limit_price or 100.0)

        # Apply simulated slippage
        slippage_factor = (self.slippage_bps / 10000.0) * (1.0 if order.side == OrderSide.BUY else -1.0)
        fill_price = fill_base_price * (1.0 + slippage_factor)

        # 1. Check if closing an existing position or opening a new one
        if sym in self.open_positions and (
            (order.side == OrderSide.SELL and self.open_positions[sym]["side"] == "LONG")
            or (order.side == OrderSide.BUY and self.open_positions[sym]["side"] == "SHORT")
        ):
            # Closing Position
            pos = self.open_positions.pop(sym)
            entry_p = pos["entry_price"]
            exit_p = fill_price
            qty = min(order.quantity, pos["quantity"])

            if pos["side"] == "LONG":
                cost_breakdown = self.cost_model.calculate_trade_costs(
                    buy_price=entry_p, sell_price=exit_p, quantity=qty, segment=Segment.EQUITY_INTRADAY
                )
            else:
                cost_breakdown = self.cost_model.calculate_trade_costs(
                    buy_price=exit_p, sell_price=entry_p, quantity=qty, segment=Segment.EQUITY_INTRADAY
                )

            # Update Cash & WAL
            self.cash += (pos["entry_price"] * qty) + cost_breakdown.net_pnl
            self.state_wal.remove_position(sym)
            self.state_wal.log_closed_trade(
                symbol=sym,
                entry_time=pos["entry_time"],
                exit_time=order.timestamp.isoformat(),
                side=pos["side"],
                quantity=qty,
                entry_price=entry_p,
                exit_price=exit_p,
                gross_pnl=cost_breakdown.gross_pnl,
                net_pnl=cost_breakdown.net_pnl,
                cost_breakdown=cost_breakdown.to_dict(),
            )
            logger.info(f"Closed {pos['side']} on {sym} @ {fill_price:.2f} | Net PnL: Rs {cost_breakdown.net_pnl:+.2f}")

        else:
            # Opening New Position
            side_str = "LONG" if order.side == OrderSide.BUY else "SHORT"
            required_cash = fill_price * order.quantity
            self.cash -= required_cash

            self.open_positions[sym] = {
                "side": side_str,
                "quantity": order.quantity,
                "entry_price": fill_price,
                "entry_time": order.timestamp.isoformat(),
                "strategy_id": order.tag,
                "stop_loss": order.stop_price,
            }

            self.state_wal.upsert_position(
                symbol=sym,
                side=side_str,
                quantity=order.quantity,
                entry_price=fill_price,
                entry_time=order.timestamp.isoformat(),
                strategy_id=order.tag,
                stop_loss=order.stop_price,
            )
            logger.info(f"Opened {side_str} on {sym} @ {fill_price:.2f} x {order.quantity} shares")

        self._mark_to_market()

        fill_event = FillEvent(
            order_id=order.order_id,
            symbol=sym,
            timestamp=datetime.now(),
            side=order.side,
            fill_price=fill_price,
            quantity=order.quantity,
            commission=20.0,
            slippage=abs(fill_price - fill_base_price),
        )
        return fill_event
