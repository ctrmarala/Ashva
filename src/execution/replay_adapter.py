"""
Ashva Replay Execution Adapter
Simulates institutional broker execution matching on historical market data.
Implements next-bar open execution, intrabar SL/TP barrier triggers, and exact Indian cost modeling.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from src.core.events import (
    OrderIntent, OrderEvent, FillEvent, MarketEvent,
    OrderSide, OrderType, OrderStatus, ProductType,
)
from src.analytics.indian_costs import IndianCostModel, Segment
from src.execution.adapter import ExecutionAdapter


class ReplayExecutionAdapter(ExecutionAdapter):
    """
    High-fidelity historical execution adapter.
    Translates OrderIntents into fills based on next-bar open and intrabar high/low extremes.
    """

    def __init__(
        self,
        cost_model: Optional[IndianCostModel] = None,
        segment: Segment = Segment.EQUITY_INTRADAY,
        slippage_bps: float = 3.0,
    ):
        self.cost_model = cost_model or IndianCostModel(default_slippage_bps=slippage_bps)
        self.segment = segment
        self.slippage_bps = slippage_bps
        
        self._pending_orders: List[OrderEvent] = []
        self._active_barriers: Dict[str, Dict[str, Any]] = {}  # symbol -> {stop_loss, take_profit, side, quantity, strategy_id, entry_price}
        self.order_counter = 1

    def submit_order(self, intent: OrderIntent) -> OrderEvent:
        """Queues an OrderIntent for execution on the next incoming market bar."""
        ord_id = f"REP_ORD_{self.order_counter:06d}"
        self.order_counter += 1

        order = OrderEvent(
            order_id=ord_id,
            symbol=intent.symbol.upper(),
            side=intent.side,
            order_type=intent.order_type,
            quantity=intent.quantity,
            status=OrderStatus.ACCEPTED,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            product_type=intent.product_type,
            strategy_id=intent.strategy_id,
            is_reduce_only=intent.is_reduce_only,
            tag=intent.tag,
            timestamp=intent.timestamp,
        )
        self._pending_orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancels a pending order."""
        for i, ord in enumerate(self._pending_orders):
            if ord.order_id == order_id:
                self._pending_orders.pop(i)
                return True
        return False

    def register_position_barriers(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: float,
        strategy_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Registers intrabar SL/TP barriers for an active open position."""
        self._active_barriers[symbol.upper()] = {
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "strategy_id": strategy_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def clear_position_barriers(self, symbol: str):
        """Removes barriers when a position is closed."""
        self._active_barriers.pop(symbol.upper(), None)

    def process_market_event(self, event: MarketEvent) -> List[FillEvent]:
        """
        Processes pending orders and active barriers against current market bar.
        1. Fills pending orders at event.open (next-bar open execution).
        2. Evaluates active SL/TP barriers against bar high/low.
        """
        fills: List[FillEvent] = []
        sym = event.symbol.upper()

        # 1. Process Pending Market Orders (Next-Bar Open Execution)
        remaining_pending = []
        for ord in self._pending_orders:
            if ord.symbol == sym:
                # Fill at bar open with slippage
                fill_p = event.open
                is_sl = False
                
                # Apply standard entry slippage
                slippage_factor = (self.slippage_bps / 10000.0) * (1.0 if ord.side == OrderSide.BUY else -1.0)
                exec_price = fill_p * (1.0 + slippage_factor)

                cost_bd = self._calculate_costs(
                    side=ord.side,
                    entry_p=exec_price,
                    exit_p=exec_price,
                    qty=ord.quantity,
                    is_sl=is_sl,
                )

                fill = FillEvent(
                    order_id=ord.order_id,
                    symbol=sym,
                    timestamp=event.timestamp,
                    side=ord.side,
                    fill_price=round(exec_price, 2),
                    quantity=ord.quantity,
                    commission=cost_bd.brokerage,
                    slippage=round(abs(exec_price - fill_p) * ord.quantity, 2),
                    cost_breakdown=cost_bd.to_dict(),
                    strategy_id=ord.strategy_id,
                    is_stop_loss=is_sl,
                )
                fills.append(fill)
            else:
                remaining_pending.append(ord)
        
        self._pending_orders = remaining_pending

        # 2. Process Intrabar Barrier Hits (SL / TP)
        if sym in self._active_barriers:
            barr = self._active_barriers[sym]
            pos_side = barr["side"]
            pos_qty = barr["quantity"]
            entry_p = barr["entry_price"]
            strat_id = barr["strategy_id"]
            sl_p = barr["stop_loss"]
            tp_p = barr["take_profit"]

            triggered = False
            exit_price = 0.0
            exit_side = OrderSide.SELL if pos_side == OrderSide.BUY else OrderSide.BUY
            is_sl_exit = False

            if pos_side == OrderSide.BUY:  # LONG
                # Check Open Gap below SL
                if sl_p is not None and sl_p > 0 and event.open <= sl_p:
                    triggered = True
                    exit_price = min(event.open, sl_p)
                    is_sl_exit = True
                # Check Ambiguity: Both SL and TP hit in same bar -> WORST_CASE SL
                elif sl_p is not None and sl_p > 0 and tp_p is not None and tp_p > 0 and event.low <= sl_p and event.high >= tp_p:
                    triggered = True
                    exit_price = sl_p
                    is_sl_exit = True
                # Check SL hit
                elif sl_p is not None and sl_p > 0 and event.low <= sl_p:
                    triggered = True
                    exit_price = sl_p
                    is_sl_exit = True
                # Check TP hit
                elif tp_p is not None and tp_p > 0 and event.high >= tp_p:
                    triggered = True
                    exit_price = tp_p
                    is_sl_exit = False

            else:  # SHORT
                # Check Open Gap above SL
                if sl_p is not None and sl_p > 0 and event.open >= sl_p:
                    triggered = True
                    exit_price = max(event.open, sl_p)
                    is_sl_exit = True
                # Check Ambiguity: Both SL and TP hit in same bar -> WORST_CASE SL
                elif sl_p is not None and sl_p > 0 and tp_p is not None and tp_p > 0 and event.high >= sl_p and event.low <= tp_p:
                    triggered = True
                    exit_price = sl_p
                    is_sl_exit = True
                # Check SL hit
                elif sl_p is not None and sl_p > 0 and event.high >= sl_p:
                    triggered = True
                    exit_price = sl_p
                    is_sl_exit = True
                # Check TP hit
                elif tp_p is not None and tp_p > 0 and event.low <= tp_p:
                    triggered = True
                    exit_price = tp_p
                    is_sl_exit = False

            if triggered:
                self.clear_position_barriers(sym)
                cost_bd = self.cost_model.calculate_trade_costs(
                    buy_price=entry_p if pos_side == OrderSide.BUY else exit_price,
                    sell_price=exit_price if pos_side == OrderSide.BUY else entry_p,
                    quantity=pos_qty,
                    segment=self.segment,
                    is_stop_loss=is_sl_exit,
                )

                fill = FillEvent(
                    order_id=f"BARRIER_{self.order_counter:06d}",
                    symbol=sym,
                    timestamp=event.timestamp,
                    side=exit_side,
                    fill_price=round(exit_price, 2),
                    quantity=pos_qty,
                    commission=cost_bd.brokerage,
                    slippage=0.0,
                    cost_breakdown=cost_bd.to_dict(),
                    strategy_id=strat_id,
                    is_stop_loss=is_sl_exit,
                )
                self.order_counter += 1
                fills.append(fill)

        return fills

    def _calculate_costs(
        self, side: OrderSide, entry_p: float, exit_p: float, qty: int, is_sl: bool
    ) -> Any:
        return self.cost_model.calculate_trade_costs(
            buy_price=entry_p if side == OrderSide.BUY else exit_p,
            sell_price=exit_p if side == OrderSide.BUY else entry_p,
            quantity=qty,
            segment=self.segment,
            is_stop_loss=is_sl,
        )

    def get_open_orders(self) -> List[OrderEvent]:
        return list(self._pending_orders)
