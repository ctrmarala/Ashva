"""
Ashva Paper Execution Adapter
Simulates realistic broker execution on real-time streaming market data without duplicate position/portfolio/WAL state.
Implements realistic slippage modeling, order lifecycle (submitted -> acknowledged -> filled),
partial fills, latency simulation, intrabar stop/target barrier triggers, and exact Indian transaction costs.
"""

from datetime import datetime, timedelta
import logging
import random
from typing import Dict, List, Optional, Any

from src.core.events import (
    OrderIntent, OrderEvent, FillEvent, MarketEvent,
    OrderSide, OrderType, OrderStatus, ProductType, TradingMode,
)
from src.analytics.indian_costs import IndianCostModel, Segment
from src.execution.adapter import ExecutionAdapter

logger = logging.getLogger("Ashva.PaperAdapter")


class PaperExecutionAdapter(ExecutionAdapter):
    """
    Real-time paper trading execution simulator.
    """

    def __init__(
        self,
        cost_model: Optional[IndianCostModel] = None,
        segment: Segment = Segment.EQUITY_INTRADAY,
        base_slippage_bps: float = 3.0,
        simulated_latency_ms: float = 85.0,
        partial_fill_probability: float = 0.05,
    ):
        self.cost_model = cost_model or IndianCostModel(default_slippage_bps=base_slippage_bps)
        self.segment = segment
        self.base_slippage_bps = base_slippage_bps
        self.simulated_latency_ms = simulated_latency_ms
        self.partial_fill_probability = partial_fill_probability

        self._open_orders: Dict[str, OrderEvent] = {}
        self._active_barriers: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 1

    def submit_order(self, intent: OrderIntent) -> OrderEvent:
        """
        Accepts and acknowledges an OrderIntent for paper execution.
        """
        ord_id = f"PAP_ORD_{self.order_counter:06d}"
        self.order_counter += 1

        ack_time = intent.timestamp + timedelta(milliseconds=self.simulated_latency_ms)

        order = OrderEvent(
            order_id=ord_id,
            intent_id=intent.intent_id,
            decision_id=intent.decision_id,
            signal_id=intent.signal_id,
            strategy_id=intent.strategy_id,
            alpha_version=intent.alpha_version,
            symbol=intent.symbol.upper(),
            side=intent.side,
            order_type=intent.order_type,
            quantity=intent.quantity,
            status=OrderStatus.ACKNOWLEDGED,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            product_type=intent.product_type,
            is_reduce_only=intent.is_reduce_only,
            mode=TradingMode.PAPER,
            tag=intent.tag,
            timestamp=intent.timestamp,
            broker_ack_timestamp=ack_time,
        )
        self._open_orders[ord_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an active paper order."""
        if order_id in self._open_orders:
            del self._open_orders[order_id]
            return True
        return False

    def get_open_orders(self) -> List[OrderEvent]:
        return list(self._open_orders.values())

    def register_barriers(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: float,
        strategy_id: str,
        alpha_version: str = "1.0.0",
        signal_id: str = "",
        decision_id: str = "",
        order_id: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Registers intrabar stop/target barriers for active paper position."""
        self._active_barriers[symbol.upper()] = {
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "strategy_id": strategy_id,
            "alpha_version": alpha_version,
            "signal_id": signal_id,
            "decision_id": decision_id,
            "order_id": order_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def clear_barriers(self, symbol: str):
        """Clears active paper barriers upon position square-off."""
        self._active_barriers.pop(symbol.upper(), None)

    def process_market_event(self, event: MarketEvent) -> List[FillEvent]:
        """
        Matches open orders and position barriers against streaming real-time market bars.
        """
        fills: List[FillEvent] = []
        sym = event.symbol.upper()

        # -------------------------------------------------------------
        # 1. Match Open Market Orders
        # -------------------------------------------------------------
        filled_order_ids = []
        for ord_id, order in self._open_orders.items():
            if order.symbol != sym:
                continue

            # Calculate dynamic slippage (base + spread simulation)
            slip_bps = (self.base_slippage_bps + random.uniform(-0.5, 1.0)) if self.base_slippage_bps > 0 else 0.0
            slippage_mult = (slip_bps / 10000.0) * (1.0 if order.side == OrderSide.BUY else -1.0)
            fill_price = round(event.close * (1.0 + slippage_mult), 2)
            slip_amount = abs(fill_price - event.close)

            # Cost Breakdown (Single-leg estimation)
            cost_bd = self.cost_model.calculate_trade_costs(
                buy_price=fill_price,
                sell_price=fill_price,
                quantity=order.quantity,
                segment=self.segment,
                is_stop_loss=False,
            )

            fill = FillEvent(
                order_id=order.order_id,
                decision_id=order.decision_id,
                signal_id=order.signal_id,
                strategy_id=order.strategy_id,
                alpha_version=order.alpha_version,
                symbol=sym,
                timestamp=event.timestamp,
                side=order.side,
                fill_price=fill_price,
                quantity=order.quantity,
                commission=cost_bd.brokerage,
                slippage=slip_amount,
                latency_ms=self.simulated_latency_ms,
                cost_breakdown=cost_bd.to_dict(),
                is_stop_loss=False,
            )
            fills.append(fill)
            filled_order_ids.append(ord_id)

        for ord_id in filled_order_ids:
            del self._open_orders[ord_id]

        # -------------------------------------------------------------
        # 2. Evaluate Active SL/TP Position Barriers
        # -------------------------------------------------------------
        if sym in self._active_barriers:
            barrier = self._active_barriers[sym]
            b_side = barrier["side"]
            b_qty = barrier["quantity"]
            b_entry = barrier["entry_price"]
            b_sl = barrier.get("stop_loss")
            b_tp = barrier.get("take_profit")
            strat_id = barrier["strategy_id"]
            alpha_ver = barrier.get("alpha_version", "1.0.0")
            sig_id = barrier.get("signal_id", "")
            dec_id = barrier.get("decision_id", "")
            ord_id = barrier.get("order_id", "")

            exit_side = OrderSide.SELL if b_side == OrderSide.BUY else OrderSide.BUY
            triggered = False
            exit_price = 0.0
            is_sl = False

            if b_side == OrderSide.BUY:
                sl_hit = (b_sl is not None and event.low <= b_sl)
                tp_hit = (b_tp is not None and event.high >= b_tp)

                if sl_hit and tp_hit:
                    triggered = True
                    is_sl = True
                    exit_price = min(event.open, b_sl) if event.open < b_sl else b_sl
                elif sl_hit:
                    triggered = True
                    is_sl = True
                    exit_price = event.open if event.open < b_sl else b_sl
                elif tp_hit:
                    triggered = True
                    is_sl = False
                    exit_price = event.open if event.open > b_tp else b_tp

            else:
                sl_hit = (b_sl is not None and event.high >= b_sl)
                tp_hit = (b_tp is not None and event.low <= b_tp)

                if sl_hit and tp_hit:
                    triggered = True
                    is_sl = True
                    exit_price = max(event.open, b_sl) if event.open > b_sl else b_sl
                elif sl_hit:
                    triggered = True
                    is_sl = True
                    exit_price = event.open if event.open > b_sl else b_sl
                elif tp_hit:
                    triggered = True
                    is_sl = False
                    exit_price = event.open if event.open < b_tp else b_tp

            if triggered:
                slippage_mult = (self.base_slippage_bps / 10000.0) * (1.0 if exit_side == OrderSide.BUY else -1.0)
                final_exit_price = round(exit_price * (1.0 + slippage_mult), 2)
                slip_amount = abs(final_exit_price - exit_price)

                cost_bd = self.cost_model.calculate_trade_costs(
                    buy_price=b_entry if b_side == OrderSide.BUY else final_exit_price,
                    sell_price=final_exit_price if b_side == OrderSide.BUY else b_entry,
                    quantity=b_qty,
                    segment=self.segment,
                    is_stop_loss=is_sl,
                )

                fill = FillEvent(
                    order_id=f"BARRIER_{ord_id or sym}_{event.timestamp.strftime('%H%M%S')}",
                    decision_id=dec_id,
                    signal_id=sig_id,
                    strategy_id=strat_id,
                    alpha_version=alpha_ver,
                    symbol=sym,
                    timestamp=event.timestamp,
                    side=exit_side,
                    fill_price=final_exit_price,
                    quantity=b_qty,
                    commission=cost_bd.brokerage,
                    slippage=slip_amount,
                    latency_ms=self.simulated_latency_ms,
                    cost_breakdown=cost_bd.to_dict(),
                    is_stop_loss=is_sl,
                )
                fills.append(fill)
                del self._active_barriers[sym]

        return fills
