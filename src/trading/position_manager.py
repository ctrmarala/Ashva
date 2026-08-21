"""
Ashva Production Position Manager
Maintains live position books, average entry prices, real-time Mark-to-Market (MTM) valuations,
and realized PnL accounting upon execution fills.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from src.core.events import FillEvent, MarketEvent, OrderSide, PositionUpdateEvent
from src.analytics.indian_costs import IndianCostModel, Segment


class Position:
    """Represents an active open position in a single symbol."""
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: float,
        entry_time: datetime,
        strategy_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_dist: Optional[float] = None,
    ):
        self.symbol = symbol.upper()
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.strategy_id = strategy_id
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.stop_dist = stop_dist or (abs(entry_price - stop_loss) if stop_loss else 0.0)
        
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

    def update_price(self, current_price: float):
        self.current_price = current_price
        direction = 1.0 if self.side == OrderSide.BUY else -1.0
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity * direction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "strategy_id": self.strategy_id,
            "current_price": self.current_price,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "stop_dist": self.stop_dist,
        }


class PositionManager:
    """
    Central position tracking engine.
    """

    def __init__(self, cost_model: Optional[IndianCostModel] = None):
        self.cost_model = cost_model or IndianCostModel()
        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict[str, Any]] = []

    def on_market_event(self, event: MarketEvent):
        """Updates MTM valuation for any open position on this symbol."""
        sym = event.symbol.upper()
        if sym in self.open_positions:
            self.open_positions[sym].update_price(event.close)

    def on_fill(self, fill: FillEvent) -> Optional[Dict[str, Any]]:
        """
        Updates position state on fill. Returns closed trade record if a position was closed.
        """
        sym = fill.symbol.upper()

        if sym not in self.open_positions:
            # 1. New Position Entry
            self.open_positions[sym] = Position(
                symbol=sym,
                side=fill.side,
                quantity=fill.quantity,
                entry_price=fill.fill_price,
                entry_time=fill.timestamp,
                strategy_id=fill.strategy_id,
            )
            return None

        pos = self.open_positions[sym]

        # 2. Check if this fill increases or closes/reduces position
        if pos.side == fill.side:
            # Increasing Position: Compute Weighted Average Entry Price
            total_qty = pos.quantity + fill.quantity
            weighted_price = ((pos.entry_price * pos.quantity) + (fill.fill_price * fill.quantity)) / total_qty
            pos.quantity = total_qty
            pos.entry_price = weighted_price
            return None

        else:
            # Closing / Reducing Position
            close_qty = min(pos.quantity, fill.quantity)
            is_long = (pos.side == OrderSide.BUY)

            buy_p = pos.entry_price if is_long else fill.fill_price
            sell_p = fill.fill_price if is_long else pos.entry_price

            cost_bd = self.cost_model.calculate_trade_costs(
                buy_price=buy_p,
                sell_price=sell_p,
                quantity=close_qty,
                segment=Segment.EQUITY_INTRADAY,
                is_stop_loss=fill.is_stop_loss,
            )

            gross_pnl = ((sell_p - buy_p) * close_qty) if is_long else ((buy_p - sell_p) * close_qty)
            net_pnl = cost_bd.net_pnl

            trade_record = {
                "symbol": sym,
                "strategy_id": pos.strategy_id,
                "side": pos.side.value,
                "quantity": close_qty,
                "entry_time": pos.entry_time,
                "exit_time": fill.timestamp,
                "entry_price": pos.entry_price,
                "exit_price": fill.fill_price,
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "costs": cost_bd.to_dict(),
                "is_stop_loss": fill.is_stop_loss,
            }
            self.closed_trades.append(trade_record)

            if pos.quantity <= close_qty:
                self.open_positions.pop(sym)
            else:
                pos.quantity -= close_qty

            return trade_record

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.open_positions.get(symbol.upper())

    def get_total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.open_positions.values())

    def get_open_positions_dict(self) -> Dict[str, Dict[str, Any]]:
        return {sym: pos.to_dict() for sym, pos in self.open_positions.items()}
