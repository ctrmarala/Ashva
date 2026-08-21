"""
Ashva Production Position Manager
Authoritative position tracking engine.
Computes real-time Mark-to-Market (MTM) valuations, continuous MFE/MAE excursions,
and produces fully attributed trade outcome records on close.
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
        alpha_version: str = "1.0.0",
        signal_id: str = "",
        decision_id: str = "",
        order_id: str = "",
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
        self.alpha_version = alpha_version
        self.signal_id = signal_id
        self.decision_id = decision_id
        self.order_id = order_id
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.stop_dist = stop_dist or (abs(entry_price - stop_loss) if stop_loss else 0.0)
        
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # MFE / MAE tracking
        self.mfe = 0.0        # Max Favorable Excursion in Rs
        self.mae = 0.0        # Max Adverse Excursion in Rs (negative or positive magnitude)
        self.mfe_pct = 0.0    # Max Favorable Excursion in %
        self.mae_pct = 0.0    # Max Adverse Excursion in %
        self.bars_held = 0

    def update_market_bar(self, event: MarketEvent):
        """Updates MTM and continuous MFE/MAE on each incoming candle."""
        self.current_price = event.close
        self.bars_held += 1
        direction = 1.0 if self.side == OrderSide.BUY else -1.0
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity * direction

        # MFE / MAE calculations against high and low
        if self.side == OrderSide.BUY:
            fav_price = event.high
            adv_price = event.low
            curr_fav_pnl = (fav_price - self.entry_price) * self.quantity
            curr_adv_pnl = (adv_price - self.entry_price) * self.quantity
            curr_fav_pct = (fav_price - self.entry_price) / self.entry_price * 100.0
            curr_adv_pct = (adv_price - self.entry_price) / self.entry_price * 100.0
        else:
            fav_price = event.low
            adv_price = event.high
            curr_fav_pnl = (self.entry_price - fav_price) * self.quantity
            curr_adv_pnl = (self.entry_price - adv_price) * self.quantity
            curr_fav_pct = (self.entry_price - fav_price) / self.entry_price * 100.0
            curr_adv_pct = (self.entry_price - adv_price) / self.entry_price * 100.0

        if curr_fav_pnl > self.mfe:
            self.mfe = curr_fav_pnl
            self.mfe_pct = curr_fav_pct
        if curr_adv_pnl < self.mae:
            self.mae = curr_adv_pnl
            self.mae_pct = curr_adv_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "strategy_id": self.strategy_id,
            "alpha_version": self.alpha_version,
            "signal_id": self.signal_id,
            "decision_id": self.decision_id,
            "order_id": self.order_id,
            "current_price": self.current_price,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "mfe": round(self.mfe, 2),
            "mae": round(self.mae, 2),
            "mfe_pct": round(self.mfe_pct, 2),
            "mae_pct": round(self.mae_pct, 2),
            "bars_held": self.bars_held,
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
        """Updates MTM valuation and MFE/MAE for open positions on this symbol."""
        sym = event.symbol.upper()
        if sym in self.open_positions:
            self.open_positions[sym].update_market_bar(event)

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
                alpha_version=fill.alpha_version,
                signal_id=fill.signal_id,
                decision_id=fill.decision_id,
                order_id=fill.order_id,
            )
            return None

        pos = self.open_positions[sym]

        # 2. Check if this fill closes/reduces existing position
        is_closing = (
            (pos.side == OrderSide.BUY and fill.side == OrderSide.SELL)
            or (pos.side == OrderSide.SELL and fill.side == OrderSide.BUY)
        )

        if is_closing:
            exit_price = fill.fill_price
            entry_price = pos.entry_price
            close_qty = min(pos.quantity, fill.quantity)

            # Compute exact statutory costs & net PnL
            if pos.side == OrderSide.BUY:
                cost_bd = self.cost_model.calculate_trade_costs(
                    buy_price=entry_price,
                    sell_price=exit_price,
                    quantity=close_qty,
                    segment=Segment.EQUITY_INTRADAY,
                    is_stop_loss=fill.is_stop_loss,
                )
            else:
                cost_bd = self.cost_model.calculate_trade_costs(
                    buy_price=exit_price,
                    sell_price=entry_price,
                    quantity=close_qty,
                    segment=Segment.EQUITY_INTRADAY,
                    is_stop_loss=fill.is_stop_loss,
                )

            closed_record = {
                "symbol": sym,
                "strategy_id": pos.strategy_id,
                "alpha_version": pos.alpha_version,
                "signal_id": pos.signal_id,
                "decision_id": pos.decision_id,
                "order_id": pos.order_id,
                "fill_id": fill.fill_id,
                "side": pos.side.value,
                "quantity": close_qty,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "entry_time": pos.entry_time,
                "exit_time": fill.timestamp,
                "gross_pnl": round(cost_bd.gross_pnl, 2),
                "net_pnl": round(cost_bd.net_pnl, 2),
                "slippage_paid": round(fill.slippage * close_qty, 2),
                "total_costs": round(cost_bd.total_tax_and_charges, 2),
                "mfe": round(pos.mfe, 2),
                "mae": round(pos.mae, 2),
                "mfe_pct": round(pos.mfe_pct, 2),
                "mae_pct": round(pos.mae_pct, 2),
                "holding_period_bars": pos.bars_held,
                "exit_reason": "STOP_LOSS" if fill.is_stop_loss else ("EOD_SQUAREOFF" if fill.order_id.startswith("EOD") else "SIGNAL_EXIT"),
                "cost_breakdown": cost_bd.to_dict(),
            }

            self.closed_trades.append(closed_record)

            # Adjust remaining quantity or delete position
            pos.quantity -= close_qty
            if pos.quantity <= 0:
                del self.open_positions[sym]

            return closed_record

        else:
            # Adding to existing position (Weighted average entry price)
            total_qty = pos.quantity + fill.quantity
            pos.entry_price = (pos.entry_price * pos.quantity + fill.fill_price * fill.quantity) / total_qty
            pos.quantity = total_qty
            return None

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.open_positions.get(symbol.upper())

    def get_all_positions(self) -> List[Position]:
        return list(self.open_positions.values())

    def get_total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.open_positions.values())
