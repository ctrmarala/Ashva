"""
Ashva Master Unified Production Trading Engine
A mode-agnostic, event-driven trading engine that processes MarketEvents, executes Qualified Alpha Contracts,
validates through LiveRiskManager, and routes OrderIntents through pluggable ExecutionAdapters.
Zero mode-branching conditionals (REPLAY / PAPER / LIVE operate via interchangeable adapters).
"""

from datetime import datetime, time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.core.events import (
    MarketEvent, SignalEvent, SignalType, OrderIntent,
    OrderEvent, FillEvent, OrderSide, OrderType, ProductType,
)
from src.market_data.provider import MarketDataProvider
from src.execution.adapter import ExecutionAdapter
from src.execution.replay_adapter import ReplayExecutionAdapter
from src.trading.contract import QualifiedAlphaContract
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.portfolio_state import PortfolioState
from src.trading.live_rms import LiveRiskManager
from src.analytics.indian_costs import IndianCostModel, Segment

logger = logging.getLogger("Ashva.TradingEngine")


class TradingEngine:
    """
    Master production execution engine for Ashva.
    """

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        execution_adapter: ExecutionAdapter,
        alpha_contracts: List[QualifiedAlphaContract],
        initial_capital: float = 500000.0,
        cost_model: Optional[IndianCostModel] = None,
        risk_manager: Optional[LiveRiskManager] = None,
    ):
        self.market_data_provider = market_data_provider
        self.execution_adapter = execution_adapter
        self.alpha_contracts = alpha_contracts
        self.initial_capital = initial_capital
        self.cost_model = cost_model or IndianCostModel()
        
        # State Managers
        self.order_manager = OrderManager()
        self.position_manager = PositionManager(cost_model=self.cost_model)
        self.portfolio_state = PortfolioState(initial_capital=initial_capital)
        self.risk_manager = risk_manager or LiveRiskManager()

        # Strategy Instances (Instantiated from Frozen Contracts)
        self._strategy_instances: Dict[str, Any] = {
            c.alpha_id: c.instantiate_strategy() for c in alpha_contracts
        }

        # Point-in-Time Rolling Bar History Buffer per symbol
        self._history_buffers: Dict[str, List[Dict[str, Any]]] = {}
        if hasattr(self.market_data_provider, "get_warmup_bars"):
            for contract in alpha_contracts:
                for sym in contract.universe:
                    s_clean = sym.upper()
                    warmup = self.market_data_provider.get_warmup_bars(s_clean, count=800)
                    if warmup:
                        self._history_buffers[s_clean] = list(warmup)

    def run(self) -> Dict[str, Any]:
        """
        Executes the main synchronous event-driven trading loop.
        Consumes streaming MarketEvents until provider completion.
        """
        for market_event in self.market_data_provider.stream_events():
            self.step(market_event)

        return self.get_summary()

    def step(self, market_event: MarketEvent):
        """
        Processes a single incoming MarketEvent through the complete engine pipeline.
        """
        sym = market_event.symbol.upper()
        current_price = market_event.close
        event_time = market_event.timestamp

        # 1. Update Historical Bar Buffer for Symbol
        if sym not in self._history_buffers:
            self._history_buffers[sym] = []
        self._history_buffers[sym].append({
            "timestamp": event_time,
            "open": market_event.open,
            "high": market_event.high,
            "low": market_event.low,
            "close": market_event.close,
            "volume": market_event.volume,
            "vwap": market_event.vwap,
        })

        # 2. Process Adapter Fills (Next-Bar Open Fills & Intrabar Barriers)
        fills = self.execution_adapter.process_market_event(market_event)
        for fill in fills:
            self.order_manager.on_fill(fill)
            closed_trade = self.position_manager.on_fill(fill)
            if closed_trade is not None:
                self.portfolio_state.on_trade_closed(closed_trade["net_pnl"], fill.timestamp)

        # 3. Update Position & Portfolio MTM Valuations
        self.position_manager.on_market_event(market_event)
        total_unrealized = self.position_manager.get_total_unrealized_pnl()
        self.portfolio_state.update_mtm(total_unrealized, event_time)

        # 4. Check Intraday Mandatory EOD Square-Off (15:15 MIS)
        pos = self.position_manager.get_position(sym)
        is_eod = (event_time.hour >= 15 and event_time.minute >= 15)

        if pos is not None and is_eod:
            # Intraday MIS mandatory EOD square-off at 15:15 bar close
            exit_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
            if isinstance(self.execution_adapter, ReplayExecutionAdapter):
                self.execution_adapter.clear_position_barriers(sym)
            
            cost_bd = self.cost_model.calculate_trade_costs(
                buy_price=pos.entry_price if pos.side == OrderSide.BUY else current_price,
                sell_price=current_price if pos.side == OrderSide.BUY else pos.entry_price,
                quantity=pos.quantity,
                segment=Segment.EQUITY_INTRADAY,
                is_stop_loss=False,
            )
            fill = FillEvent(
                order_id=f"EOD_SQOFF_{len(self.order_manager.fills)+1:06d}",
                symbol=sym,
                timestamp=event_time,
                side=exit_side,
                fill_price=round(current_price, 2),
                quantity=pos.quantity,
                commission=cost_bd.brokerage,
                slippage=0.0,
                cost_breakdown=cost_bd.to_dict(),
                strategy_id=pos.strategy_id,
                is_stop_loss=False,
            )
            self.order_manager.on_fill(fill)
            closed_trade = self.position_manager.on_fill(fill)
            if closed_trade is not None:
                self.portfolio_state.on_trade_closed(closed_trade["net_pnl"], event_time)
            return

        # 5. Evaluate Alpha Contracts if Not in Position
        if pos is None and not is_eod:
            # Find relevant contracts for this symbol
            relevant_contracts = [
                c for c in self.alpha_contracts if sym in [s.upper() for s in c.universe]
            ]
            # Sort contracts by priority score descending
            relevant_contracts.sort(key=lambda c: c.priority_score, reverse=True)

            for contract in relevant_contracts:
                strat = self._strategy_instances.get(contract.alpha_id)
                if strat is None:
                    continue

                # Prepare DataFrame from point-in-time historical buffer
                df_hist = pd.DataFrame(self._history_buffers[sym]).set_index("timestamp")
                if len(df_hist) < 15:  # Minimum warmup bars
                    continue

                try:
                    df_sig = strat.generate_signals(df_hist)
                    if "signal" not in df_sig.columns:
                        continue
                    
                    latest_sig = float(df_sig["signal"].iloc[-1])
                    if latest_sig == 0.0 or np.isnan(latest_sig):
                        continue

                    # Signal Generated: Sizing based on real stop loss / risk
                    side = OrderSide.BUY if latest_sig > 0 else OrderSide.SELL
                    
                    # Extract stop-loss / take-profit from signal df if present
                    sl_val = float(df_sig["stop_loss"].iloc[-1]) if "stop_loss" in df_sig.columns and pd.notna(df_sig["stop_loss"].iloc[-1]) else None
                    tp_val = float(df_sig["take_profit"].iloc[-1]) if "take_profit" in df_sig.columns and pd.notna(df_sig["take_profit"].iloc[-1]) else None

                    # If strategy didn't provide stops, use contract fallbacks
                    if sl_val is None or sl_val <= 0:
                        if contract.stop_loss_pct is not None:
                            sl_val = current_price * (1.0 - contract.stop_loss_pct) if side == OrderSide.BUY else current_price * (1.0 + contract.stop_loss_pct)
                        else:
                            sl_val = current_price * 0.99 if side == OrderSide.BUY else current_price * 1.01

                    stop_dist = max(0.05, abs(current_price - sl_val))
                    risk_budget = max(500.0, self.portfolio_state.current_equity * contract.risk_per_trade_pct)
                    qty_from_risk = int(risk_budget / stop_dist)
                    max_cap_qty = int((self.portfolio_state.current_equity * contract.max_capital_allocation_pct) / current_price)
                    qty = max(1, min(qty_from_risk, max_cap_qty))

                    intent = OrderIntent(
                        strategy_id=contract.alpha_id,
                        symbol=sym,
                        side=side,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                        product_type=ProductType.INTRADAY,
                        is_reduce_only=False,
                        tag=contract.alpha_id,
                        timestamp=event_time,
                    )

                    # 6. Validate via LiveRiskManager
                    is_approved, reject_reason = self.risk_manager.validate_order(
                        intent=intent,
                        current_price=current_price,
                        position_manager=self.position_manager,
                        portfolio_state=self.portfolio_state,
                    )

                    if is_approved:
                        order_event = self.execution_adapter.submit_order(intent)
                        self.order_manager.on_order_submitted(intent, order_event)

                        # Register barriers with replay adapter if supported
                        if isinstance(self.execution_adapter, ReplayExecutionAdapter):
                            self.execution_adapter.register_position_barriers(
                                symbol=sym,
                                side=side,
                                quantity=qty,
                                entry_price=current_price,
                                strategy_id=contract.alpha_id,
                                stop_loss=sl_val,
                                take_profit=tp_val,
                            )
                        break  # Only one alpha entry per symbol per bar
                    else:
                        self.order_manager.on_order_rejected(intent.intent_id, reject_reason or "RMS Rejection")

                except Exception as e:
                    logger.error(f"Strategy {contract.alpha_id} execution error: {e}")
                    continue

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated trading metrics and trade list."""
        portfolio_summary = self.portfolio_state.get_summary()
        closed_trades = self.position_manager.closed_trades

        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t["net_pnl"] > 0)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_wins = sum(t["gross_pnl"] for t in closed_trades if t["gross_pnl"] > 0)
        gross_losses = abs(sum(t["gross_pnl"] for t in closed_trades if t["gross_pnl"] < 0))
        gross_pf = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 0.0)

        net_wins = sum(t["net_pnl"] for t in closed_trades if t["net_pnl"] > 0)
        net_losses = abs(sum(t["net_pnl"] for t in closed_trades if t["net_pnl"] < 0))
        net_pf = (net_wins / net_losses) if net_losses > 0 else (99.0 if net_wins > 0 else 0.0)

        return {
            **portfolio_summary,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate_pct": round(win_rate, 1),
            "gross_profit_factor": round(gross_pf, 2),
            "net_profit_factor": round(net_pf, 2),
            "closed_trades": closed_trades,
        }
