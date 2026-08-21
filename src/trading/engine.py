"""
Ashva Master Unified Production Trading Engine
A single, mode-agnostic, event-driven trading engine supporting REPLAY, PAPER, and LIVE execution.
Eliminates all mode-specific branching in core logic. Integrates with MultiAlphaAllocator,
LiveRiskManager, ExecutionAdapter, and non-blocking Async TradingLedger.
"""

from datetime import datetime, time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.core.events import (
    MarketEvent, SignalEvent, SignalType, OrderIntent,
    OrderEvent, FillEvent, OrderSide, OrderType, ProductType,
    PortfolioUpdateEvent, TradingMode,
)
from src.market_data.provider import MarketDataProvider
from src.execution.adapter import ExecutionAdapter
from src.trading.contract import QualifiedAlphaContract
from src.trading.manifest import TradingManifest
from src.trading.allocator import MultiAlphaAllocator
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.portfolio_state import PortfolioState
from src.trading.live_rms import LiveRiskManager
from src.trading.ledger import TradingLedger
from src.analytics.indian_costs import IndianCostModel, Segment

logger = logging.getLogger("Ashva.TradingEngine")


class TradingEngine:
    """
    Master unified production execution engine for Ashva.
    """

    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        execution_adapter: ExecutionAdapter,
        alpha_contracts: Optional[List[QualifiedAlphaContract]] = None,
        manifest: Optional[TradingManifest] = None,
        initial_capital: float = 500000.0,
        cost_model: Optional[IndianCostModel] = None,
        risk_manager: Optional[LiveRiskManager] = None,
        allocator: Optional[MultiAlphaAllocator] = None,
        ledger: Optional[TradingLedger] = None,
        mode: TradingMode = TradingMode.REPLAY,
    ):
        self.market_data_provider = market_data_provider
        self.execution_adapter = execution_adapter
        self.mode = mode
        self.initial_capital = initial_capital
        self.cost_model = cost_model or IndianCostModel()

        # Contract Manifest
        self.manifest = manifest or TradingManifest(contracts=alpha_contracts or [])
        
        # State Managers (Single Authoritative Source of Truth)
        self.order_manager = OrderManager()
        self.position_manager = PositionManager(cost_model=self.cost_model)
        self.portfolio_state = PortfolioState(initial_capital=initial_capital)
        self.risk_manager = risk_manager or LiveRiskManager()
        self.allocator = allocator or MultiAlphaAllocator()
        self.ledger = ledger or TradingLedger()

        # Strategy Instances (Instantiated directly from Frozen Contracts)
        self._strategy_instances: Dict[str, Any] = {}
        for contract in self.manifest.get_active_contracts():
            self._strategy_instances[contract.alpha_id] = contract.instantiate_strategy()

        # Point-in-Time Rolling Bar History Buffer per symbol
        self._history_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._prime_warmup_buffers()

    def _prime_warmup_buffers(self):
        """Pre-loads historical warmup bars to prime rolling technical indicators."""
        symbols = self.manifest.get_all_symbols()
        for sym in symbols:
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

        # Flush async ledger upon completion
        self.ledger.flush()
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
            self.ledger.log_fill(fill)
            
            closed_trade = self.position_manager.on_fill(fill)
            if closed_trade is not None:
                closed_trade["mode"] = self.mode.value if hasattr(self.mode, "value") else str(self.mode)
                self.portfolio_state.on_trade_closed(closed_trade["net_pnl"], fill.timestamp)
                self.ledger.log_closed_trade(closed_trade)

        # 3. Update Position & Portfolio MTM Valuations
        self.position_manager.on_market_event(market_event)
        total_unrealized = self.position_manager.get_total_unrealized_pnl()
        self.portfolio_state.update_mtm(total_unrealized, event_time)

        # Log periodic portfolio snapshot to async ledger
        p_snapshot = PortfolioUpdateEvent(
            timestamp=event_time,
            cash=self.portfolio_state.cash,
            realized_pnl=self.portfolio_state.realized_pnl,
            unrealized_pnl=total_unrealized,
            total_equity=self.portfolio_state.current_equity,
            open_positions_count=len(self.position_manager.get_all_positions()),
            drawdown_pct=self.portfolio_state.get_drawdown_pct(),
            daily_loss_pct=self.portfolio_state.get_daily_loss_pct(),
            mode=self.mode,
        )
        self.ledger.log_portfolio_snapshot(p_snapshot)

        # 4. Check Intraday Mandatory EOD Square-Off (15:15 MIS)
        pos = self.position_manager.get_position(sym)
        is_eod = (event_time.hour >= 15 and event_time.minute >= 15)

        if pos is not None and is_eod:
            exit_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
            # Polymorphic barrier clear on execution adapter
            self.execution_adapter.clear_barriers(sym)
            
            cost_bd = self.cost_model.calculate_trade_costs(
                buy_price=pos.entry_price if pos.side == OrderSide.BUY else current_price,
                sell_price=current_price if pos.side == OrderSide.BUY else pos.entry_price,
                quantity=pos.quantity,
                segment=Segment.EQUITY_INTRADAY,
                is_stop_loss=False,
            )
            fill = FillEvent(
                order_id=f"EOD_SQOFF_{len(self.order_manager.fills)+1:06d}",
                decision_id=pos.decision_id,
                signal_id=pos.signal_id,
                strategy_id=pos.strategy_id,
                alpha_version=pos.alpha_version,
                symbol=sym,
                timestamp=event_time,
                side=exit_side,
                fill_price=round(current_price, 2),
                quantity=pos.quantity,
                commission=cost_bd.brokerage,
                slippage=0.0,
                cost_breakdown=cost_bd.to_dict(),
                is_stop_loss=False,
            )
            self.order_manager.on_fill(fill)
            self.ledger.log_fill(fill)
            
            closed_trade = self.position_manager.on_fill(fill)
            if closed_trade is not None:
                closed_trade["mode"] = self.mode.value if hasattr(self.mode, "value") else str(self.mode)
                self.portfolio_state.on_trade_closed(closed_trade["net_pnl"], event_time)
                self.ledger.log_closed_trade(closed_trade)
            return

        # 5. Evaluate Alpha Contracts & Generate Signals
        if pos is None and not is_eod:
            active_contracts = self.manifest.get_contracts_for_symbol(sym)
            if not active_contracts:
                return

            candidate_signals: List[SignalEvent] = []
            df_hist = pd.DataFrame(self._history_buffers[sym]).set_index("timestamp")
            if len(df_hist) < 15:
                return

            contracts_map = {c.alpha_id: c for c in active_contracts}

            for contract in active_contracts:
                strat = self._strategy_instances.get(contract.alpha_id)
                if strat is None:
                    continue

                try:
                    df_sig = strat.generate_signals(df_hist)
                    if "signal" not in df_sig.columns:
                        continue
                    
                    latest_sig = float(df_sig["signal"].iloc[-1])
                    if latest_sig == 0.0 or np.isnan(latest_sig):
                        continue

                    sig_type = SignalType.LONG if latest_sig > 0 else SignalType.SHORT
                    sl_val = float(df_sig["stop_loss"].iloc[-1]) if "stop_loss" in df_sig.columns and pd.notna(df_sig["stop_loss"].iloc[-1]) else None
                    tp_val = float(df_sig["take_profit"].iloc[-1]) if "take_profit" in df_sig.columns and pd.notna(df_sig["take_profit"].iloc[-1]) else None

                    conf_val = float(df_sig["confidence"].iloc[-1]) if "confidence" in df_sig.columns and pd.notna(df_sig["confidence"].iloc[-1]) else 1.0

                    sig = SignalEvent(
                        symbol=sym,
                        timestamp=event_time,
                        strategy_id=contract.alpha_id,
                        alpha_version=contract.alpha_version,
                        signal_type=sig_type,
                        confidence=conf_val,
                        suggested_stop_loss=sl_val,
                        suggested_take_profit=tp_val,
                    )
                    candidate_signals.append(sig)
                    self.ledger.log_signal(sig)

                except Exception as e:
                    logger.error(f"Alpha {contract.alpha_id} signal generation error: {e}")
                    continue

            # 6. Multi-Alpha Allocation & Sizing
            intents, decisions = self.allocator.allocate(
                candidate_signals=candidate_signals,
                contracts_map=contracts_map,
                current_prices={sym: current_price},
                position_manager=self.position_manager,
                portfolio_state=self.portfolio_state,
            )

            for dec in decisions:
                self.ledger.log_decision(dec)

            # 7. Risk Validation & Submission for Approved Intents
            for intent in intents:
                is_approved, reject_reason = self.risk_manager.validate_order(
                    intent=intent,
                    current_price=current_price,
                    position_manager=self.position_manager,
                    portfolio_state=self.portfolio_state,
                )

                if is_approved:
                    order_event = self.execution_adapter.submit_order(intent)
                    self.order_manager.on_order_submitted(intent, order_event)
                    self.ledger.log_order(order_event)

                    # Polymorphic barrier registration
                    self.execution_adapter.register_barriers(
                        symbol=sym,
                        side=intent.side,
                        quantity=intent.quantity,
                        entry_price=current_price,
                        strategy_id=intent.strategy_id,
                        alpha_version=intent.alpha_version,
                        signal_id=intent.signal_id,
                        decision_id=intent.decision_id,
                        order_id=order_event.order_id,
                        stop_loss=intent.stop_loss,
                        take_profit=intent.take_profit,
                    )
                else:
                    self.order_manager.on_order_rejected(intent.intent_id, reject_reason or "RMS Rejection")

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
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate_pct": round(win_rate, 1),
            "gross_profit_factor": round(gross_pf, 2),
            "net_profit_factor": round(net_pf, 2),
            "closed_trades": closed_trades,
        }
