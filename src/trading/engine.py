"""
Ashva Master Unified Production Trading Engine
A single, mode-agnostic, event-driven trading engine supporting REPLAY, PAPER, and LIVE execution.
Eliminates all mode-specific branching in core logic. Integrates with MultiAlphaAllocator,
LiveRiskManager, ExecutionAdapter, and non-blocking Async TradingLedger.
Enforces per-alpha entry windows, actual-fill barrier registration, and strict risk decision logging.
"""

from datetime import datetime, time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.core.events import (
    MarketEvent, SignalEvent, SignalType, OrderIntent,
    OrderEvent, FillEvent, OrderSide, OrderType, ProductType,
    PortfolioUpdateEvent, TradingMode, DecisionEvent,
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


class ReplayDiagnosticTracker:
    """Tracks signal generation drops per alpha during Replay mode."""
    def __init__(self):
        # alpha_id -> metrics dict
        self.stats: Dict[str, Dict[str, Any]] = {}
        
    def init_alpha(self, alpha_id: str, symbol: str, entry_start: time, entry_end: time):
        if alpha_id not in self.stats:
            self.stats[alpha_id] = {
                "alpha_id": alpha_id,
                "symbols_evaluated": set(),
                "bars_received": 0,
                "generate_signals_calls": 0,
                "raw_signals": 0,
                "accepted_signals": 0,
                "allocator_rejected": 0,
                "risk_rejected": 0,
                "final_trades": 0,
                "entry_window": f"{entry_start.strftime('%H:%M')}-{entry_end.strftime('%H:%M')}"
            }
        self.stats[alpha_id]["symbols_evaluated"].add(symbol)
        
    def get_summary(self) -> List[Dict[str, Any]]:
        res = []
        for v in self.stats.values():
            d = v.copy()
            d["symbols_evaluated"] = ",".join(sorted(list(d["symbols_evaluated"])))
            res.append(d)
        return res


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

        self.diagnostic_tracker = ReplayDiagnosticTracker()
        for contract in self.manifest.get_active_contracts():
            sym_list = getattr(contract, "symbols", None) or getattr(contract, "universe", [])
            for sym in sym_list:
                self.diagnostic_tracker.init_alpha(contract.alpha_id, sym, contract.entry_start_time, contract.entry_end_time)

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
        event_tod = event_time.time()

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
        pos_before = self.position_manager.get_position(sym)
        fills = self.execution_adapter.process_market_event(market_event)
        
        for fill in fills:
            self.order_manager.on_fill(fill)
            self.ledger.log_fill(fill)
            
            closed_trade = self.position_manager.on_fill(fill)
            if closed_trade is not None:
                closed_trade["mode"] = self.mode.value if hasattr(self.mode, "value") else str(self.mode)
                self.portfolio_state.on_trade_closed(closed_trade["net_pnl"], fill.timestamp)
                self.ledger.log_closed_trade(closed_trade)
            elif pos_before is None:
                # Newly opened position -> Register barrier with adapter using actual fill price!
                self.execution_adapter.register_barriers(
                    symbol=sym,
                    side=fill.side,
                    quantity=fill.quantity,
                    entry_price=fill.fill_price,
                    strategy_id=fill.strategy_id,
                    alpha_version=fill.alpha_version,
                    signal_id=fill.signal_id,
                    decision_id=fill.decision_id,
                    order_id=fill.order_id,
                    stop_loss=fill.stop_loss,
                    take_profit=fill.take_profit,
                )

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

        # 4. Check Intraday Mandatory EOD Square-Off based on Contract Configuration
        pos = self.position_manager.get_position(sym)
        if pos is not None:
            contract = self.manifest.get_contract(pos.strategy_id)
            sq_time = contract.square_off_time if contract else time(15, 15)
            is_eod = (event_tod >= sq_time)

            if is_eod:
                exit_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
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

        # 5. Evaluate Alpha Contracts & Generate Signals (Only if Not in Position)
        if pos is None:
            active_contracts = self.manifest.get_contracts_for_symbol(sym)
            if not active_contracts:
                return

            candidate_signals: List[SignalEvent] = []
            df_hist = pd.DataFrame(self._history_buffers[sym]).set_index("timestamp")
            if len(df_hist) < 15:
                return

            contracts_map = {}

            for contract in active_contracts:
                # ENFORCE ENTRY WINDOW: Skip if current bar time is outside contract's qualified entry window
                if not (contract.entry_start_time <= event_tod <= contract.entry_end_time):
                    continue

                strat = self._strategy_instances.get(contract.alpha_id)
                if strat is None:
                    continue

                self.diagnostic_tracker.stats[contract.alpha_id]["bars_received"] += 1
                self.diagnostic_tracker.stats[contract.alpha_id]["generate_signals_calls"] += 1

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
                    self.diagnostic_tracker.stats[contract.alpha_id]["raw_signals"] += 1
                    candidate_signals.append(sig)
                    contracts_map[contract.alpha_id] = contract
                    self.ledger.log_signal(sig)

                except Exception as e:
                    logger.error(f"Alpha {contract.alpha_id} signal generation error: {e}")
                    continue

            if not candidate_signals:
                return

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
                if not dec.is_accepted:
                    self.diagnostic_tracker.stats[dec.alpha_id]["allocator_rejected"] += 1
                else:
                    self.diagnostic_tracker.stats[dec.alpha_id]["accepted_signals"] += 1

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
                    self.diagnostic_tracker.stats[intent.strategy_id]["final_trades"] += 1
                else:
                    # Log explicit RMS rejection in the decision ledger for full transparency
                    self.diagnostic_tracker.stats[intent.strategy_id]["risk_rejected"] += 1
                    rms_dec = DecisionEvent(
                        decision_id=f"DEC_RMS_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                        signal_id=intent.signal_id,
                        timestamp=event_time,
                        alpha_id=intent.strategy_id,
                        alpha_version=intent.alpha_version,
                        symbol=sym,
                        is_accepted=False,
                        rejection_reason=f"RMS_REJECTED: {reject_reason or 'Risk Rule Breach'}",
                    )
                    self.ledger.log_decision(rms_dec)
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

        # Write diagnostics to DB
        try:
            with sqlite3.connect("data_lake/trading_ledger.db") as conn:
                df_diag = pd.DataFrame(self.diagnostic_tracker.get_summary())
                if not df_diag.empty:
                    df_diag["timestamp"] = datetime.now().isoformat()
                    df_diag.to_sql("replay_diagnostics", conn, if_exists="append", index=False)
        except Exception as e:
            logger.error(f"Failed to write replay diagnostics: {e}")

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
