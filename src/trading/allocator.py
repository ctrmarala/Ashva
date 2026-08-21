"""
Ashva Multi-Alpha Capital Allocator
Deterministically allocates capital across competing alpha signals on identical or different symbols.
Preserves all candidate signals, evaluates priority and risk capacity, and records complete decision rationale.
Enforces strict stop-loss requirements without silent arbitrary fallbacks.
"""

from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any

from src.core.events import (
    SignalEvent, SignalType, OrderIntent, OrderSide, OrderType,
    ProductType, DecisionEvent,
)
from src.trading.contract import QualifiedAlphaContract
from src.trading.position_manager import PositionManager
from src.trading.portfolio_state import PortfolioState

logger = logging.getLogger("Ashva.MultiAlphaAllocator")


class MultiAlphaAllocator:
    """
    Evaluates competing signals across active alphas and computes deterministic capital allocations.
    """

    def __init__(self, default_min_risk_budget: float = 500.0, max_portfolio_risk_pct: float = 0.05):
        self.default_min_risk_budget = default_min_risk_budget
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.decision_history: List[DecisionEvent] = []

    def allocate(
        self,
        candidate_signals: List[SignalEvent],
        contracts_map: Dict[str, QualifiedAlphaContract],
        current_prices: Dict[str, float],
        position_manager: PositionManager,
        portfolio_state: PortfolioState,
    ) -> Tuple[List[OrderIntent], List[DecisionEvent]]:
        """
        Processes candidate signals, groups by symbol, resolves competition deterministically,
        and generates OrderIntents and DecisionEvents.
        """
        intents: List[OrderIntent] = []
        decisions: List[DecisionEvent] = []

        if not candidate_signals:
            return intents, decisions

        current_equity = portfolio_state.current_equity
        max_total_risk = current_equity * self.max_portfolio_risk_pct

        # Calculate current deployed risk from open positions
        current_deployed_risk = sum(
            pos.quantity * pos.stop_dist for pos in position_manager.get_all_positions()
        )
        remaining_portfolio_risk = max(0.0, max_total_risk - current_deployed_risk)

        # Group candidate signals by symbol
        signals_by_sym: Dict[str, List[SignalEvent]] = {}
        for sig in candidate_signals:
            sym = sig.symbol.upper()
            if sym not in signals_by_sym:
                signals_by_sym[sym] = []
            signals_by_sym[sym].append(sig)

        # Stage 1: Determine symbol-level winners
        symbol_winners: List[Tuple[float, float, SignalEvent, QualifiedAlphaContract, List[SignalEvent]]] = []

        for sym, sym_signals in signals_by_sym.items():
            current_price = current_prices.get(sym)
            if current_price is None or current_price <= 0:
                for sig in sym_signals:
                    dec = DecisionEvent(
                        decision_id=f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                        signal_id=sig.signal_id,
                        timestamp=sig.timestamp,
                        alpha_id=sig.strategy_id,
                        alpha_version=sig.alpha_version,
                        symbol=sym,
                        is_accepted=False,
                        rejection_reason="MARKET_PRICE_UNAVAILABLE",
                        competing_alphas=[s.strategy_id for s in sym_signals],
                    )
                    decisions.append(dec)
                continue

            existing_pos = position_manager.get_position(sym)
            if existing_pos is not None:
                for sig in sym_signals:
                    dec = DecisionEvent(
                        decision_id=f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                        signal_id=sig.signal_id,
                        timestamp=sig.timestamp,
                        alpha_id=sig.strategy_id,
                        alpha_version=sig.alpha_version,
                        symbol=sym,
                        is_accepted=False,
                        rejection_reason=f"EXISTING_POSITION_ACTIVE (Qty={existing_pos.quantity}, Side={existing_pos.side.value})",
                        competing_alphas=[s.strategy_id for s in sym_signals],
                    )
                    decisions.append(dec)
                continue

            # Rank signals by contract priority score descending, then by signal confidence descending
            ranked_signals = []
            for s in sym_signals:
                contract = contracts_map.get(s.strategy_id)
                priority = contract.priority_score if contract else 1.0
                ranked_signals.append((priority, s.confidence, s))

            ranked_signals.sort(key=lambda x: (x[0], x[1]), reverse=True)

            winning_priority, winning_conf, winning_signal = ranked_signals[0]
            winning_contract = contracts_map.get(winning_signal.strategy_id)

            if winning_contract is None or winning_contract.status != "ACTIVE":
                for _, _, sig in ranked_signals:
                    dec = DecisionEvent(
                        decision_id=f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                        signal_id=sig.signal_id,
                        timestamp=sig.timestamp,
                        alpha_id=sig.strategy_id,
                        alpha_version=sig.alpha_version,
                        symbol=sym,
                        is_accepted=False,
                        rejection_reason="ALPHA_CONTRACT_INACTIVE_OR_UNAVAILABLE",
                        competing_alphas=[s.strategy_id for s in sym_signals],
                    )
                    decisions.append(dec)
                continue

            # Reject non-winning candidates on this symbol
            for prio, conf, rejected_sig in ranked_signals[1:]:
                rej_dec = DecisionEvent(
                    decision_id=f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                    signal_id=rejected_sig.signal_id,
                    timestamp=rejected_sig.timestamp,
                    alpha_id=rejected_sig.strategy_id,
                    alpha_version=rejected_sig.alpha_version,
                    symbol=sym,
                    is_accepted=False,
                    rejection_reason=f"LOWER_PRIORITY_SCORE (Selected: {winning_signal.strategy_id} Priority={winning_priority} vs {prio})",
                    competing_alphas=[s.strategy_id for s in sym_signals],
                )
                decisions.append(rej_dec)

            symbol_winners.append((winning_priority, winning_conf, winning_signal, winning_contract, sym_signals))

        # Stage 2: Cross-symbol portfolio-level risk rationing
        # Sort winners across all symbols by priority descending
        symbol_winners.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for winning_priority, winning_conf, winning_signal, winning_contract, sym_signals in symbol_winners:
            sym = winning_signal.symbol.upper()
            current_price = current_prices[sym]
            side = OrderSide.BUY if winning_signal.signal_type == SignalType.LONG else OrderSide.SELL
            sl_val = winning_signal.suggested_stop_loss
            tp_val = winning_signal.suggested_take_profit

            # Strict Stop Loss Validation (NO silent arbitrary 1% guessing)
            if sl_val is None or sl_val <= 0:
                if winning_contract.stop_loss_pct is not None and winning_contract.stop_loss_pct > 0:
                    sl_val = current_price * (1.0 - winning_contract.stop_loss_pct) if side == OrderSide.BUY else current_price * (1.0 + winning_contract.stop_loss_pct)
                else:
                    dec = DecisionEvent(
                        decision_id=f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                        signal_id=winning_signal.signal_id,
                        timestamp=winning_signal.timestamp,
                        alpha_id=winning_signal.strategy_id,
                        alpha_version=winning_signal.alpha_version,
                        symbol=sym,
                        is_accepted=False,
                        rejection_reason="MISSING_STOP_LOSS_DEFINITION (Strategy did not provide stop and contract has no stop_loss_pct)",
                        competing_alphas=[s.strategy_id for s in sym_signals],
                    )
                    decisions.append(dec)
                    continue

            stop_dist = max(0.05, abs(current_price - sl_val))
            trade_risk_budget = max(self.default_min_risk_budget, current_equity * winning_contract.risk_per_trade_pct)

            # Check if remaining portfolio risk capacity is exhausted
            if trade_risk_budget > remaining_portfolio_risk and remaining_portfolio_risk <= self.default_min_risk_budget:
                dec = DecisionEvent(
                    decision_id=f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                    signal_id=winning_signal.signal_id,
                    timestamp=winning_signal.timestamp,
                    alpha_id=winning_signal.strategy_id,
                    alpha_version=winning_signal.alpha_version,
                    symbol=sym,
                    is_accepted=False,
                    rejection_reason=f"PORTFOLIO_RISK_CAPACITY_EXHAUSTED (Remaining: Rs {remaining_portfolio_risk:,.2f} < Needed: Rs {trade_risk_budget:,.2f})",
                    competing_alphas=[s.strategy_id for s in sym_signals],
                )
                decisions.append(dec)
                continue

            qty_from_risk = int(trade_risk_budget / stop_dist)
            max_cap_qty = int((current_equity * winning_contract.max_capital_allocation_pct) / current_price)
            allocated_qty = max(1, min(qty_from_risk, max_cap_qty))
            actual_risk_taken = allocated_qty * stop_dist

            remaining_portfolio_risk -= actual_risk_taken
            decision_id = f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"

            winning_dec = DecisionEvent(
                decision_id=decision_id,
                signal_id=winning_signal.signal_id,
                timestamp=winning_signal.timestamp,
                alpha_id=winning_signal.strategy_id,
                alpha_version=winning_signal.alpha_version,
                symbol=sym,
                is_accepted=True,
                allocated_quantity=allocated_qty,
                risk_budget=trade_risk_budget,
                competing_alphas=[s.strategy_id for s in sym_signals],
            )
            decisions.append(winning_dec)

            intent = OrderIntent(
                strategy_id=winning_signal.strategy_id,
                symbol=sym,
                side=side,
                quantity=allocated_qty,
                order_type=OrderType.MARKET,
                product_type=ProductType.INTRADAY,
                is_reduce_only=False,
                alpha_version=winning_signal.alpha_version,
                signal_id=winning_signal.signal_id,
                decision_id=decision_id,
                stop_loss=sl_val,
                take_profit=tp_val,
                tag=winning_signal.strategy_id,
                timestamp=winning_signal.timestamp,
            )
            intents.append(intent)

        self.decision_history.extend(decisions)
        return intents, decisions
