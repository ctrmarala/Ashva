"""
Ashva Multi-Alpha Capital Allocator
Deterministically allocates capital across competing alpha signals on identical or different symbols.
Preserves all candidate signals, evaluates priority and risk capacity, and records complete decision rationale.
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

    def __init__(self, default_min_risk_budget: float = 500.0):
        self.default_min_risk_budget = default_min_risk_budget
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

        # Group candidate signals by symbol
        signals_by_sym: Dict[str, List[SignalEvent]] = {}
        for sig in candidate_signals:
            sym = sig.symbol.upper()
            if sym not in signals_by_sym:
                signals_by_sym[sym] = []
            signals_by_sym[sym].append(sig)

        current_equity = portfolio_state.current_equity

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
                # Already have an open position in this symbol
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

            # Check for opposing signals on same symbol (e.g. A31 Short vs A67 Long)
            long_signals = [s for s in sym_signals if s.signal_type == SignalType.LONG]
            short_signals = [s for s in sym_signals if s.signal_type == SignalType.SHORT]

            if long_signals and short_signals:
                logger.warning(f"Conflicting signals on {sym}: {len(long_signals)} LONG vs {len(short_signals)} SHORT")

            # Rank signals by contract priority score descending, then by signal confidence descending
            ranked_signals = []
            for s in sym_signals:
                contract = contracts_map.get(s.strategy_id)
                priority = contract.priority_score if contract else 1.0
                ranked_signals.append((priority, s.confidence, s))

            ranked_signals.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Winner is the highest ranked candidate
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

            # Compute Sizing for the Winning Signal
            side = OrderSide.BUY if winning_signal.signal_type == SignalType.LONG else OrderSide.SELL
            sl_val = winning_signal.suggested_stop_loss
            tp_val = winning_signal.suggested_take_profit

            # Stop loss fallback if strategy didn't provide one
            if sl_val is None or sl_val <= 0:
                if winning_contract.stop_loss_pct is not None:
                    sl_val = current_price * (1.0 - winning_contract.stop_loss_pct) if side == OrderSide.BUY else current_price * (1.0 + winning_contract.stop_loss_pct)
                else:
                    sl_val = current_price * 0.99 if side == OrderSide.BUY else current_price * 1.01

            stop_dist = max(0.05, abs(current_price - sl_val))
            risk_budget = max(self.default_min_risk_budget, current_equity * winning_contract.risk_per_trade_pct)
            qty_from_risk = int(risk_budget / stop_dist)
            max_cap_qty = int((current_equity * winning_contract.max_capital_allocation_pct) / current_price)
            allocated_qty = max(1, min(qty_from_risk, max_cap_qty))

            decision_id = f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"

            # Accepted decision for the winner
            winning_dec = DecisionEvent(
                decision_id=decision_id,
                signal_id=winning_signal.signal_id,
                timestamp=winning_signal.timestamp,
                alpha_id=winning_signal.strategy_id,
                alpha_version=winning_signal.alpha_version,
                symbol=sym,
                is_accepted=True,
                allocated_quantity=allocated_qty,
                risk_budget=risk_budget,
                competing_alphas=[s.strategy_id for s in sym_signals],
            )
            decisions.append(winning_dec)

            # Rejected decisions for remaining candidates on this symbol
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

            # Generate OrderIntent for the approved winner
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
