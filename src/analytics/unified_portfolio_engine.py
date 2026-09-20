"""
Ashva Unified Multi-Alpha Shared-Capital Portfolio Engine
Simulates a true shared portfolio cash pool (e.g. ₹5,00,000) across all candidate alphas,
enforcing hard concurrency constraints (max concurrent positions, e.g. 4 slots @ 25% sizing),
dynamic cash availability, trade-level capital recalculation, signal rejection logging,
consecutive loss & daily trade limits, and exact Indian regulatory tax/fee deduction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from src.analytics.indian_costs import IndianCostModel, Segment, TradeCostBreakdown
from src.analytics.metrics import calculate_profit_factor


@dataclass
class UnifiedTrade:
    trade_id: int
    alpha_id: str
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    allocated_capital: float
    gross_pnl: float
    cost_breakdown: TradeCostBreakdown
    net_pnl: float
    exit_reason: str
    roi_pct: float


@dataclass
class RejectedSignal:
    alpha_id: str
    symbol: str
    side: str
    signal_time: pd.Timestamp
    entry_price: float
    reason: str  # "CONCURRENCY_LIMIT", "INSUFFICIENT_CASH", "ZERO_QUANTITY", "STRATEGY_DAILY_LIMIT", "PORTFOLIO_DAILY_LIMIT", "CONSECUTIVE_LOSS_LOCKOUT"
    active_positions_count: int
    available_cash: float


class UnifiedPortfolioEngine:
    """
    Simulates a unified chronological execution ledger for a multi-alpha portfolio
    sharing a single pool of capital under strict capacity constraints.
    """

    def __init__(
        self,
        initial_capital: float = 500000.0,
        max_concurrent_positions: int = 4,
        capital_per_trade_pct: Optional[float] = None,
        compound_capital: bool = False,
        max_consecutive_losses: Optional[int] = None,
        max_daily_trades_per_strategy: Optional[int] = None,
        max_daily_trades_total: Optional[int] = None,
        cost_model: Optional[IndianCostModel] = None,
        segment: Segment = Segment.EQUITY_INTRADAY,
    ):
        self.initial_capital = float(initial_capital)
        self.max_concurrent_positions = max(1, int(max_concurrent_positions))
        if capital_per_trade_pct is not None:
            self.capital_per_trade_pct = float(capital_per_trade_pct)
        else:
            self.capital_per_trade_pct = 1.0 / self.max_concurrent_positions

        self.compound_capital = compound_capital
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_trades_per_strategy = max_daily_trades_per_strategy
        self.max_daily_trades_total = max_daily_trades_total
        self.cost_model = cost_model or IndianCostModel(default_slippage_bps=3.0)
        self.segment = segment

    def run_portfolio_simulation(
        self,
        candidate_trades: List[Dict[str, Any]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes chronological portfolio allocation over candidate trades.
        """
        if not candidate_trades:
            return self._empty_result(start_date, end_date)

        # Filter by period if specified
        valid_candidates = []
        s_ts = pd.to_datetime(f"{start_date} 00:00:00") if start_date else None
        e_ts = pd.to_datetime(f"{end_date} 23:59:59") if end_date else None

        for c in candidate_trades:
            ent = pd.to_datetime(c["entry_time"])
            ext = pd.to_datetime(c["exit_time"])
            if s_ts and ent < s_ts:
                continue
            if e_ts and ent > e_ts:
                continue
            valid_candidates.append({
                "alpha_id": c["alpha_id"],
                "symbol": c["symbol"],
                "side": c.get("side", "BUY"),
                "entry_time": ent,
                "exit_time": ext,
                "entry_price": float(c["entry_price"]),
                "exit_price": float(c["exit_price"]),
                "exit_reason": c.get("exit_reason", "SIGNAL"),
            })

        if not valid_candidates:
            return self._empty_result(start_date, end_date)

        # Sort candidate entries chronologically
        valid_candidates.sort(key=lambda x: (x["entry_time"], x["alpha_id"], x["symbol"]))

        # Simulation state
        cash = self.initial_capital
        active_positions: List[Dict[str, Any]] = []
        executed_trades: List[UnifiedTrade] = []
        rejected_signals: List[RejectedSignal] = []

        # Guardrails tracking
        strat_consecutive_losses: Dict[str, int] = {}
        strat_daily_trades: Dict[str, Dict[str, int]] = {}
        portfolio_daily_trades: Dict[str, int] = {}

        trade_counter = 0

        for cand in valid_candidates:
            current_time = cand["entry_time"]
            current_date_str = current_time.strftime("%Y-%m-%d")

            # Step 1: Process all active positions that have exited at or before current_time
            still_active = []
            active_positions.sort(key=lambda x: x["exit_time"])
            for pos in active_positions:
                if pos["exit_time"] <= current_time:
                    # Position is closed!
                    cash += pos["allocated_capital"] + pos["net_pnl"]
                    trade_counter += 1
                    u_trade = UnifiedTrade(
                        trade_id=trade_counter,
                        alpha_id=pos["alpha_id"],
                        symbol=pos["symbol"],
                        side=pos["side"],
                        entry_time=pos["entry_time"],
                        exit_time=pos["exit_time"],
                        entry_price=pos["entry_price"],
                        exit_price=pos["exit_price"],
                        quantity=pos["quantity"],
                        allocated_capital=pos["allocated_capital"],
                        gross_pnl=pos["gross_pnl"],
                        cost_breakdown=pos["costs"],
                        net_pnl=pos["net_pnl"],
                        exit_reason=pos["exit_reason"],
                        roi_pct=(pos["net_pnl"] / pos["allocated_capital"]) * 100.0 if pos["allocated_capital"] > 0 else 0.0,
                    )
                    executed_trades.append(u_trade)

                    # Update consecutive losses tracking
                    if pos["net_pnl"] < 0:
                        strat_consecutive_losses[pos["alpha_id"]] = strat_consecutive_losses.get(pos["alpha_id"], 0) + 1
                    else:
                        strat_consecutive_losses[pos["alpha_id"]] = 0
                else:
                    still_active.append(pos)
            active_positions = still_active

            # Step 2: Guardrail checks on new candidate
            strat_id = cand["alpha_id"]

            # Guardrail A: Consecutive Loss Lockout
            if self.max_consecutive_losses and strat_consecutive_losses.get(strat_id, 0) >= self.max_consecutive_losses:
                rejected_signals.append(RejectedSignal(
                    alpha_id=strat_id,
                    symbol=cand["symbol"],
                    side=cand["side"],
                    signal_time=current_time,
                    entry_price=cand["entry_price"],
                    reason="CONSECUTIVE_LOSS_LOCKOUT",
                    active_positions_count=len(active_positions),
                    available_cash=cash,
                ))
                continue

            # Guardrail B: Strategy Daily Trade Limit
            strat_today_count = strat_daily_trades.get(strat_id, {}).get(current_date_str, 0)
            if self.max_daily_trades_per_strategy and strat_today_count >= self.max_daily_trades_per_strategy:
                rejected_signals.append(RejectedSignal(
                    alpha_id=strat_id,
                    symbol=cand["symbol"],
                    side=cand["side"],
                    signal_time=current_time,
                    entry_price=cand["entry_price"],
                    reason="STRATEGY_DAILY_LIMIT",
                    active_positions_count=len(active_positions),
                    available_cash=cash,
                ))
                continue

            # Guardrail C: Portfolio Daily Trade Limit
            port_today_count = portfolio_daily_trades.get(current_date_str, 0)
            if self.max_daily_trades_total and port_today_count >= self.max_daily_trades_total:
                rejected_signals.append(RejectedSignal(
                    alpha_id=strat_id,
                    symbol=cand["symbol"],
                    side=cand["side"],
                    signal_time=current_time,
                    entry_price=cand["entry_price"],
                    reason="PORTFOLIO_DAILY_LIMIT",
                    active_positions_count=len(active_positions),
                    available_cash=cash,
                ))
                continue

            # Guardrail D: Concurrency Limit
            if len(active_positions) >= self.max_concurrent_positions:
                rejected_signals.append(RejectedSignal(
                    alpha_id=strat_id,
                    symbol=cand["symbol"],
                    side=cand["side"],
                    signal_time=current_time,
                    entry_price=cand["entry_price"],
                    reason="CONCURRENCY_LIMIT",
                    active_positions_count=len(active_positions),
                    available_cash=cash,
                ))
                continue

            # Guardrail E: Capital Allocation & Cash Availability
            base_cap = cash if self.compound_capital else self.initial_capital
            target_trade_cap = base_cap * self.capital_per_trade_pct
            alloc_cap = min(target_trade_cap, cash)

            if alloc_cap < cand["entry_price"]:
                rejected_signals.append(RejectedSignal(
                    alpha_id=strat_id,
                    symbol=cand["symbol"],
                    side=cand["side"],
                    signal_time=current_time,
                    entry_price=cand["entry_price"],
                    reason="INSUFFICIENT_CASH",
                    active_positions_count=len(active_positions),
                    available_cash=cash,
                ))
                continue

            quantity = int(alloc_cap // cand["entry_price"])
            if quantity <= 0:
                rejected_signals.append(RejectedSignal(
                    alpha_id=strat_id,
                    symbol=cand["symbol"],
                    side=cand["side"],
                    signal_time=current_time,
                    entry_price=cand["entry_price"],
                    reason="ZERO_QUANTITY",
                    active_positions_count=len(active_positions),
                    available_cash=cash,
                ))
                continue

            actual_allocated = quantity * cand["entry_price"]

            # Calculate exact trade PnL and Indian statutory friction
            is_buy = (cand["side"] == "BUY")
            is_stop_loss = ("STOP" in cand["exit_reason"].upper() or "SL" in cand["exit_reason"].upper())
            if is_buy:
                buy_px = cand["entry_price"]
                sell_px = cand["exit_price"]
            else:
                buy_px = cand["exit_price"]
                sell_px = cand["entry_price"]

            costs = self.cost_model.calculate_trade_costs(
                buy_price=buy_px,
                sell_price=sell_px,
                quantity=quantity,
                segment=self.segment,
                is_stop_loss=is_stop_loss,
            )
            gross_pnl = costs.gross_pnl
            net_pnl = costs.net_pnl

            # Lock capital and register active position
            cash -= actual_allocated
            active_positions.append({
                "alpha_id": cand["alpha_id"],
                "symbol": cand["symbol"],
                "side": cand["side"],
                "entry_time": cand["entry_time"],
                "exit_time": cand["exit_time"],
                "entry_price": cand["entry_price"],
                "exit_price": cand["exit_price"],
                "quantity": quantity,
                "allocated_capital": actual_allocated,
                "gross_pnl": gross_pnl,
                "costs": costs,
                "net_pnl": net_pnl,
                "exit_reason": cand["exit_reason"],
            })

            # Update daily counts
            if strat_id not in strat_daily_trades:
                strat_daily_trades[strat_id] = {}
            strat_daily_trades[strat_id][current_date_str] = strat_today_count + 1
            portfolio_daily_trades[current_date_str] = port_today_count + 1

        # Step 3: Flush remaining open positions at simulation end
        active_positions.sort(key=lambda x: x["exit_time"])
        for pos in active_positions:
            cash += pos["allocated_capital"] + pos["net_pnl"]
            trade_counter += 1
            u_trade = UnifiedTrade(
                trade_id=trade_counter,
                alpha_id=pos["alpha_id"],
                symbol=pos["symbol"],
                side=pos["side"],
                entry_time=pos["entry_time"],
                exit_time=pos["exit_time"],
                entry_price=pos["entry_price"],
                exit_price=pos["exit_price"],
                quantity=pos["quantity"],
                allocated_capital=pos["allocated_capital"],
                gross_pnl=pos["gross_pnl"],
                cost_breakdown=pos["costs"],
                net_pnl=pos["net_pnl"],
                exit_reason=pos["exit_reason"],
                roi_pct=(pos["net_pnl"] / pos["allocated_capital"]) * 100.0 if pos["allocated_capital"] > 0 else 0.0,
            )
            executed_trades.append(u_trade)

        # Sort all executed trades by exit_time
        executed_trades.sort(key=lambda x: x.exit_time)

        # Compile performance metrics
        total_executed = len(executed_trades)
        total_rejected = len(rejected_signals)
        total_candidates = total_executed + total_rejected

        total_gross = sum(t.gross_pnl for t in executed_trades)
        total_taxes = sum(t.cost_breakdown.total_tax_and_charges for t in executed_trades)
        total_net = sum(t.net_pnl for t in executed_trades)
        wins = [t for t in executed_trades if t.net_pnl > 0]
        losses = [t for t in executed_trades if t.net_pnl <= 0]
        win_rate = (len(wins) / max(1, total_executed)) * 100.0
        profit_factor = calculate_profit_factor([t.net_pnl for t in executed_trades]) if total_executed > 0 else 0.0
        net_roi_pct = (total_net / self.initial_capital) * 100.0

        # Build clean closed-trade equity curve and drawdown profile
        eq_records = [{"time": valid_candidates[0]["entry_time"], "equity": self.initial_capital}]
        running_equity = self.initial_capital
        for t in executed_trades:
            running_equity += t.net_pnl
            eq_records.append({"time": t.exit_time, "equity": running_equity})

        eq_df = pd.DataFrame(eq_records)
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["dd_inr"] = eq_df["equity"] - eq_df["peak"]
        eq_df["dd_pct"] = (eq_df["dd_inr"] / eq_df["peak"]) * 100.0
        max_dd_inr = abs(float(eq_df["dd_inr"].min()))
        max_dd_pct = abs(float(eq_df["dd_pct"].min()))

        # Sharpe ratio from daily PnL
        daily_rets = []
        if total_executed > 0:
            trade_df = pd.DataFrame([{"time": t.exit_time, "net_pnl": t.net_pnl} for t in executed_trades])
            trade_df["date"] = trade_df["time"].dt.date
            daily_pnl = trade_df.groupby("date")["net_pnl"].sum()
            daily_rets = (daily_pnl / self.initial_capital).values
        std_ret = float(np.std(daily_rets)) if len(daily_rets) > 1 else 0.0
        mean_ret = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0.0
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-7 else 0.0

        # Strategy breakdown
        strategy_stats = {}
        for t in executed_trades:
            if t.alpha_id not in strategy_stats:
                strategy_stats[t.alpha_id] = {"trades": 0, "wins": 0, "gross": 0.0, "costs": 0.0, "net": 0.0}
            strategy_stats[t.alpha_id]["trades"] += 1
            if t.net_pnl > 0:
                strategy_stats[t.alpha_id]["wins"] += 1
            strategy_stats[t.alpha_id]["gross"] += t.gross_pnl
            strategy_stats[t.alpha_id]["costs"] += t.cost_breakdown.total_tax_and_charges
            strategy_stats[t.alpha_id]["net"] += t.net_pnl

        for s_id, s_data in strategy_stats.items():
            s_data["win_rate"] = (s_data["wins"] / max(1, s_data["trades"])) * 100.0
            s_data["net_pf"] = calculate_profit_factor([t.net_pnl for t in executed_trades if t.alpha_id == s_id])

        # Rejection breakdown
        rejection_breakdown = {}
        for r in rejected_signals:
            rejection_breakdown[r.reason] = rejection_breakdown.get(r.reason, 0) + 1

        return {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": self.initial_capital,
            "final_equity": running_equity,
            "total_candidates": total_candidates,
            "total_executed_trades": total_executed,
            "total_rejected_signals": total_rejected,
            "acceptance_rate_pct": (total_executed / max(1, total_candidates)) * 100.0,
            "gross_pnl": total_gross,
            "total_taxes_and_costs": total_taxes,
            "net_pnl": total_net,
            "roi_pct": net_roi_pct,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown_inr": max_dd_inr,
            "max_drawdown_pct": max_dd_pct,
            "strategy_breakdown": strategy_stats,
            "rejection_breakdown": rejection_breakdown,
            "executed_trades": executed_trades,
            "rejected_signals": rejected_signals,
            "equity_curve": eq_df,
        }

    def _empty_result(self, start_date: Optional[str], end_date: Optional[str]) -> Dict[str, Any]:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": self.initial_capital,
            "final_equity": self.initial_capital,
            "total_candidates": 0,
            "total_executed_trades": 0,
            "total_rejected_signals": 0,
            "acceptance_rate_pct": 0.0,
            "gross_pnl": 0.0,
            "total_taxes_and_costs": 0.0,
            "net_pnl": 0.0,
            "roi_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_inr": 0.0,
            "max_drawdown_pct": 0.0,
            "strategy_breakdown": {},
            "rejection_breakdown": {},
            "executed_trades": [],
            "rejected_signals": [],
            "equity_curve": pd.DataFrame(),
        }
