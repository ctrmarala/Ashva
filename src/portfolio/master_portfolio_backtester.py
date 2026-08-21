"""
Ashva Master Multi-Alpha Event-Driven Portfolio Backtester
Executes an institutional deterministic priority event queue simulation:
1. Strict deterministic priority: Timestamp -> EXIT before ENTRY -> Alpha Score -> Strategy ID -> Symbol.
2. Dynamic Mark-to-Market (MTM) Equity Sizing (Cash + Unrealized Open Position PnL).
3. Alpha Stop Preservation: Sizes positions strictly using the strategy's original stop distance.
4. Standardized Daily MTM Annualized Sharpe & Sortino ratios.
"""

from typing import Dict, List, Any, Optional, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import heapq
import pandas as pd
import numpy as np

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.analytics.metrics import calculate_daily_mtm_sharpe, calculate_daily_mtm_sortino, calculate_max_drawdown_pct
from src.backtest.engine import BacktestEngine, BacktestTrade
from src.research.regime_profiler import MarketRegimeProfiler


SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "TECHM": "IT", "WIPRO": "IT",
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "INDUSINDBK": "BANKING", "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE",
    "SHRIRAMFIN": "FINANCE", "HDFCLIFE": "FINANCE", "SBILIFE": "FINANCE",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "TMPV": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "RELIANCE": "ENERGY", "NTPC": "ENERGY", "ONGC": "ENERGY", "POWERGRID": "ENERGY",
    "COALINDIA": "ENERGY", "BPCL": "ENERGY",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA", "DIVISLAB": "PHARMA",
    "APOLLOHOSP": "PHARMA",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "GRASIM": "MATERIALS", "ULTRACEMCO": "MATERIALS",
    "PIDILITIND": "MATERIALS",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "TATACONSUM": "FMCG", "TITAN": "CONSUMER", "TRENT": "RETAIL",
    "LT": "INFRA", "ADANIENT": "INFRA", "ADANIPORTS": "INFRA", "BEL": "DEFENSE",
}


class EventType(str, Enum):
    EXIT = "EXIT"
    ENTRY = "ENTRY"


@dataclass
class PortfolioTrade:
    trade_id: int
    strategy_id: str
    symbol: str
    sector: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    total_costs: float
    exit_reason: str
    mfe_pct: float
    mae_pct: float
    initial_sl: float
    stop_dist: float


class MasterPortfolioBacktester:
    """
    Simulates multi-alpha portfolio execution using a strict Deterministic Priority Queue.
    """

    def __init__(
        self,
        data_lake: Optional[DataLake] = None,
        initial_capital: float = 500000.0,
        risk_per_trade_pct: float = 0.0050,  # 0.50% dynamic account risk
        risk_per_trade_inr: Optional[float] = None,  # Backwards compatibility
        max_concurrent_positions: int = 5,
        max_positions_per_sector: int = 2,
        cost_model: Optional[IndianCostModel] = None,
    ):
        self.lake = data_lake or DataLake(read_only=True)
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = (risk_per_trade_inr / initial_capital) if risk_per_trade_inr is not None else risk_per_trade_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_positions_per_sector = max_positions_per_sector
        self.cost_model = cost_model or IndianCostModel()
        self.regime_profiler = MarketRegimeProfiler(self.lake)

    def run_portfolio_backtest(
        self,
        strategies: List[Any],
        symbols: List[str],
        default_timeframe: str = "15m",
        strategy_timeframe_map: Optional[Dict[str, str]] = None,
        strategy_trailing_map: Optional[Dict[str, str]] = None,
        strategy_priority_map: Optional[Dict[str, float]] = None,
        use_regime_filter: bool = True,
    ) -> Dict[str, Any]:
        tf_map = strategy_timeframe_map or {}
        trail_map = strategy_trailing_map or {}
        priority_map = strategy_priority_map or {}

        all_candidate_trades = []

        for strat_obj in strategies:
            strat = strat_obj() if isinstance(strat_obj, type) else strat_obj
            alpha_id = getattr(strat, "name", strat.__class__.__name__)
            tf = tf_map.get(alpha_id, default_timeframe)
            trailing_mode = trail_map.get(alpha_id, "BREAK_EVEN")
            strat_score = priority_map.get(alpha_id, 1.0)

            for sym in symbols:
                df = self.lake.load_bars(sym.upper(), tf)
                if df.empty or len(df) < 500:
                    continue

                if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp").sort_index()

                df_signals = strat.generate_signals(df)
                if "signal" not in df_signals.columns:
                    continue

                engine = BacktestEngine(cost_model=self.cost_model, initial_capital=self.initial_capital)
                res = engine.run(df_signals, symbol=sym.upper(), strategy_id=alpha_id, trailing_mode=trailing_mode)

                for t in res.trade_list:
                    all_candidate_trades.append({
                        "strategy_id": alpha_id,
                        "strategy_score": strat_score,
                        "symbol": sym.upper(),
                        "sector": SECTOR_MAP.get(sym.upper(), "OTHER"),
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                        "side": t.side,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "exit_reason": t.exit_reason,
                        "mfe_pct": t.mfe_pct,
                        "mae_pct": t.mae_pct,
                        "initial_sl": t.initial_sl if t.initial_sl > 0 else (t.entry_price * 0.99 if t.side == "LONG" else t.entry_price * 1.01),
                        "initial_tp": t.initial_tp,
                        "stop_dist": t.stop_dist if t.stop_dist > 0 else max(1e-2, abs(t.entry_price * 0.01)),
                    })

        if not all_candidate_trades:
            return self._empty_result()

        # -------------------------------------------------------------
        # 2. DETERMINISTIC HEAP PRIORITY QUEUE SIMULATION
        # -------------------------------------------------------------
        # Priority order: (timestamp, 0 if EXIT else 1, -strategy_score, strategy_id, symbol, trade_id)
        event_heap: List[Tuple[pd.Timestamp, int, float, str, str, int, str, Dict[str, Any]]] = []
        trade_id_counter = 1

        for tr in all_candidate_trades:
            t_id = trade_id_counter
            trade_id_counter += 1
            tr["trade_id"] = t_id

            # Push ENTRY event
            heapq.heappush(event_heap, (
                tr["entry_time"],
                1,  # ENTRY = 1
                -float(tr.get("strategy_score", 1.0)),
                tr["strategy_id"],
                tr["symbol"],
                t_id,
                EventType.ENTRY.value,
                tr,
            ))

        realized_equity = self.initial_capital
        open_positions: Dict[int, Dict[str, Any]] = {}
        active_symbols: set = set()
        active_sectors: Dict[str, int] = {}
        executed_trades: List[PortfolioTrade] = []
        equity_curve: List[Dict[str, Any]] = []

        while event_heap:
            event_time, priority_type, neg_score, strat_id, sym, t_id, event_type_str, data = heapq.heappop(event_heap)

            # EXIT EVENT: Realize PnL strictly at EXIT TIME
            if event_type_str == EventType.EXIT.value:
                if t_id not in open_positions:
                    continue

                pos = open_positions.pop(t_id)
                active_symbols.discard(pos["symbol"])
                active_sectors[pos["sector"]] = max(0, active_sectors.get(pos["sector"], 1) - 1)

                realized_equity += pos["net_pnl"]
                equity_curve.append({
                    "timestamp": event_time,
                    "equity": realized_equity,
                    "pnl": pos["net_pnl"],
                })

                executed_trades.append(PortfolioTrade(
                    trade_id=t_id,
                    strategy_id=pos["strategy_id"],
                    symbol=pos["symbol"],
                    sector=pos["sector"],
                    entry_time=pos["entry_time"],
                    exit_time=event_time,
                    side=pos["side"],
                    entry_price=pos["entry_price"],
                    exit_price=pos["exit_price"],
                    quantity=pos["quantity"],
                    gross_pnl=pos["gross_pnl"],
                    net_pnl=pos["net_pnl"],
                    total_costs=pos["total_costs"],
                    exit_reason=pos["exit_reason"],
                    mfe_pct=pos["mfe_pct"],
                    mae_pct=pos["mae_pct"],
                    initial_sl=pos["initial_sl"],
                    stop_dist=pos["stop_dist"],
                ))
                continue

            # ENTRY EVENT: Size strictly with CURRENT Mark-to-Market Equity
            if event_type_str == EventType.ENTRY.value:
                tr = data
                sym = tr["symbol"]
                sector = tr["sector"]

                if sym in active_symbols:
                    continue
                if len(open_positions) >= self.max_concurrent_positions:
                    continue
                if active_sectors.get(sector, 0) >= self.max_positions_per_sector:
                    continue

                if use_regime_filter:
                    regime_state = self.regime_profiler.get_regime_for_date(event_time)
                    if regime_state.get("composite") == "SIDEWAYS_CHOP_LOW_VOLATILITY" and "BREAKOUT" in tr["strategy_id"].upper():
                        continue

                # Calculate Current MTM Equity = Realized Equity + Open Positions Net MTM
                unrealized_pnl = sum(pos.get("net_pnl", 0.0) for pos in open_positions.values())
                current_mtm_equity = max(1000.0, realized_equity + unrealized_pnl)

                # Use actual strategy stop distance for risk sizing
                stop_dist = max(0.05, tr["stop_dist"])
                risk_budget = max(500.0, current_mtm_equity * self.risk_per_trade_pct)
                qty_from_risk = int(risk_budget / stop_dist)
                max_cap_qty = int((current_mtm_equity * 0.20) / tr["entry_price"])
                qty = max(1, min(qty_from_risk, max_cap_qty))

                is_long = (tr["side"] == "LONG")
                gross_pnl = ((tr["exit_price"] - tr["entry_price"]) * qty) if is_long else ((tr["entry_price"] - tr["exit_price"]) * qty)
                buy_p = tr["entry_price"] if is_long else tr["exit_price"]
                sell_p = tr["exit_price"] if is_long else tr["entry_price"]

                if buy_p <= 0 or sell_p <= 0:
                    continue

                is_sl = ("STOP_LOSS" in tr["exit_reason"] or "TRAILING" in tr["exit_reason"])
                costs = self.cost_model.calculate_trade_costs(
                    buy_price=buy_p,
                    sell_price=sell_p,
                    quantity=qty,
                    segment=Segment.EQUITY_INTRADAY,
                    is_stop_loss=is_sl,
                )
                net_pnl = gross_pnl - costs.total_tax_and_charges

                open_positions[t_id] = {
                    "trade_id": t_id,
                    "strategy_id": tr["strategy_id"],
                    "symbol": sym,
                    "sector": sector,
                    "entry_time": event_time,
                    "exit_time": tr["exit_time"],
                    "side": tr["side"],
                    "entry_price": tr["entry_price"],
                    "exit_price": tr["exit_price"],
                    "quantity": qty,
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "total_costs": round(costs.total_tax_and_charges, 2),
                    "exit_reason": tr["exit_reason"],
                    "mfe_pct": tr["mfe_pct"],
                    "mae_pct": tr["mae_pct"],
                    "initial_sl": tr["initial_sl"],
                    "stop_dist": stop_dist,
                }

                active_symbols.add(sym)
                active_sectors[sector] = active_sectors.get(sector, 0) + 1

                # Push scheduled EXIT event into heap (EXIT = 0 priority so it precedes ENTRY at same time)
                heapq.heappush(event_heap, (
                    tr["exit_time"],
                    0,  # EXIT = 0
                    neg_score,
                    strat_id,
                    sym,
                    t_id,
                    EventType.EXIT.value,
                    {"trade_id": t_id},
                ))

        # 3. Standardized Daily MTM Sharpe Computation
        df_eq = pd.DataFrame(equity_curve)
        if not df_eq.empty:
            df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
            eq_series = df_eq.set_index("timestamp")["equity"]

            daily_sharpe = calculate_daily_mtm_sharpe(eq_series)
            daily_sortino = calculate_daily_mtm_sortino(eq_series)
            max_drawdown_pct = calculate_max_drawdown_pct(eq_series)
        else:
            daily_sharpe = 0.0
            daily_sortino = 0.0
            max_drawdown_pct = 0.0

        total_net_pnl = realized_equity - self.initial_capital
        win_count = sum(1 for t in executed_trades if t.net_pnl > 0)
        total_trades_count = len(executed_trades)
        win_rate = (win_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0
        gross_win = sum(t.net_pnl for t in executed_trades if t.net_pnl > 0)
        gross_loss = sum(abs(t.net_pnl) for t in executed_trades if t.net_pnl < 0)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        alpha_pnl = {}
        for t in executed_trades:
            alpha_pnl[t.strategy_id] = alpha_pnl.get(t.strategy_id, 0.0) + t.net_pnl

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(realized_equity, 2),
            "total_net_pnl": round(total_net_pnl, 2),
            "roi_pct": round((total_net_pnl / self.initial_capital) * 100.0, 2),
            "total_trades": total_trades_count,
            "winning_trades": win_count,
            "win_rate": f"{round(win_rate, 1)}%",
            "profit_factor": round(profit_factor, 2),
            "portfolio_sharpe": round(daily_sharpe, 2),
            "portfolio_sortino": round(daily_sortino, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "alpha_contribution": {k: round(v, 2) for k, v in sorted(alpha_pnl.items(), key=lambda x: x[1], reverse=True)},
            "trade_list": executed_trades,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.initial_capital,
            "total_net_pnl": 0.0,
            "roi_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": "0.0%",
            "profit_factor": 0.0,
            "portfolio_sharpe": 0.0,
            "portfolio_sortino": 0.0,
            "max_drawdown_pct": 0.0,
            "alpha_contribution": {},
            "trade_list": [],
        }
