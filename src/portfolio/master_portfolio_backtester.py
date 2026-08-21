"""
Ashva Master Multi-Alpha Event-Driven Portfolio Backtester
Executes an institutional event-driven simulation (ENTRY_EVENT & EXIT_EVENT)
to eliminate PnL realization lookahead bias and enforce dynamic compounding,
sector caps, single-symbol collision locks, and regime gating.
"""

from typing import Dict, List, Any, Optional, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
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
    ENTRY = "ENTRY"
    EXIT = "EXIT"


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


class MasterPortfolioBacktester:
    """
    Simulates multi-alpha portfolio execution using a strict Event-Driven architecture.
    Guarantees PnL is realized ONLY upon position exit (zero capital lookahead).
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
        use_regime_filter: bool = True,
    ) -> Dict[str, Any]:
        tf_map = strategy_timeframe_map or {}
        trail_map = strategy_trailing_map or {}

        all_candidate_trades = []

        for strat_obj in strategies:
            strat = strat_obj() if isinstance(strat_obj, type) else strat_obj
            alpha_id = getattr(strat, "name", strat.__class__.__name__)
            tf = tf_map.get(alpha_id, default_timeframe)
            trailing_mode = trail_map.get(alpha_id, "BREAK_EVEN")

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
                        "initial_sl": t.entry_price * 0.99 if t.side == "LONG" else t.entry_price * 1.01,
                    })

        if not all_candidate_trades:
            return self._empty_result()

        # -------------------------------------------------------------
        # 2. EVENT-DRIVEN QUEUE SIMULATION (Zero Future PnL Leakage)
        # -------------------------------------------------------------
        events = []
        for tr in all_candidate_trades:
            events.append((tr["entry_time"], EventType.ENTRY, tr))

        events.sort(key=lambda x: (x[0], 0 if x[1] == EventType.EXIT else 1))

        realized_equity = self.initial_capital
        open_positions: Dict[int, Dict[str, Any]] = {}
        active_symbols: set = set()
        active_sectors: Dict[str, int] = {}
        executed_trades: List[PortfolioTrade] = []
        equity_curve: List[Dict[str, Any]] = []
        trade_id_counter = 1

        event_queue = events
        i = 0
        while i < len(event_queue):
            event_queue.sort(key=lambda x: (x[0], 0 if x[1] == EventType.EXIT else 1))
            event_time, event_type, data = event_queue[i]
            i += 1

            # EXIT EVENT: Realize PnL strictly at EXIT TIME
            if event_type == EventType.EXIT:
                t_id = data["trade_id"]
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
                ))
                continue

            # ENTRY EVENT: Size strictly with CURRENT Realized Equity
            if event_type == EventType.ENTRY:
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

                stop_dist = max(1e-2, abs(tr["entry_price"] - tr["initial_sl"]))
                risk_budget = max(500.0, realized_equity * self.risk_per_trade_pct)
                qty_from_risk = int(risk_budget / stop_dist)
                max_cap_qty = int((realized_equity * 0.20) / tr["entry_price"])
                qty = max(1, min(qty_from_risk, max_cap_qty))

                is_long = (tr["side"] == "LONG")
                gross_pnl = ((tr["exit_price"] - tr["entry_price"]) * qty) if is_long else ((tr["entry_price"] - tr["exit_price"]) * qty)
                buy_p = tr["entry_price"] if is_long else tr["exit_price"]
                sell_p = tr["exit_price"] if is_long else tr["entry_price"]

                if buy_p <= 0 or sell_p <= 0:
                    continue

                costs = self.cost_model.calculate_trade_costs(buy_price=buy_p, sell_price=sell_p, quantity=qty, segment=Segment.EQUITY_INTRADAY)
                net_pnl = gross_pnl - costs.total_tax_and_charges

                t_id = trade_id_counter
                trade_id_counter += 1

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
                }

                active_symbols.add(sym)
                active_sectors[sector] = active_sectors.get(sector, 0) + 1

                # Schedule EXIT event
                event_queue.append((tr["exit_time"], EventType.EXIT, {"trade_id": t_id}))

        # 3. Metrics Computation
        df_eq = pd.DataFrame(equity_curve)
        if not df_eq.empty:
            df_eq["peak"] = df_eq["equity"].cummax()
            df_eq["drawdown"] = (df_eq["equity"] - df_eq["peak"]) / df_eq["peak"] * 100.0
            max_drawdown_pct = float(abs(df_eq["drawdown"].min()))

            pnl_series = df_eq["pnl"]
            sharpe = float((pnl_series.mean() / (pnl_series.std() + 1e-6)) * np.sqrt(252)) if len(pnl_series) > 5 else 0.0
            neg_pnl = pnl_series[pnl_series < 0]
            sortino = float((pnl_series.mean() / (neg_pnl.std() + 1e-6)) * np.sqrt(252)) if len(neg_pnl) > 3 else sharpe
        else:
            max_drawdown_pct = 0.0
            sharpe = 0.0
            sortino = 0.0

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
            "portfolio_sharpe": round(sharpe, 2),
            "portfolio_sortino": round(sortino, 2),
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
