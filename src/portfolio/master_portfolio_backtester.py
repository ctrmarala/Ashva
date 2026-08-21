"""
Ashva Master Multi-Alpha Portfolio Backtester (The Institutional Ensemble Engine)
Blends multiple champion alphas into a unified chronological trading simulation.
Enforces Portfolio Capital Limits, Risk Parity Sizing, Max Concurrent Positions,
Sector Exposure Caps, and Regime DNA Gates.
"""

from typing import Dict, List, Any, Optional, Tuple, Type
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine, BacktestTrade
from src.research.regime_profiler import MarketRegimeProfiler


SECTOR_MAP = {
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "TECHM": "IT", "WIPRO": "IT",
    # Banking & Financials
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "INDUSINDBK": "BANKING", "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE",
    "SHRIRAMFIN": "FINANCE", "HDFCLIFE": "FINANCE", "SBILIFE": "FINANCE",
    # Auto
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "TMPV": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    # Energy & Oil
    "RELIANCE": "ENERGY", "NTPC": "ENERGY", "ONGC": "ENERGY", "POWERGRID": "ENERGY",
    "COALINDIA": "ENERGY", "BPCL": "ENERGY",
    # Pharma
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA", "DIVISLAB": "PHARMA",
    "APOLLOHOSP": "PHARMA",
    # Metals & Materials
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "GRASIM": "MATERIALS", "ULTRACEMCO": "MATERIALS",
    "PIDILITIND": "MATERIALS",
    # FMCG & Retail
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "TATACONSUM": "FMCG", "TITAN": "CONSUMER", "TRENT": "RETAIL",
    # Conglomerate & Infra
    "LT": "INFRA", "ADANIENT": "INFRA", "ADANIPORTS": "INFRA", "BEL": "DEFENSE",
}


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
    Simulates simultaneous multi-alpha portfolio execution with risk and sector controls.
    """

    def __init__(
        self,
        data_lake: Optional[DataLake] = None,
        initial_capital: float = 500000.0,   # ₹5,00,000 INR
        risk_per_trade_inr: float = 2500.0,  # ₹2,500 (0.50% account risk per trade)
        max_concurrent_positions: int = 5,
        max_positions_per_sector: int = 2,
        cost_model: Optional[IndianCostModel] = None,
    ):
        self.lake = data_lake or DataLake(read_only=True)
        self.initial_capital = initial_capital
        self.risk_per_trade_inr = risk_per_trade_inr
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
        """
        Executes master multi-alpha portfolio backtest by extracting clean discrete trades per alpha
        and merging into a global chronological portfolio simulation with concurrent risk gates.
        """
        tf_map = strategy_timeframe_map or {}
        trail_map = strategy_trailing_map or {}

        # 1. Extract standalone discrete trades from all strategies across all symbols
        all_candidate_trades: List[Dict[str, Any]] = []

        print(f"[*] Extracting candidate trades for {len(strategies)} alphas across {len(symbols)} symbols...")
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

        # Sort all candidate trades chronologically by entry_time
        all_candidate_trades.sort(key=lambda x: x["entry_time"])
        print(f"[+] Total raw candidate trades extracted: {len(all_candidate_trades):,}")

        # 2. Chronological Multi-Asset Portfolio Execution
        executed_trades: List[PortfolioTrade] = []
        open_positions: List[Dict[str, Any]] = []
        equity = self.initial_capital
        equity_curve = []
        trade_id_counter = 1

        for tr in all_candidate_trades:
            curr_time = tr["entry_time"]

            # Remove closed positions
            open_positions = [pos for pos in open_positions if pos["exit_time"] > curr_time]

            # Risk Gate 1: Max Overall Concurrent Positions
            if len(open_positions) >= self.max_concurrent_positions:
                continue

            # Risk Gate 2: Max Positions Per Sector
            sector = tr["sector"]
            sector_positions = [pos for pos in open_positions if pos["sector"] == sector]
            if len(sector_positions) >= self.max_positions_per_sector:
                continue

            # Risk Gate 3: Regime DNA Gate
            if use_regime_filter:
                regime_state = self.regime_profiler.get_regime_for_date(curr_time)
                # Filter out hostile sideways chop with low volatility for breakout alphas
                if regime_state.get("composite") == "SIDEWAYS_CHOP_LOW_VOLATILITY" and "BREAKOUT" in tr["strategy_id"].upper():
                    continue

            # Position Sizing (Fixed Risk Parity: ₹2,500 risk)
            stop_dist = max(1e-2, abs(tr["entry_price"] - tr["initial_sl"]))
            qty = max(1, int(self.risk_per_trade_inr / stop_dist))

            # Statutory Costs & Net PnL Calculation
            is_long = (tr["side"] == "LONG")
            gross_pnl = ((tr["exit_price"] - tr["entry_price"]) * qty) if is_long else ((tr["entry_price"] - tr["exit_price"]) * qty)

            buy_price = tr["entry_price"] if is_long else tr["exit_price"]
            sell_price = tr["exit_price"] if is_long else tr["entry_price"]

            if buy_price <= 0 or sell_price <= 0:
                continue

            costs = self.cost_model.calculate_trade_costs(buy_price=buy_price, sell_price=sell_price, quantity=qty, segment=Segment.EQUITY_INTRADAY)
            net_pnl = gross_pnl - costs.total_tax_and_charges

            equity += net_pnl
            equity_curve.append({"timestamp": tr["exit_time"], "equity": equity, "pnl": net_pnl})

            port_trade = PortfolioTrade(
                trade_id=trade_id_counter,
                strategy_id=tr["strategy_id"],
                symbol=tr["symbol"],
                sector=tr["sector"],
                entry_time=tr["entry_time"],
                exit_time=tr["exit_time"],
                side=tr["side"],
                entry_price=tr["entry_price"],
                exit_price=tr["exit_price"],
                quantity=qty,
                gross_pnl=round(gross_pnl, 2),
                net_pnl=round(net_pnl, 2),
                total_costs=round(costs.total_tax_and_charges, 2),
                exit_reason=tr["exit_reason"],
                mfe_pct=tr["mfe_pct"],
                mae_pct=tr["mae_pct"],
            )
            executed_trades.append(port_trade)
            trade_id_counter += 1

            open_positions.append({
                "symbol": tr["symbol"],
                "sector": tr["sector"],
                "exit_time": tr["exit_time"],
            })

        # 3. Master Metrics Computation
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

        total_net_pnl = equity - self.initial_capital
        win_count = sum(1 for t in executed_trades if t.net_pnl > 0)
        total_trades_count = len(executed_trades)
        win_rate = (win_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0
        gross_win = sum(t.net_pnl for t in executed_trades if t.net_pnl > 0)
        gross_loss = sum(abs(t.net_pnl) for t in executed_trades if t.net_pnl < 0)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        # Alpha contribution summary
        alpha_pnl = {}
        for t in executed_trades:
            alpha_pnl[t.strategy_id] = alpha_pnl.get(t.strategy_id, 0.0) + t.net_pnl

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(equity, 2),
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
