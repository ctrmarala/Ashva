"""
Ashva Cross-Timeframe Alpha Sweet-Spot Scanner
Systematically evaluates alpha strategies across multiple intraday timeframes (5m, 10m, 15m, 30m)
and all 50 NIFTY equities to identify the exact timeframe & asset cluster sweet spot.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine


class CrossTimeframeScanner:
    """
    Evaluates alpha performance across 5m, 10m, 15m, and 30m bar resolutions.
    """

    TIMEFRAMES = ["5m", "10m", "15m", "30m"]

    def __init__(self, data_lake: Optional[DataLake] = None):
        self.lake = data_lake or DataLake(read_only=True)
        self.cost_model = IndianCostModel()

    def scan_strategy_across_timeframes(
        self,
        strategy_class: Any,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        trailing_mode: str = "BREAK_EVEN",
        capital_per_trade: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Runs strategy across all given timeframes and symbols.
        """
        tfs = timeframes or self.TIMEFRAMES
        results_by_tf = {}

        strat_instance = strategy_class()
        alpha_name = getattr(strat_instance, "name", strategy_class.__name__)

        print(f"[*] Scanning {alpha_name} across timeframes: {tfs} for {len(symbols)} symbols...")

        for tf in tfs:
            total_trades = 0
            gross_pnl = 0.0
            net_pnl = 0.0
            wins = 0
            gross_win = 0.0
            gross_loss = 0.0
            all_trade_returns = []
            symbol_pnl_map = {}

            for sym in symbols:
                df = self.lake.load_bars(sym.upper(), tf)
                if df.empty or len(df) < 500:
                    continue

                if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp").sort_index()

                strat = strategy_class()
                df_signals = strat.generate_signals(df)
                engine = BacktestEngine(cost_model=self.cost_model, initial_capital=capital_per_trade)
                res = engine.run(df_signals, symbol=sym.upper(), strategy_id=alpha_name, trailing_mode=trailing_mode)

                trade_count = res.total_trades
                sym_net = res.total_net_pnl
                total_trades += trade_count
                gross_pnl += (res.final_equity - res.initial_capital)
                net_pnl += sym_net
                wins += res.winning_trades
                gross_win += sum([t.net_pnl for t in res.trade_list if t.net_pnl > 0])
                gross_loss += sum([abs(t.net_pnl) for t in res.trade_list if t.net_pnl < 0])
                all_trade_returns.extend([t.net_pnl for t in res.trade_list])
                symbol_pnl_map[sym] = round(sym_net, 2)

            win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
            pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
            ret_series = pd.Series(all_trade_returns)
            sharpe = float((ret_series.mean() / (ret_series.std() + 1e-6)) * np.sqrt(252)) if len(ret_series) > 5 else 0.0

            results_by_tf[tf] = {
                "timeframe": tf,
                "total_trades": total_trades,
                "net_pnl": round(net_pnl, 2),
                "win_rate": f"{round(win_rate, 1)}%",
                "profit_factor": round(pf, 2),
                "sharpe": round(sharpe, 2),
                "top_performing_assets": sorted(symbol_pnl_map.items(), key=lambda x: x[1], reverse=True)[:5],
            }

        # Find Sweet Spot Timeframe (Max Sharpe with Net PnL > 0)
        valid_tfs = [tf for tf, data in results_by_tf.items() if data["net_pnl"] > 0]
        sweet_spot = max(valid_tfs, key=lambda tf: results_by_tf[tf]["sharpe"]) if valid_tfs else "NONE"

        return {
            "alpha_name": alpha_name,
            "trailing_mode": trailing_mode,
            "sweet_spot_timeframe": sweet_spot,
            "timeframe_matrix": results_by_tf,
        }
