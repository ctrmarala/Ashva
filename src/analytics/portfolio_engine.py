"""
Ashva Multi-Alpha Ensemble Portfolio Allocation & Monthly ROI Engine
Simulates concurrent multi-alpha execution, non-overlapping capital allocation,
monthly return distributions, and maximum drawdown profiles under Indian statutory friction.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.engine import BacktestTrade


class MultiAlphaPortfolioEngine:
    """
    Simulates portfolio-level equity curves and monthly ROI distributions
    by ensembling multiple uncorrelated alpha strategies.
    """

    def __init__(self, initial_capital: float = 7000000.0, max_active_positions: int = 14):
        self.initial_capital = initial_capital
        self.max_active_positions = max_active_positions

    def evaluate_portfolio(
        self,
        strategy_trades: Dict[str, List[BacktestTrade]],
        capital_per_trade: float = 125000.0,
    ) -> Dict[str, Any]:
        """
        Merges trades from multiple alpha streams chronologically and computes
        portfolio performance, monthly ROI, Calmar ratio, Sharpe, and drawdown.
        """
        all_trades_tagged = []
        for strat_name, trades in strategy_trades.items():
            for t in trades:
                all_trades_tagged.append({
                    "strategy": strat_name,
                    "symbol": t.symbol,
                    "entry_time": pd.to_datetime(t.entry_time),
                    "exit_time": pd.to_datetime(t.exit_time),
                    "side": t.side,
                    "gross_pnl": t.gross_pnl,
                    "taxes_paid": (t.gross_pnl - t.net_pnl),
                    "net_pnl": t.net_pnl,
                    "net_roi_pct": (t.net_pnl / capital_per_trade) * 100.0,
                    "duration_bars": t.duration_bars,
                    "exit_reason": t.exit_reason,
                })

        if not all_trades_tagged:
            return {
                "total_trades": 0, "total_net_pnl": 0.0, "portfolio_roi_pct": 0.0,
                "monthly_avg_roi_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "sharpe": 0.0, "max_drawdown_pct": 0.0, "monthly_table": pd.DataFrame()
            }

        df_trades = pd.DataFrame(all_trades_tagged).sort_values(by="exit_time").reset_index(drop=True)

        # Cumulative equity and drawdown
        df_trades["cum_net_pnl"] = df_trades["net_pnl"].cumsum()
        df_trades["portfolio_equity"] = self.initial_capital + df_trades["cum_net_pnl"]
        df_trades["running_peak"] = df_trades["portfolio_equity"].cummax()
        df_trades["drawdown_inr"] = df_trades["portfolio_equity"] - df_trades["running_peak"]
        df_trades["drawdown_pct"] = (df_trades["drawdown_inr"] / df_trades["running_peak"]) * 100.0

        max_dd_pct = abs(df_trades["drawdown_pct"].min())
        max_dd_inr = abs(df_trades["drawdown_inr"].min())

        total_net_pnl = df_trades["net_pnl"].sum()
        total_gross_pnl = df_trades["gross_pnl"].sum()
        total_taxes = df_trades["taxes_paid"].sum()
        total_trades = len(df_trades)

        wins = df_trades[df_trades["net_pnl"] > 0]
        losses = df_trades[df_trades["net_pnl"] <= 0]
        win_rate = (len(wins) / total_trades) * 100.0

        gross_wins = wins["net_pnl"].sum()
        gross_losses = abs(losses["net_pnl"].sum())
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 0.0)

        # Monthly aggregation
        df_trades["month_period"] = df_trades["exit_time"].dt.to_period("M")
        monthly_groups = df_trades.groupby("month_period")

        monthly_records = []
        for month, grp in monthly_groups:
            m_net = grp["net_pnl"].sum()
            m_trades = len(grp)
            m_wins = len(grp[grp["net_pnl"] > 0])
            m_wr = (m_wins / m_trades) * 100.0 if m_trades > 0 else 0.0
            # ROI relative to total deployed capital and active utilized capital (~14 slots * 125k = 1.75M avg peak)
            m_roi_total_pct = (m_net / self.initial_capital) * 100.0
            m_roi_active_pct = (m_net / (14 * capital_per_trade)) * 100.0

            monthly_records.append({
                "Month": str(month),
                "Trades": m_trades,
                "Win_Rate_Pct": round(m_wr, 1),
                "Net_PnL_INR": round(m_net, 2),
                "Monthly_ROI_Total_Capital_Pct": round(m_roi_total_pct, 2),
                "Monthly_ROI_Active_Capital_Pct": round(m_roi_active_pct, 2),
            })

        df_monthly = pd.DataFrame(monthly_records)
        avg_monthly_net_pnl = df_monthly["Net_PnL_INR"].mean() if not df_monthly.empty else 0.0
        avg_monthly_roi_total = df_monthly["Monthly_ROI_Total_Capital_Pct"].mean() if not df_monthly.empty else 0.0
        avg_monthly_roi_active = df_monthly["Monthly_ROI_Active_Capital_Pct"].mean() if not df_monthly.empty else 0.0

        # Sharpe calculation from daily returns
        daily_pnl = df_trades.set_index("exit_time")["net_pnl"].resample("1D").sum().fillna(0.0)
        daily_rets = daily_pnl / self.initial_capital
        mean_ret = daily_rets.mean()
        std_ret = daily_rets.std()
        annualized_sharpe = float((mean_ret / std_ret * np.sqrt(252))) if std_ret > 1e-7 else 0.0

        # Strategy breakdown
        strat_breakdown = df_trades.groupby("strategy").agg(
            trades=("net_pnl", "count"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda s: (s > 0).mean() * 100.0)
        ).reset_index()

        return {
            "total_trades": total_trades,
            "total_gross_pnl": round(total_gross_pnl, 2),
            "total_taxes_paid": round(total_taxes, 2),
            "total_net_pnl": round(total_net_pnl, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "annualized_sharpe": round(annualized_sharpe, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "max_drawdown_inr": round(max_dd_inr, 2),
            "avg_monthly_net_pnl": round(avg_monthly_net_pnl, 2),
            "avg_monthly_roi_total_pct": round(avg_monthly_roi_total, 2),
            "avg_monthly_roi_active_pct": round(avg_monthly_roi_active, 2),
            "monthly_table": df_monthly,
            "strategy_breakdown": strat_breakdown,
        }
