"""
Ashva Execution Gap & Slippage Attribution Analyzer
Compares theoretical Backtest signals vs actual Live/Paper Broker order fills.
Measures execution slippage, latency degradation, fill rates, and net alpha decay.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ExecutionGapReport:
    symbol: str
    total_trades_compared: int
    mean_entry_slippage_bps: float
    mean_exit_slippage_bps: float
    total_expected_pnl_inr: float
    total_actual_pnl_inr: float
    pnl_execution_gap_inr: float
    alpha_retention_pct: float
    fill_rate_pct: float
    execution_quality_rating: str  # "EXCELLENT", "ACCEPTABLE", "DEGRADED", "UNACCEPTABLE"

    def summary(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trades_compared": self.total_trades_compared,
            "mean_entry_slippage_bps": round(self.mean_entry_slippage_bps, 2),
            "mean_exit_slippage_bps": round(self.mean_exit_slippage_bps, 2),
            "expected_pnl": round(self.total_expected_pnl_inr, 2),
            "actual_pnl": round(self.total_actual_pnl_inr, 2),
            "pnl_gap": round(self.pnl_execution_gap_inr, 2),
            "alpha_retention_pct": round(self.alpha_retention_pct, 2),
            "fill_rate_pct": round(self.fill_rate_pct, 2),
            "rating": self.execution_quality_rating,
        }


class ExecutionGapAnalyzer:
    """
    Audits the fidelity between historical research backtests and forward paper/live trading.
    """

    @staticmethod
    def evaluate_gap(
        backtest_trades: List[Any],
        actual_trades: List[Dict[str, Any]],
        symbol: str = "ASSET",
    ) -> ExecutionGapReport:
        """
        Calculates trade-by-trade execution differences between backtest expectations and broker fills.
        """
        n_bt = len(backtest_trades)
        n_act = len(actual_trades)

        if n_bt == 0 or n_act == 0:
            return ExecutionGapReport(
                symbol=symbol,
                total_trades_compared=0,
                mean_entry_slippage_bps=0.0,
                mean_exit_slippage_bps=0.0,
                total_expected_pnl_inr=0.0,
                total_actual_pnl_inr=0.0,
                pnl_execution_gap_inr=0.0,
                alpha_retention_pct=100.0,
                fill_rate_pct=0.0,
                execution_quality_rating="INSUFFICIENT_DATA",
            )

        n_compare = min(n_bt, n_act)
        entry_slips = []
        exit_slips = []
        expected_pnls = []
        actual_pnls = []

        for i in range(n_compare):
            bt = backtest_trades[i]
            act = actual_trades[i]

            bt_entry = getattr(bt, "entry_price", 0.0)
            act_entry = float(act.get("entry_price", bt_entry))
            if bt_entry > 0:
                slip_entry_bps = abs(act_entry - bt_entry) / bt_entry * 10000.0
                entry_slips.append(slip_entry_bps)

            bt_exit = getattr(bt, "exit_price", 0.0)
            act_exit = float(act.get("exit_price", bt_exit))
            if bt_exit > 0:
                slip_exit_bps = abs(act_exit - bt_exit) / bt_exit * 10000.0
                exit_slips.append(slip_exit_bps)

            expected_pnls.append(getattr(bt, "net_pnl", 0.0))
            actual_pnls.append(float(act.get("net_pnl", 0.0)))

        tot_exp = sum(expected_pnls)
        tot_act = sum(actual_pnls)
        gap = tot_act - tot_exp
        alpha_retention = (tot_act / tot_exp * 100.0) if tot_exp > 0 else 100.0

        mean_en = float(np.mean(entry_slips)) if entry_slips else 0.0
        mean_ex = float(np.mean(exit_slips)) if exit_slips else 0.0
        fill_rate = (n_act / n_bt * 100.0) if n_bt > 0 else 0.0

        if alpha_retention >= 85.0 and mean_en <= 5.0:
            rating = "EXCELLENT"
        elif alpha_retention >= 70.0 and mean_en <= 10.0:
            rating = "ACCEPTABLE"
        elif alpha_retention >= 50.0:
            rating = "DEGRADED"
        else:
            rating = "UNACCEPTABLE"

        return ExecutionGapReport(
            symbol=symbol,
            total_trades_compared=n_compare,
            mean_entry_slippage_bps=mean_en,
            mean_exit_slippage_bps=mean_ex,
            total_expected_pnl_inr=tot_exp,
            total_actual_pnl_inr=tot_act,
            pnl_execution_gap_inr=gap,
            alpha_retention_pct=alpha_retention,
            fill_rate_pct=fill_rate,
            execution_quality_rating=rating,
        )
