"""
Ashva Autonomous Alpha Discovery & Validation Campaign Loop
Executes closed-loop discovery:
1. Sequential Hypothesis Formulation (informed by accumulated knowledge)
2. Stage-0 Vectorized Empirical Diagnostic on Real DataLake Bars
3. Strategy Implementation & Contract Verification (pytest)
4. Authoritative DEV Backtest (BacktestEngine + IndianCostModel on 540-day lookback)
5. Statistical Validation & Multiple-Testing Adjustment (DSR)
6. Immutable SQLite Ledger Journaling
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Add root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine, BacktestResult
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha
from src.research.knowledge_map import AlphaKnowledgeMap, AlphaCategory, MechanismStatus
from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    HypothesisStatus,
    StrategyHorizon,
    MarketMechanism,
)

lake = DataLake(read_only=True)
cost_model = IndianCostModel()
engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
ledger = ResearchExperimentLedger()
git_sha = get_current_git_sha()

symbols = [
    "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
    "BAJFINANCE", "MARUTI", "SUNPHARMA"
]

print("=" * 110)
print("[*] ASHVA CLOSED-LOOP AUTONOMOUS ALPHA DISCOVERY CAMPAIGN")
print(f"[*] Platform: Frozen Factory v1 | Scope: Strictly Intraday Cash Equities | Cumulative Trials: {ledger.get_total_trials()}")
print("=" * 110)

def run_dev_backtest(strategy: BaseHypothesis, strat_id: str) -> Tuple[Dict[str, Any], List[Any]]:
    all_trades = []
    total_gross = 0.0
    total_net = 0.0
    total_costs = 0.0
    participating = 0
    active_days = set()
    per_symbol = []
    
    for sym in symbols:
        df = lake.load_bars(sym, "15m", max_lookback_days=540)
        if df.empty:
            continue
        sig_df = strategy.generate_signals(df)
        res = engine.run(sig_df, symbol=sym, strategy_id=strat_id, capital_per_trade_pct=0.25, risk_per_trade_pct=0.005)
        
        if res.total_trades > 0:
            participating += 1
            for t in res.trade_list:
                all_trades.append(t)
                active_days.add(pd.to_datetime(t.entry_time).date())
                total_gross += t.gross_pnl
                total_net += t.net_pnl
                total_costs += (t.gross_pnl - t.net_pnl)
                
            per_symbol.append({
                "symbol": sym,
                "trades": res.total_trades,
                "net_pnl": res.total_net_pnl,
                "win_rate": res.win_rate_pct,
                "pf": res.net_profit_factor,
            })
            
    n_trades = len(all_trades)
    if n_trades == 0:
        return {"total_trades": 0, "net_pnl": 0.0, "net_pf": 0.0, "win_rate": 0.0, "is_sharpe": 0.0, "max_dd_pct": 0.0}, []
        
    wins = [t for t in all_trades if t.net_pnl > 0]
    losses = [t for t in all_trades if t.net_pnl <= 0]
    win_rate = (len(wins) / n_trades) * 100.0
    
    net_wins = sum(t.net_pnl for t in wins)
    net_losses = abs(sum(t.net_pnl for t in losses))
    net_pf = (net_wins / net_losses) if net_losses > 0 else 0.0
    
    # Calculate Sharpe from trade returns
    returns = [t.net_pnl / 125000.0 for t in all_trades] # ~25% capital per trade
    std = np.std(returns)
    sharpe = (np.mean(returns) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0 # ~annualized
    
    all_trades_sorted = sorted(all_trades, key=lambda t: t.exit_time)
    cum_equity = np.cumsum([t.net_pnl for t in all_trades_sorted])
    running_max = np.maximum.accumulate(cum_equity)
    drawdowns = cum_equity - running_max
    max_dd_inr = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_dd_pct = (max_dd_inr / 500000.0) * 100.0
    
    summary = {
        "total_trades": n_trades,
        "active_days": len(active_days),
        "participating_stocks": participating,
        "gross_pnl": total_gross,
        "total_costs": total_costs,
        "net_pnl": total_net,
        "net_pf": net_pf,
        "win_rate": win_rate,
        "is_sharpe": sharpe,
        "max_dd_inr": max_dd_inr,
        "max_dd_pct": max_dd_pct,
        "avg_trade_pnl": total_net / n_trades,
    }
    return summary, per_symbol

print("[+] Backtest engine & helper initialized.")
