import json
import numpy as np
import pandas as pd
from typing import List

from src.strategies.alpha_001_opening_gap_continuation import Alpha1OpeningGapContinuation
from src.core.universe_manager import get_universe_symbols
from src.data.data_lake import DataLake
from src.backtest.engine import BacktestEngine
from src.analytics.indian_costs import IndianCostModel, Segment
from src.research.cpcv_engine import CPCVEngine
from src.research.validator import StatisticalValidator
from src.analytics.metrics import calculate_profit_factor

def main():
    print("Starting independent audit for Alpha 1...")
    strat = Alpha1OpeningGapContinuation()
    symbols = get_universe_symbols()
    lake = DataLake()
    cost_model = IndianCostModel()
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
    
    all_trades = []
    
    print(f"Testing {len(symbols)} symbols...")
    for sym in symbols:
        df = lake.load_bars(sym, "15m", max_lookback_days=540)
        if df.empty or len(df) < 50:
            continue
            
        sig_df = strat.generate_signals(df)
        res = engine.run(sig_df, symbol=sym, strategy_id="1_alpha", capital_per_trade_pct=0.25)
        all_trades.extend(res.trade_list)
        
    print(f"Total trades across panel: {len(all_trades)}")
    
    # Analyze raw panel metrics
    net_pnls = [t.net_pnl for t in all_trades]
    gross_pnls = [t.gross_pnl for t in all_trades]
    profit_factor = calculate_profit_factor(net_pnls)
    
    wins = sum(1 for p in net_pnls if p > 0)
    win_rate = (wins / len(net_pnls) * 100.0) if all_trades else 0.0
    
    # CPCV Evaluation
    cpcv = CPCVEngine()
    df_trades = pd.DataFrame([{
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "net_pnl": t.net_pnl
    } for t in all_trades])
    
    cpcv_res = cpcv.evaluate_trades(df_trades, initial_capital=500000.0)
    
    # Monte Carlo Tail Risk
    df_tr = df_trades.copy()
    if not df_tr.empty:
        df_tr["date"] = pd.to_datetime(df_tr["entry_time"]).dt.date
        daily_pnl = df_tr.groupby("date")["net_pnl"].sum()
        daily_returns = (daily_pnl / 500000.0).values
        mc_res = StatisticalValidator.run_monte_carlo_drawdown_test(daily_returns, num_simulations=5000)
    else:
        mc_res = {"p95_max_dd": 0.0}
    
    report = {
        "strategy": "1_alpha (Opening Gap Continuation)",
        "universe_size": len(symbols),
        "total_trades": len(all_trades),
        "win_rate_pct": round(win_rate, 2),
        "net_profit_factor": round(profit_factor, 2),
        "cpcv_oos_sharpe": cpcv_res.get("mean_oos_sharpe", 0.0),
        "cpcv_pbo_pct": cpcv_res.get("pbo_pct", "0.0%"),
        "cpcv_degradation": cpcv_res.get("degradation_ratio", 0.0),
        "monte_carlo_p95_dd": round(mc_res.get("p95_max_dd", 0.0), 2)
    }
    
    with open("temp_audit_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Audit complete. Report generated at temp_audit_report.json.")

if __name__ == "__main__":
    main()
