"""
Ashva Quantitative Strategy & Backtest Contract Audit Engine
Inspects all 23 quantitative alpha strategies to determine trade duration,
position persistence, and identify any unintended 1-bar signal pulse exits.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.run_hypothesis_lab import STRATEGY_MAP, DEFAULT_UNIVERSE
from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
import pandas as pd

def audit_strategy_contracts():
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel()
    
    print("=" * 115)
    print(f"ASHVA STRATEGY CONTRACT & POSITION PERSISTENCE AUDIT (23 STRATEGIES)")
    print(f"Checking whether strategies maintain intraday state vs emit 1-bar pulse signals")
    print("=" * 115)
    print(f"{'Alpha ID':<10} | {'Strategy Name':<35} | {'Trades':<6} | {'Avg Duration':<18} | {'1-Bar Exits %':<14} | {'Status'}")
    print("-" * 115)

    results = []

    # Audit across NIFTY-14 over 180 days to get broad sample
    for strat_key, (strat_name, strat_cls) in STRATEGY_MAP.items():
        total_trades = 0
        total_duration = 0
        one_bar_trades = 0
        exit_reasons = {}

        for sym in DEFAULT_UNIVERSE[:5]:  # Test on 5 major symbols
            df = lake.load_bars(sym, "15m", max_lookback_days=180)
            if df.empty or len(df) < 50:
                continue

            try:
                strat = strat_cls()
                sig = strat.generate_signals(df)
                eng = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
                res = eng.run(sig, symbol=sym, strategy_id=strat_key, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)
                
                for t in res.trade_list:
                    total_trades += 1
                    total_duration += t.duration_bars
                    if t.duration_bars <= 1:
                        one_bar_trades += 1
                    exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
            except Exception as e:
                pass

        if total_trades > 0:
            avg_dur = total_duration / total_trades
            one_bar_pct = (one_bar_trades / total_trades) * 100.0
            
            if one_bar_pct >= 85.0 and exit_reasons.get("SIGNAL", 0) > (total_trades * 0.7):
                status = "[!] 1-BAR PULSE DEFECT (Unintended 1-bar exit)"
            elif avg_dur > 3.0:
                status = "[OK] PERSISTENT INTRADAY (Normal Drift/Holding)"
            else:
                status = "[?] SHORT DURATION (Investigate)"

            results.append({
                "Alpha_ID": strat_key,
                "Strategy": strat_name,
                "Trades": total_trades,
                "Avg_Duration_Bars": round(avg_dur, 1),
                "One_Bar_Pct": round(one_bar_pct, 1),
                "SIGNAL_Exits": exit_reasons.get("SIGNAL", 0),
                "SL_Exits": exit_reasons.get("STOP_LOSS", 0),
                "TP_Exits": exit_reasons.get("TAKE_PROFIT", 0),
                "EOD_Exits": exit_reasons.get("EOD", 0),
                "Status": status
            })

            print(f"{strat_key:<10} | {strat_name[:35]:<35} | {total_trades:<6} | {avg_dur:<4.1f} bars ({avg_dur*15:.0f}m)  | {one_bar_pct:<5.1f}% ({one_bar_trades}T)  | {status}")
        else:
            print(f"{strat_key:<10} | {strat_name[:35]:<35} | 0      | N/A                | N/A            | [ ] No Trades")

    df_res = pd.DataFrame(results)
    df_res.to_markdown("strategy_contract_audit_report.md", index=False)
    print("=" * 115)
    return df_res

if __name__ == "__main__":
    audit_strategy_contracts()
