import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from src.ui.data_access import UIDataAccess
import pandas as pd
import numpy as np

dal = UIDataAccess()
df = dal.get_alpha_registry_table()
proven = df[df["status"] == "PROVEN"].copy()

print("=" * 95)
print(f"ASHVA FULL 18-MONTH LIFETIME PORTFOLIO SUMMARY ({len(proven)} PROVEN STRATEGIES)")
print("=" * 95)

total_net_pnl = 0.0
total_trades = 0

for idx, row in proven.iterrows():
    pnl_str = str(row["net_pnl"]).replace("Rs", "").replace(",", "").strip()
    try:
        pnl_val = float(pnl_str)
    except Exception:
        pnl_val = 0.0
    
    trades_val = int(str(row["trades"]).replace(",", "")) if str(row["trades"]).replace(",", "").isdigit() else 0
    total_net_pnl += pnl_val
    total_trades += trades_val
    print(f"{row['alpha_id']:<12} | TF: {row['timeframe']:<4} | Trades: {trades_val:5d} | Win Rate: {row['win_rate']:>6s} | Lifetime Net PnL: Rs {pnl_val:>+12,.2f} | Net PF: {row['profit_factor']}")

print("-" * 95)
capital = 500000.0
total_roi = (total_net_pnl / capital) * 100.0
# 18 months = 1.5 years
monthly_avg_roi = total_roi / 18.0
monthly_net_cash = total_net_pnl / 18.0

print(f"Total Portfolio Trades Executed:   {total_trades:,}")
print(f"Total Cumulative Net Cash Flow:    Rs {total_net_pnl:+,.2f}")
print(f"Portfolio Base Capital:            Rs {capital:,.2f}")
print(f"Total 18-Month Cumulative ROI:     {total_roi:+.2f}%")
print(f"Average Monthly Net Cash Flow:     Rs {monthly_net_cash:+,.2f} / month")
print(f"Average Monthly Net ROI:           {monthly_avg_roi:+.2f}% / month")
print("=" * 95)
