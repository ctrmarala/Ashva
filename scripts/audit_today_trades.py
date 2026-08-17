"""
Ashva Today's Live Forward Paper Trading Audit
Summarizes all trades executed today, fees deducted, open positions, and net realized P&L.

Usage:
    python scripts/audit_today_trades.py
"""

import sqlite3
import json
from datetime import datetime

def main():
    conn = sqlite3.connect("data_lake/ashva_state_wal.db")
    c = conn.cursor()

    today_str = datetime.now().strftime("%Y-%m-%d")

    print("=" * 90)
    print(f"[*] ASHVA LIVE FORWARD PAPER TRADING AUDIT: TODAY ({today_str})")
    print("=" * 90)

    # 1. Closed Trades Today
    closed = c.execute("SELECT trade_id, symbol, entry_time, exit_time, side, quantity, entry_price, exit_price, gross_pnl, net_pnl, cost_breakdown_json FROM trade_ledger WHERE entry_time LIKE ? ORDER BY trade_id ASC", (f"{today_str}%",)).fetchall()

    total_closed_net_pnl = 0.0
    total_closed_gross_pnl = 0.0
    total_taxes = 0.0

    print(f"\n[+] CLOSED TRADES EXECUTED TODAY: {len(closed)}")
    print(f"{'#':2s} | {'Symbol':10s} | {'Side':5s} | {'Qty':4s} | {'Entry Price':>11s} | {'Exit Price':>11s} | {'Gross PnL':>10s} | {'Taxes/Fees':>10s} | {'Net PnL':>10s}")
    print("-" * 90)

    for r in closed:
        tid, sym, entry_t, exit_t, side, qty, ep, xp, gpnl, npnl, costs_json = r
        costs = json.loads(costs_json) if costs_json else {}
        tax = costs.get("total_tax_and_charges", gpnl - npnl)
        total_closed_gross_pnl += gpnl
        total_closed_net_pnl += npnl
        total_taxes += tax
        print(f"#{tid:<2d}| {sym:10s} | {side:5s} | {qty:4d} | Rs {ep:>8.2f} | Rs {xp:>8.2f} | Rs {gpnl:>+7.2f} | Rs {tax:>7.2f} | Rs {npnl:>+7.2f}")

    print("-" * 90)

    # 2. Open Positions
    open_pos = c.execute("SELECT symbol, side, quantity, entry_price, entry_time, strategy_id FROM open_positions").fetchall()
    print(f"\n[+] CURRENTLY OPEN POSITIONS: {len(open_pos)}")
    for op in open_pos:
        sym, side, qty, ep, etime, strat = op
        print(f"    - {sym:10s} | {side:5s} | {qty} Shares @ Rs {ep:.2f} (Entry: {etime[:16]}) | Strategy: {strat}")

    if not open_pos:
        print("    (No open positions. Portfolio is 100% in Cash)")

    print("\n" + "=" * 90)
    print(f"[*] TODAY'S PERFORMANCE SUMMARY:")
    print(f"    - Total Executed Trades        : {len(closed) + len(open_pos)}")
    print(f"    - Realized Gross P&L           : Rs {total_closed_gross_pnl:+,.2f}")
    print(f"    - Total STT, GST & Brokerage   : Rs {total_taxes:,.2f}")
    print(f"    - REALIZED NET PROFIT (POST-TAX): Rs {total_closed_net_pnl:+,.2f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
