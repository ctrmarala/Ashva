"""
Clean Mock Unit Test Records from Live WAL Database
"""

import sqlite3

def clean_db():
    conn = sqlite3.connect("data_lake/ashva_state_wal.db")
    c = conn.cursor()

    # Remove test trades (dummy 2500 -> 2550 trades from pytest)
    c.execute("DELETE FROM trade_ledger WHERE entry_price = 2500.0 AND exit_price = 2550.0")
    conn.commit()

    rows = c.execute("SELECT trade_id, symbol, entry_time, exit_time, side, quantity, entry_price, exit_price, net_pnl FROM trade_ledger ORDER BY trade_id ASC").fetchall()
    print("=" * 80)
    print(f"DATABASE CLEANED. TOTAL REAL MARKET TRADES TODAY: {len(rows)}")
    print("=" * 80)
    total_real_pnl = sum(r[8] for r in rows)
    for r in rows:
        print(f"#{r[0]:<2d} | {r[1]:10s} | {r[4]:5s} | Qty: {r[5]:3d} | Entry: Rs {r[6]:.2f} ({r[2][11:19]}) | Exit: Rs {r[7]:.2f} | Net: Rs {r[8]:+.2f}")
    print("-" * 80)
    print(f"REAL NET DAILY P&L: Rs {total_real_pnl:+,.2f}")
    print("=" * 80)
    conn.close()

if __name__ == "__main__":
    clean_db()
