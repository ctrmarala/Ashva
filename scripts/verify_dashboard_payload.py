"""
Verify Live Dashboard Telemetry Payload
"""

import urllib.request
import json

def main():
    try:
        req = urllib.request.urlopen("http://localhost:8080/api/telemetry")
        data = json.loads(req.read().decode("utf-8"))
        p = data["portfolio"]

        print("=" * 80)
        print("DASHBOARD TELEMETRY VERIFICATION:")
        print("=" * 80)
        print(f"System Time   : {data['system_time']}")
        print(f"Total Equity  : Rs {p['total_equity']:,.2f}")
        print(f"Free Cash     : Rs {p['free_cash']:,.2f}")
        print(f"Blocked Margin: Rs {p['blocked_margin']:,.2f}")
        print(f"Daily Net PnL : Rs {p['daily_pnl']:+,.2f} ({p['daily_pnl_pct']:+.2f}%)")
        print(f"Gross PnL     : Rs {p['today_gross_pnl']:+,.2f}")
        print(f"Taxes & Fees  : Rs {p['today_taxes_and_brokerage']:,.2f}")
        print(f"Closed Trades : {len(data['closed_trades'])} Orders")
        print("=" * 80)
    except Exception as e:
        print(f"Error fetching dashboard telemetry: {e}")

if __name__ == "__main__":
    main()
