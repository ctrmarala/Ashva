"""
Ashva Corporate Actions Sync & Anomaly Inspector CLI
Nightly maintenance tool to detect unadjusted price cliffs and apply
backward adjustments for splits, bonuses, and special dividends.

Usage:
  # Scan all active universe stocks for unadjusted corporate action anomalies
  python scripts/sync_corporate_actions.py --scan-universe

  # Apply a 1:2 Split on a specific stock
  python scripts/sync_corporate_actions.py --apply --symbol ITC --type SPLIT --ex-date 2026-06-15 --old 1 --new 2

  # View Audit Ledger of applied corporate actions
  python scripts/sync_corporate_actions.py --history
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.corporate_actions import CorporateActionManager, CorporateAction, CorporateActionType


UNIVERSE_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "ITC",
    "LT", "HINDUNILVR", "BAJFINANCE", "TATAMOTORS", "MARUTI", "M&M", "SUNPHARMA",
    "AXISBANK", "NTPC", "ONGC", "POWERGRID", "TITAN", "TRENT", "BEL", "BAJAJFINSV",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "TATASTEEL", "JSWSTEEL", "GRASIM",
    "ULTRACEMCO", "TECHM", "HCLTECH", "WIPRO", "NESTLEIND", "BRITANNIA", "CIPLA",
    "DRREDDY", "DIVISLAB", "APOLLOHOSP", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO",
    "INDUSINDBK", "KOTAKBANK", "SHRIRAMFIN", "HDFCLIFE", "SBILIFE", "BPCL", "PIDILITIND",
]


def main():
    parser = argparse.ArgumentParser(description="Ashva Corporate Action Engine & Nightly Inspector")
    parser.add_argument("--scan-universe", action="store_true", help="Scans all 50 universe stocks for unadjusted gap cliffs")
    parser.add_argument("--scan-symbol", type=str, help="Scans a single symbol for unadjusted gap cliffs")
    parser.add_argument("--apply", action="store_true", help="Apply a corporate action")
    parser.add_argument("--symbol", type=str, help="Asset symbol (e.g. ITC, TATAMOTORS)")
    parser.add_argument("--type", type=str, choices=["SPLIT", "BONUS", "SPECIAL_DIVIDEND", "RIGHTS"], help="Type of corporate action")
    parser.add_argument("--ex-date", type=str, help="Ex-Date in YYYY-MM-DD format")
    parser.add_argument("--old", type=float, default=1.0, help="Pre-event ratio units (e.g. 1)")
    parser.add_argument("--new", type=float, default=1.0, help="Post-event ratio units (e.g. 2 for 1:2 split)")
    parser.add_argument("--cash", type=float, default=0.0, help="Cash dividend amount per share")
    parser.add_argument("--pre-price", type=float, default=None, help="Pre-event closing price")
    parser.add_argument("--notes", type=str, default="", help="Optional context or exchange circular reference")
    parser.add_argument("--history", action="store_true", help="Print the corporate actions audit ledger")

    args = parser.parse_args()
    manager = CorporateActionManager()

    if args.history:
        print("\n==========================================================================================")
        print("                  ASHVA CORPORATE ACTIONS AUDIT LEDGER")
        print("==========================================================================================")
        if not manager.ledger:
            print("No corporate actions recorded in the audit ledger.")
        else:
            df_ledger = pd.DataFrame(manager.ledger)
            print(df_ledger.to_string(index=False))
        print("==========================================================================================\n")
        return

    if args.scan_universe or args.scan_symbol:
        symbols_to_scan = [args.scan_symbol.upper()] if args.scan_symbol else UNIVERSE_50
        print("\n==========================================================================================")
        print(f"       SCANNING {len(symbols_to_scan)} ASSETS FOR UNADJUSTED OVERNIGHT CLIFFS (>20% DROP)")
        print("==========================================================================================")
        
        all_anomalies = []
        for sym in symbols_to_scan:
            anoms = manager.detect_unadjusted_anomalies(sym, timeframe="1d", threshold_pct=0.20)
            all_anomalies.extend(anoms)

        if not all_anomalies:
            print("[SUCCESS] Zero unadjusted corporate action anomalies detected across scanned universe.")
            print("All historical price series are continuous and split-adjusted.")
        else:
            print(f"[WARNING] Detected {len(all_anomalies)} potential unadjusted corporate actions:\n")
            df_anoms = pd.DataFrame(all_anomalies)
            print(df_anoms.to_string(index=False))
            print("\nUse --apply --symbol <SYM> --type SPLIT --ex-date <DATE> --old <O> --new <N> to adjust.")
        print("==========================================================================================\n")
        return

    if args.apply:
        if not args.symbol or not args.type or not args.ex_date:
            print("[ERROR] --symbol, --type, and --ex-date are required to apply a corporate action.")
            sys.exit(1)

        action = CorporateAction(
            symbol=args.symbol.upper(),
            action_type=CorporateActionType(args.type),
            ex_date=args.ex_date,
            ratio_old=args.old,
            ratio_new=args.new,
            cash_amount=args.cash,
            pre_event_price=args.pre_price,
            notes=args.notes,
        )

        res = manager.register_and_apply(action)
        print("\n==========================================================================================")
        print("                  CORPORATE ACTION ADJUSTMENT RESULT")
        print("==========================================================================================")
        for k, v in res.items():
            print(f"  {k:<22}: {v}")
        print("==========================================================================================\n")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
