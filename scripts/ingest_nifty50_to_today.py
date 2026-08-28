"""
Ashva NIFTY 50 Incremental Ingestion Engine
Fetches up-to-date intraday bars (15m) for all 50 NIFTY equities up to today (2026-08-28 15:30 IST)
from Angel One SmartAPI and updates the Ashva DataLake (DuckDB + Parquet).
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yaml

sys.path.append(str(Path.cwd()))

from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher

NIFTY_50_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "ITC", "HINDUNILVR",
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "ADANIENT", "KOTAKBANK", "TATAMOTORS", "TATASTEEL",
    "NTPC", "AXISBANK", "ONGC", "TITAN", "ADANIPORTS", "COALINDIA", "POWERGRID", "M&M", "BAJAJFINSV", "WIPRO",
    "NESTLEIND", "ULTRACEMCO", "JSWSTEEL", "GRASIM", "TECHM", "EICHERMOT", "HDFCLIFE", "BPCL", "DRREDDY",
    "BRITANNIA", "CIPLA", "APOLLOHOSP", "TATACONSUM", "SHRIRAMFIN", "BAJAJ-AUTO", "BEL", "HEROMOTOCO",
    "INDUSINDBK", "SBILIFE", "TRENT", "DIVISLAB", "PIDILITIND"
]

config_path = Path("config/angel_one.yaml")
if not config_path.exists():
    print(f"[!] Error: {config_path} not found.")
    sys.exit(1)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f).get("smartapi", {})

lake = DataLake(read_only=False)
fetcher = AngelHistoricalFetcher(
    api_key=cfg.get("api_key"),
    client_code=cfg.get("client_code"),
    password=cfg.get("password") or cfg.get("pin"),
    totp_secret=cfg.get("totp_secret") or cfg.get("totp_token"),
    data_lake=lake,
)

print("=" * 110)
print(f"[*] INGESTING NIFTY 50 DATA UP TO TODAY (2026-08-28 15:30 IST)")
print(f"[*] Total Equities: {len(NIFTY_50_UNIVERSE)} | Timeframe: 15m")
print("=" * 110)

try:
    fetcher.initialize_session()
    print("[+] Logged into Angel One SmartAPI successfully.\n")
except Exception as e:
    print(f"[!] SmartAPI Login failed: {e}")
    sys.exit(1)

from_date = "2026-08-15 09:15"
to_date = datetime.now().strftime("%Y-%m-%d %H:%M")

success_count = 0
failed_syms = []

for idx, sym in enumerate(NIFTY_50_UNIVERSE, 1):
    print(f"[{idx:02d}/50] Ingesting {sym:12s} (15m)... ", end="", flush=True)
    try:
        df = fetcher.fetch_and_store(
            symbol=sym,
            timeframe="15m",
            from_date=from_date,
            to_date=to_date
        )
        latest_bar = df.index.max()
        print(f"[OK -> Latest: {latest_bar} | Total Bars: {len(df):,}]")
        success_count += 1
    except Exception as e:
        print(f"[FAILED: {e}]")
        failed_syms.append(sym)
    time.sleep(0.5)

print("\n" + "=" * 110)
print(f"[*] INGESTION COMPLETE: {success_count}/50 symbols successfully updated to today.")
if failed_syms:
    print(f"[-] Failed symbols: {failed_syms}")
print("=" * 110)
