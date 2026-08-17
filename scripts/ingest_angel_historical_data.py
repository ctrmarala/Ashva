"""
Ashva Angel One Official Historical Data Ingestion Pipeline
Fetches clean 60-day 15m OHLCV intraday data for the qualified 8-stock universe directly from Angel One SmartAPI.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import yaml
import pandas as pd

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher


QUALIFIED_UNIVERSE = [
    "INFY",
    "TCS",
    "ICICIBANK",
    "HDFCBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "RELIANCE",
    "LT",
    "TATASTEEL",
    "BHARTIARTL",
]


def ingest_data(timeframe: str = "15m", days: int = 60):
    print("=" * 80)
    print(f"[*] INGESTING ANGEL ONE HISTORICAL DATA ({timeframe.upper()} - LAST {days} DAYS)")
    print("=" * 80)

    # 1. Load credentials
    config_path = Path("config/angel_one.yaml")
    if not config_path.exists():
        print(f"[!] Error: {config_path} not found.")
        return

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

    try:
        fetcher.initialize_session()
        print("[+] Logged into Angel One SmartAPI successfully.")
    except Exception as e:
        print(f"[!] SmartAPI Login failed: {e}")
        return

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    success_count = 0
    for symbol in QUALIFIED_UNIVERSE:
        try:
            print(f"[>] Fetching {symbol} ({timeframe}) from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}...")
            df = fetcher.fetch_and_store(
                symbol=symbol,
                exchange="NSE",
                timeframe=timeframe,
                from_date=from_date,
                to_date=to_date,
            )
            if not df.empty:
                print(f"    [+] Saved {len(df)} bars for {symbol} (Date Range: {df.index.min()} to {df.index.max()})")
                success_count += 1
            else:
                print(f"    [!] Warning: Empty data returned for {symbol}")
        except Exception as e:
            print(f"    [!] Error fetching {symbol}: {e}")

    print("=" * 80)
    print(f"[+] Ingestion complete: {success_count}/{len(QUALIFIED_UNIVERSE)} symbols stored in Data Lake.")
    print("=" * 80)


if __name__ == "__main__":
    ingest_data(timeframe="15m", days=60)
