"""
Ashva NIFTY 50 Full 540-Day Historical Data Ingestion Pipeline
Fetches clean 540-day intraday bar data for all 50 Nifty equities directly from Angel One SmartAPI.
Stores in Ashva DataLake (parquet) for offline backtesting and alpha discovery.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd
import yaml

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher
from src.core.universe_manager import get_universe_symbols, get_universe_name

TARGET_UNIVERSE = get_universe_symbols()
UNIVERSE_NAME = get_universe_name()
NIFTY_50_UNIVERSE = TARGET_UNIVERSE

MAX_HORIZON_DAYS = 540


def run_ingestion(timeframe: str = "15m", days: int = 540):
    days = min(days, MAX_HORIZON_DAYS)
    print("=" * 100)
    print(f"[*] ASHVA ANGEL ONE HISTORICAL INGESTION ENGINE")
    print(f"[*] Universe: {len(TARGET_UNIVERSE)} {UNIVERSE_NAME} Equities | Timeframe: {timeframe.upper()} | Lookback: {days} Days (18 Months)")
    print("=" * 100)

    # 1. Load Angel One credentials
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

    end_date = datetime.now()
    chunk_size_days = 60
    success_count = 0
    failed_symbols = []

    for idx, symbol in enumerate(TARGET_UNIVERSE, 1):
        print(f"\n[{idx:02d}/{len(TARGET_UNIVERSE)}] Ingesting {symbol:12s} ({timeframe})...", end="", flush=True)
        try:
            existing = lake.load_bars(symbol.upper(), timeframe.lower())
            if not existing.empty and len(existing) >= 5000:
                print(f" [OK: Already cached ({len(existing):,} bars)]")
                success_count += 1
                continue

            token = fetcher.get_token_for_symbol(symbol, "NSE")
            if not token:
                print(f" [FAILED: Token not found in Scrip Master]")
                failed_symbols.append(symbol)
                continue

            interval = fetcher.INTERVAL_MAP.get(timeframe.lower())
            if not interval:
                print(f" [FAILED: Unsupported timeframe {timeframe}]")
                break

            all_chunks = []
            curr_end = end_date
            total_fetched_days = 0

            while total_fetched_days < days:
                curr_start = curr_end - timedelta(days=min(chunk_size_days, days - total_fetched_days))
                f_str = curr_start.strftime("%Y-%m-%d 09:15")
                t_str = curr_end.strftime("%Y-%m-%d 15:30")

                params = {
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": interval,
                    "fromdate": f_str,
                    "todate": t_str,
                }

                time.sleep(0.4)  # 2.5 req/sec (Safe within Angel One 3 req/sec limit)

                candle_response = None
                for attempt in range(3):
                    try:
                        candle_response = fetcher.smart_api.getCandleData(params)
                        if isinstance(candle_response, dict) and candle_response.get("status"):
                            break
                    except Exception:
                        time.sleep(1.0 * (attempt + 1))

                if isinstance(candle_response, dict) and candle_response.get("status"):
                    raw_data = candle_response.get("data", [])
                    if raw_data:
                        df_chunk = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        all_chunks.append(df_chunk)

                total_fetched_days += min(chunk_size_days, days - total_fetched_days)
                curr_end = curr_start

            if all_chunks:
                merged_df = pd.concat(all_chunks, ignore_index=True)
                merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
                merged_df = merged_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                merged_df["open"] = merged_df["open"].astype(float)
                merged_df["high"] = merged_df["high"].astype(float)
                merged_df["low"] = merged_df["low"].astype(float)
                merged_df["close"] = merged_df["close"].astype(float)
                merged_df["volume"] = merged_df["volume"].astype(int)

                # Save directly to DataLake
                lake.save_bars(
                    df=merged_df,
                    symbol=symbol.upper(),
                    timeframe=timeframe.lower(),
                    source="ANGEL_ONE"
                )
                bars_count = len(merged_df)
                min_date = merged_df['timestamp'].min().strftime('%Y-%m-%d')
                max_date = merged_df['timestamp'].max().strftime('%Y-%m-%d')
                print(f" [OK: {bars_count:,} bars ({min_date} -> {max_date})]")
                success_count += 1
            else:
                print(" [FAILED: No candle data returned]")
                failed_symbols.append(symbol)

        except Exception as e:
            print(f" [FAILED: {e}]")
            failed_symbols.append(symbol)

    print("\n" + "=" * 100)
    print(f"[*] INGESTION SUMMARY: {success_count}/{len(NIFTY_50_UNIVERSE)} Symbols Successfully Stored in DataLake")
    if failed_symbols:
        print(f"[!] Failed symbols: {failed_symbols}")
    print("=" * 100)


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 540
    run_ingestion(timeframe=tf, days=d)
