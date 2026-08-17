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
    "BAJFINANCE",
    "MARUTI",
    "TATAMOTORS",
    "SUNPHARMA",
]


# Ashva 18-Month Historical Horizon Policy
MAX_HORIZON_DAYS = 540  # Hard ceiling: 18 Months (~1.5 Years)


def ingest_data(timeframe: str = "15m", days: int = 540):
    days = min(days, MAX_HORIZON_DAYS)
    print("=" * 80)
    print(f"[*] INGESTING ANGEL ONE HISTORICAL DATA ({timeframe.upper()} - LAST {days} DAYS / 18 MONTHS CHUNKED)")
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

    end_date = datetime.now()
    chunk_size_days = 60
    success_count = 0

    for symbol in QUALIFIED_UNIVERSE:
        try:
            print(f"\n[>] Ingesting {symbol} ({timeframe}) across {days} days...")
            all_chunks = []
            
            # Fetch in 60-day chunks moving backwards
            curr_end = end_date
            total_fetched_days = 0
            while total_fetched_days < days:
                curr_start = curr_end - timedelta(days=min(chunk_size_days, days - total_fetched_days))
                
                # Fetch chunk from Angel One
                token = fetcher.get_token_for_symbol(symbol, "NSE")
                if not token:
                    print(f"    [!] Could not find token for {symbol}")
                    break
                
                interval = fetcher.INTERVAL_MAP.get(timeframe.lower())
                f_str = curr_start.strftime("%Y-%m-%d %H:%M")
                t_str = curr_end.strftime("%Y-%m-%d %H:%M")
                
                import time as _t
                _t.sleep(0.6)
                
                params = {
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": interval,
                    "fromdate": f_str,
                    "todate": t_str,
                }
                
                candle_resp = fetcher.smart_api.getCandleData(params)
                if isinstance(candle_resp, dict) and candle_resp.get("status"):
                    raw_data = candle_resp.get("data", [])
                    if raw_data:
                        chunk_df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"])
                        all_chunks.append(chunk_df)
                        print(f"    [+] Chunk ({curr_start.strftime('%Y-%m-%d')} to {curr_end.strftime('%Y-%m-%d')}): {len(chunk_df)} bars")
                
                curr_end = curr_start
                total_fetched_days += chunk_size_days

            if all_chunks:
                full_df = pd.concat(all_chunks, ignore_index=True)
                full_df.drop_duplicates(subset=["timestamp"], inplace=True)
                full_df.sort_values(by="timestamp", inplace=True)
                full_df["open"] = full_df["open"].astype(float)
                full_df["high"] = full_df["high"].astype(float)
                full_df["low"] = full_df["low"].astype(float)
                full_df["close"] = full_df["close"].astype(float)
                full_df["volume"] = full_df["volume"].astype(int)

                lake.save_bars(full_df, symbol=symbol.upper(), timeframe=timeframe.lower(), source="ANGEL_ONE")
                print(f"    [+] Successfully stored {len(full_df)} total bars for {symbol} ({full_df['timestamp'].min()} to {full_df['timestamp'].max()})")
                success_count += 1
            else:
                print(f"    [!] No data returned for {symbol}")
        except Exception as e:
            print(f"    [!] Error ingesting {symbol}: {e}")

    print("\n" + "=" * 80)
    print(f"[+] 180-Day Ingestion complete: {success_count}/{len(QUALIFIED_UNIVERSE)} symbols stored in Data Lake.")
    print("=" * 80)


if __name__ == "__main__":
    ingest_data(timeframe="15m", days=180)
