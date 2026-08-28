"""
Ashva NIFTY 50 Full Multi-Timeframe Historical Data Ingestion Engine
Ingests clean 540-day intraday data for all 50 Nifty equities directly from Angel One SmartAPI.
Covers 5m, 30m, and 1d intervals, complementing the already-cached 15m bars.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import pandas as pd
import yaml

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher
from src.core.universe_manager import get_universe_symbols, get_universe_name

TARGET_UNIVERSE = get_universe_symbols()
NIFTY_50_UNIVERSE = TARGET_UNIVERSE

INTERVAL_SETTINGS = {
    "1m": {"interval": "ONE_MINUTE", "chunk_days": 25, "min_bars": 50000},
    "5m": {"interval": "FIVE_MINUTE", "chunk_days": 60, "min_bars": 15000},
    "10m": {"interval": "TEN_MINUTE", "chunk_days": 90, "min_bars": 7000},
    "15m": {"interval": "FIFTEEN_MINUTE", "chunk_days": 60, "min_bars": 5000},
    "30m": {"interval": "THIRTY_MINUTE", "chunk_days": 120, "min_bars": 3000},
    "1d": {"interval": "ONE_DAY", "chunk_days": 540, "min_bars": 300},
}

MAX_HORIZON_DAYS = 540


def run_multi_timeframe_ingestion(timeframes: List[str] = ["5m", "30m", "1d"], days: int = 540):
    days = min(days, MAX_HORIZON_DAYS)
    print("=" * 100)
    print(f"[*] ASHVA ANGEL ONE MULTI-TIMEFRAME INGESTION ENGINE")
    print(f"[*] Universe: {len(NIFTY_50_UNIVERSE)} Equities | Timeframes: {timeframes} | Horizon: {days} Days (18 Months)")
    print("=" * 100)

    # 1. Load Angel One credentials
    config_path = Path("config/angel_one.yaml")
    if not config_path.exists():
        print(f"[!] Error: {config_path} not found.")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f).get("smartapi", {})

    settings_cfg = Path("config/settings.yaml")
    token_file_path = "config/nifty50_tokens.json"
    if settings_cfg.exists():
        try:
            with open(settings_cfg, "r") as sf:
                s_data = yaml.safe_load(sf) or {}
                token_file_path = s_data.get("universe", {}).get("token_file", "config/nifty50_tokens.json")
        except Exception:
            pass

    token_file = Path(token_file_path)
    token_map: Dict[str, str] = {}
    if token_file.exists():
        try:
            with open(token_file, "r") as f:
                token_map = json.load(f)
        except Exception:
            token_map = {}

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

    for tf in timeframes:
        tf_clean = tf.lower()
        if tf_clean not in INTERVAL_SETTINGS:
            print(f"[!] Warning: Timeframe '{tf_clean}' not supported. Skipping.")
            continue

        setting = INTERVAL_SETTINGS[tf_clean]
        interval_code = setting["interval"]
        chunk_size_days = setting["chunk_days"]
        min_bars_thresh = setting["min_bars"]

        print(f"\n" + "-" * 80)
        print(f"[*] STARTING INGESTION FOR TIMEFRAME: {tf_clean.upper()} ({interval_code})")
        print("-" * 80)

        success_count = 0
        failed_symbols = []

        for idx, symbol in enumerate(TARGET_UNIVERSE, 1):
            print(f"[{idx:02d}/{len(TARGET_UNIVERSE)}] Ingesting {symbol:12s} ({tf_clean})...", end="", flush=True)

            existing = lake.load_bars(symbol.upper(), tf_clean)
            if not existing.empty and len(existing) >= min_bars_thresh:
                print(f" [OK: Already cached ({len(existing):,} bars)]")
                success_count += 1
                continue

            token = token_map.get(symbol.upper()) or fetcher.get_token_for_symbol(symbol.upper())
            if not token:
                print(f" [FAILED: Token not found in token map or Angel One Scrip Master]")
                failed_symbols.append(symbol)
                continue

            try:
                all_chunks = []
                curr_end = end_date
                total_fetched_days = 0

                while total_fetched_days < days:
                    fetch_days = min(chunk_size_days, days - total_fetched_days)
                    curr_start = curr_end - timedelta(days=fetch_days)
                    f_str = curr_start.strftime("%Y-%m-%d 09:15")
                    t_str = curr_end.strftime("%Y-%m-%d 15:30")

                    params = {
                        "exchange": "NSE",
                        "symboltoken": token,
                        "interval": interval_code,
                        "fromdate": f_str,
                        "todate": t_str,
                    }

                    time.sleep(0.35)  # 2.8 req/sec (safely within 3 req/sec limit)

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

                    total_fetched_days += fetch_days
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

                    lake.save_bars(
                        df=merged_df,
                        symbol=symbol.upper(),
                        timeframe=tf_clean,
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

        print(f"\n[*] {tf_clean.upper()} SUMMARY: {success_count}/{len(TARGET_UNIVERSE)} Symbols Stored.")

    print("\n" + "=" * 100)
    print(f"[*] ALL REQUESTED TIMEFRAMES INGESTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    tfs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["5m", "30m", "1d"]
    run_multi_timeframe_ingestion(timeframes=tfs, days=540)
