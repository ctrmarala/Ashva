"""
Ashva Data Sync Script
CLI tool to download historical market data for Indian equities/indices and store in DataLake.

Usage:
    python scripts/run_data_sync.py --symbol RELIANCE --timeframe 5m --period 1mo
    python scripts/run_data_sync.py --universe --timeframe 15m --period 3mo
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher


def load_config():
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def sync_symbol(symbol: str, timeframe: str, data_lake: DataLake):
    angel_cfg = Path("config/angel_one.yaml")
    if not angel_cfg.exists():
        print(f"[-] Angel One configuration {angel_cfg} not found. Ingestion requires active credentials.")
        return
    with open(angel_cfg, "r") as f:
        cfg = yaml.safe_load(f).get("smartapi", {})
    fetcher = AngelHistoricalFetcher(
        api_key=cfg.get("api_key"),
        client_code=cfg.get("client_code"),
        password=cfg.get("password") or cfg.get("pin"),
        totp_secret=cfg.get("totp_secret") or cfg.get("totp_token"),
        data_lake=data_lake,
    )
    fetcher.initialize_session()
    print(f"[*] Fetching historical {timeframe} data for {symbol} via Angel One SmartAPI...")
    try:
        df = fetcher.fetch_and_store(symbol=symbol, timeframe=timeframe)
        print(f"[+] Successfully saved {len(df)} candles for {symbol} into Data Lake.")
    except Exception as e:
        print(f"[-] Error syncing {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ashva Historical Data Sync Utility")
    parser.add_argument("--symbol", type=str, help="Single stock symbol (e.g. RELIANCE, TCS, ^NSEI)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe (1m, 5m, 15m, 1h, 1d)")
    parser.add_argument("--period", type=str, default="1mo", help="Lookback period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--universe", action="store_true", help="Sync all default universe tickers from settings.yaml")
    
    args = parser.parse_args()
    config = load_config()
    data_lake = DataLake()

    if args.universe:
        tickers = config.get("universe", {}).get("primary_tickers", ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])
        benchmark = config.get("universe", {}).get("benchmark", "^NSEI")
        all_symbols = tickers
        print(f"[*] Syncing full universe ({len(all_symbols)} assets): {all_symbols}")
        for sym in all_symbols:
            sync_symbol(sym, args.timeframe, data_lake)
    elif args.symbol:
        sync_symbol(args.symbol, args.timeframe, data_lake)
    else:
        print("[!] Please specify --symbol <TICKER> or --universe. Run with --help for options.")


if __name__ == "__main__":
    main()
