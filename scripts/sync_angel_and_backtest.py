"""
Ashva Angel One Official Historical Data Harvester & Strategy Validator
Fetches institutional candlestick data directly from Angel One SmartAPI servers (getCandleData)
and re-validates the ML Meta-Labeled Asymmetric Trend Pullback Strategy.

Usage:
    python scripts/sync_angel_and_backtest.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import pandas as pd

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher
from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy
from src.backtest.engine import BacktestEngine
from src.analytics.tearsheet import QuantTearsheetGenerator


def main():
    print("=" * 90)
    print("[*] ASHVA ANGEL ONE OFFICIAL HISTORICAL DATA SYNC & STRATEGY VALIDATION")
    print("=" * 90)

    # 1. Load Config
    with open("config/angel_one.yaml", "r") as f:
        cfg = yaml.safe_load(f).get("smartapi", {})

    data_lake = DataLake()
    fetcher = AngelHistoricalFetcher(
        api_key=cfg.get("api_key"),
        client_code=cfg.get("client_code"),
        password=cfg.get("password"),
        totp_secret=cfg.get("totp_secret"),
        data_lake=data_lake,
    )

    print(f"[*] Authenticating session with Angel One SmartAPI (Client: {cfg.get('client_code')})...")
    fetcher.initialize_session()
    print("[+] Angel One Session Authenticated Successfully! [PASS]")

    # 2. Date Range: Last 30-60 days
    to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M")
    timeframe = "15m"

    # Instrument Tokens on NSE
    target_symbols = {
        "INFY": "1594",
        "TCS": "11536",
        "ICICIBANK": "4963",
        "RELIANCE": "2885",
    }

    print(f"\n[*] Fetching official 15m historical candles from Angel One (From: {from_date} to {to_date})...")
    
    for symbol, token in target_symbols.items():
        try:
            print(f"    - Fetching {symbol} (Token: {token})...")
            df = fetcher.fetch_and_store(
                symbol=symbol,
                token=token,
                timeframe=timeframe,
                from_date=from_date,
                to_date=to_date,
                exchange="NSE",
            )
            print(f"      [+] Stored {len(df)} bars for {symbol} from Angel One.")
        except Exception as e:
            print(f"      [!] Error fetching {symbol}: {e}")

    # 3. Re-Validate Strategy on Angel One Data
    print("\n" + "=" * 90)
    print("[*] RE-VALIDATING ML META-LABELED ASYMMETRIC TREND PULLBACK ON ANGEL ONE DATA")
    print("=" * 90)

    tearsheet_gen = QuantTearsheetGenerator()

    for symbol in target_symbols.keys():
        df = data_lake.load_bars(symbol, timeframe)
        if df.empty:
            continue

        primary = AlphaInstitutionalTrendPullback()
        meta_strat = AlphaMetaLabeledStrategy(primary_strategy=primary, parameters={"min_conviction_threshold": 0.50})
        
        # Fit on historical and generate signals
        meta_strat.fit_meta_model(df)
        signals_df = meta_strat.generate_signals(df)

        engine = BacktestEngine(initial_capital=500000.0)
        res = engine.run(signals_df, symbol=symbol, strategy_id=meta_strat.strategy_id)
        s = res.summary()

        status = "[PROFITABLE]" if s["total_net_pnl"] > 0 else "[LOSS]"
        print(f"{status:12s} {symbol:10s} | Net PnL: Rs {s['total_net_pnl']:>10,.2f} | Net ROI: {s['net_roi_pct']:>6.2f}% | WinRate: {s['win_rate_pct']:>5.1f}% | Net PF: {s['net_profit_factor']:>4.2f} | Trades: {s['total_trades']:>2d} | Sharpe: {res.sharpe_ratio:>5.2f} | MaxDD: {res.max_drawdown_pct:>4.2f}%")

        if s["total_net_pnl"] > 0:
            path = tearsheet_gen.generate_html_tearsheet(res)
            print(f"             +---> Tearsheet generated: {path}")

    print("=" * 90)


if __name__ == "__main__":
    main()
