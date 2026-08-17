"""
Ashva Statistical Arbitrage & Cointegration Pairs Trading CLI
Evaluates cointegration and executes backtests on Indian banking/tech pairs.

Usage:
    python scripts/run_pairs_trading.py --asset_a HDFCBANK --asset_b ICICIBANK --timeframe 5m
"""

import argparse
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.yfinance_loader import YFinanceLoader
from src.strategies.alpha_pairs import AlphaCointegrationPairs
from src.backtest.engine import BacktestEngine
from src.analytics.tearsheet import QuantTearsheetGenerator


def main():
    parser = argparse.ArgumentParser(description="Ashva Cointegration Pairs Trading")
    parser.add_argument("--asset_a", type=str, default="HDFCBANK", help="First asset in pair")
    parser.add_argument("--asset_b", type=str, default="ICICIBANK", help="Second asset in pair")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe")

    args = parser.parse_args()
    data_lake = DataLake()
    loader = YFinanceLoader(data_lake=data_lake)

    print("=" * 80)
    print(f"[*] ASHVA STATISTICAL ARBITRAGE & COINTEGRATION LAB")
    print(f"[*] Pair: {args.asset_a} vs {args.asset_b} | Timeframe: {args.timeframe}")
    print("=" * 80)

    # 1. Load Data for Both Assets
    df_a = data_lake.load_bars(args.asset_a, args.timeframe)
    df_b = data_lake.load_bars(args.asset_b, args.timeframe)

    if df_a.empty:
        df_a = loader.fetch_and_store(args.asset_a, timeframe=args.timeframe, period="1mo")
    if df_b.empty:
        df_b = loader.fetch_and_store(args.asset_b, timeframe=args.timeframe, period="1mo")

    strat = AlphaCointegrationPairs()

    # 2. Test Cointegration
    score, p_val, is_coint = strat.test_cointegration(df_a["close"], df_b["close"])
    print(f"[*] Engle-Granger Cointegration Test:")
    print(f"    - Test Statistic : {score:.4f}")
    print(f"    - p-value        : {p_val:.4f}")
    print(f"    - Cointegrated?  : {'[YES] Statistically Significant (p < 0.05)' if is_coint else '[NO] Non-cointegrated'}")

    # 3. Spread & Beta
    spread, z_score, beta = strat.calculate_spread_and_zscore(df_a["close"], df_b["close"])
    print(f"    - Dynamic Hedge Beta: {beta:.4f}")
    print(f"    - Current Spread Z-Score: {z_score.iloc[-1]:+.2f}\n")

    # 4. Generate Signals & Backtest on Asset A
    signals_df = strat.generate_signals(df_a)
    engine = BacktestEngine(initial_capital=500000.0)
    res = engine.run(signals_df, symbol=f"{args.asset_a}_{args.asset_b}", strategy_id=strat.strategy_id)

    print("=" * 80)
    print("[*] PAIRS STRATEGY PERFORMANCE SUMMARY (NET OF INDIAN TAXES)")
    print("=" * 80)
    for k, v in res.summary().items():
        print(f"  {k:28s}: {v}")
    print("=" * 80)

    # 5. Generate Tearsheet
    tearsheet_gen = QuantTearsheetGenerator()
    tearsheet_path = tearsheet_gen.generate_html_tearsheet(res)
    print(f"[+] Quant Tearsheet exported to: {tearsheet_path}")


if __name__ == "__main__":
    main()
