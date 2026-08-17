"""
Ashva Automated Asset-to-Strategy Universe Scanner CLI
Demonstrates how Ashva analyzes microstructure signatures and assigns the optimal strategy to each asset.

Usage:
    python scripts/run_asset_strategy_scanner.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.portfolio.strategy_selector import StrategySelector


def main():
    print("=" * 105)
    print("[*] ASHVA AUTOMATED ASSET-TO-STRATEGY DYNAMIC SELECTION ENGINE")
    print("[*] Market Regimes: Hurst Exponent (Memory) | Bollinger-Keltner Volatility Squeeze | VWAP Drift")
    print("=" * 105)

    selector = StrategySelector()
    universe = ["INFY", "TCS", "ICICIBANK", "RELIANCE"]

    df_scan = selector.scan_universe(universe, timeframe="15m")

    for idx, row in df_scan.iterrows():
        print(f"\n[+] Stock: {row['Symbol']:10s} | Active Regime: {row['Regime']}")
        print(f"    - Hurst Exponent    : {row['Hurst']}")
        print(f"    - In Vol Squeeze?   : {row['In Squeeze']}")
        print(f"    - ASSIGNED STRATEGY : \033[92m{row['Assigned Strategy']}\033[0m")
        print(f"    - Quantitative Why  : {row['Rationale']}")

    print("\n" + "=" * 105)
    print("[*] SUMMARY MAPPING FOR EXECUTION RUNNER:")
    for idx, row in df_scan.iterrows():
        print(f"    - {row['Symbol']:10s} ===> {row['Assigned Strategy']}")
    print("=" * 105)


if __name__ == "__main__":
    main()
