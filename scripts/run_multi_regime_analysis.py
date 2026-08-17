"""
Ashva Multi-Regime Quantitative Persistence Engine
Evaluates strategies across the 3-Tier Horizon Policy:
- Tier 1: Current Regime (0–6 Months / 180 Days) -> Is it working now?
- Tier 2: Recent Regime (6–12 Months / 180–365 Days) -> Is the edge persistent?
- Tier 3: Extended Context (12–18 Months / 365–540 Days) -> Does it survive prior regimes?
- Hard Ceiling: 18 Months (Never exceeds 18 months).
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.strategies.alpha_orb_pro import AlphaAuctionORBPro
from src.strategies.alpha_04_gap_and_go import Alpha04GapAndGo
from src.research.validator import StatisticalValidator
from src.analytics.indian_costs import IndianCostModel, Segment
from src.research.experiment_ledger import ResearchExperimentLedger


def run_analysis():
    print("=" * 105)
    print("[*] ASHVA INSTITUTIONAL MULTI-REGIME PERSISTENCE ANALYSIS (3-TIER HORIZON POLICY)")
    print("=" * 105)

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    validator = StatisticalValidator(cost_model=cost_model)

    candidates = [
        ("ALPHA_02_AUCTION_ORB", AlphaAuctionORBPro(), ["INFY", "BHARTIARTL", "RELIANCE", "TCS", "BAJFINANCE"]),
        ("ALPHA_04_GAP_AND_GO", Alpha04GapAndGo(), ["BAJFINANCE", "INFY", "TCS", "MARUTI", "HDFCBANK"]),
    ]

    for strat_name, strat_obj, symbols in candidates:
        print(f"\n" + "#" * 105)
        print(f"[+] STRATEGY: {strat_name}")
        print("#" * 105)

        for sym in symbols:
            df = lake.load_bars(sym, "15m")
            if df.empty or len(df) < 200:
                print(f"[-] {sym}: Insufficient data")
                continue

            regime_df = validator.evaluate_multi_regime_persistence(strat_obj, df, symbol=sym)
            if regime_df.empty:
                continue

            print(f"\n--- CANDIDATE: {sym} ({strat_name}) ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(regime_df.to_string(index=False))

    print("\n" + "=" * 105)
    print("[*] MULTI-REGIME PERSISTENCE ANALYSIS COMPLETED")
    print("=" * 105)


if __name__ == "__main__":
    run_analysis()
