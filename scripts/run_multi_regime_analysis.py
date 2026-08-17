"""
Ashva Institutional Multi-Regime Quantitative Persistence Engine
Evaluates strategies across the 3-Tier Horizon Policy:
- Tier 1: Current Regime (0–6 Months / 180 Days) -> Is it working now?
- Tier 2: Recent Regime (6–12 Months / 180–365 Days) -> Is the edge persistent?
- Tier 3: Extended Context (12–18 Months / 365–540 Days) -> Does it survive prior regimes?
- Hard Ceiling: 18 Months (Never exceeds 18 months).

Outputs:
- Multi-Regime Breakdown Tables
- Final Persistence Classification:
  * 🟢 PERSISTENT: Positive Net PF (>1.0) and Positive Sharpe in ALL 3 periods.
  * 🟡 RECENT_REGIME: Good current period (0-6m), but weak older periods (6-18m).
  * 🟠 HISTORICAL_ONLY: Old periods good, current period weak.
  * 🔴 NO_EVIDENCE: Fails across all periods.
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


def run_analysis():
    print("=" * 110)
    print("[*] ASHVA INSTITUTIONAL MULTI-REGIME PERSISTENCE ANALYSIS (3-TIER 18-MONTH POLICY)")
    print("=" * 110)

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    validator = StatisticalValidator(cost_model=cost_model)

    candidates = [
        ("ALPHA_02_AUCTION_ORB", AlphaAuctionORBPro(), ["INFY", "RELIANCE", "BHARTIARTL", "TCS", "BAJFINANCE"]),
        ("ALPHA_04_GAP_AND_GO", Alpha04GapAndGo(), ["BAJFINANCE", "INFY", "TCS", "MARUTI", "HDFCBANK"]),
    ]

    persistence_summary = []

    for strat_name, strat_obj, symbols in candidates:
        print(f"\n" + "#" * 110)
        print(f"[+] STRATEGY: {strat_name}")
        print("#" * 110)

        for sym in symbols:
            df = lake.load_bars(sym, "15m")
            if df.empty or len(df) < 100:
                print(f"[-] {sym}: Insufficient data")
                continue

            regime_df = validator.evaluate_multi_regime_persistence(strat_obj, df, symbol=sym)
            if regime_df.empty:
                continue

            print(f"\n--- CANDIDATE: {sym} ({strat_name}) ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(regime_df.to_string(index=False))

            # Extract window metrics
            c_row = regime_df[regime_df["Regime_Window"] == "Current (0-6m)"]
            r_row = regime_df[regime_df["Regime_Window"] == "Recent (6-12m)"]
            e_row = regime_df[regime_df["Regime_Window"] == "Extended (12-18m)"]
            o_row = regime_df[regime_df["Regime_Window"] == "Overall (0-18m Full)"]

            c_pf = c_row["Net_Profit_Factor"].iloc[0] if not c_row.empty else 0.0
            r_pf = r_row["Net_Profit_Factor"].iloc[0] if not r_row.empty else 0.0
            e_pf = e_row["Net_Profit_Factor"].iloc[0] if not e_row.empty else 0.0
            o_pf = o_row["Net_Profit_Factor"].iloc[0] if not o_row.empty else 0.0

            c_sh = c_row["Sharpe"].iloc[0] if not c_row.empty else 0.0
            r_sh = r_row["Sharpe"].iloc[0] if not r_row.empty else 0.0
            e_sh = e_row["Sharpe"].iloc[0] if not e_row.empty else 0.0

            total_trades = o_row["Trades"].iloc[0] if not o_row.empty else 0
            c_trades = c_row["Trades"].iloc[0] if not c_row.empty else 0

            # Classification Logic
            if c_pf >= 1.10 and r_pf >= 1.05 and e_pf >= 1.00 and c_sh > 0 and r_sh > 0:
                classification = "PERSISTENT (Multi-Regime Winner)"
            elif c_pf >= 1.15 and (r_pf < 1.0 or e_pf < 1.0):
                classification = "RECENT_REGIME (Current Edge / Decaying Historically)"
            elif c_pf < 1.0 and (r_pf >= 1.15 or e_pf >= 1.15):
                classification = "HISTORICAL_ONLY (Old Regime / Dead Currently)"
            else:
                classification = "NO_EVIDENCE / REJECTED"

            persistence_summary.append({
                "Strategy": strat_name,
                "Symbol": sym,
                "Total_Trades_18m": total_trades,
                "Current_PF_0_6m": c_pf,
                "Recent_PF_6_12m": r_pf,
                "Extended_PF_12_18m": e_pf,
                "Overall_PF_18m": o_pf,
                "Current_Sharpe": c_sh,
                "Recent_Sharpe": r_sh,
                "Extended_Sharpe": e_sh,
                "Classification": classification,
            })

    print("\n" + "=" * 110)
    print("[*] MULTI-REGIME PERSISTENCE SCORECARD")
    print("=" * 110)
    summary_df = pd.DataFrame(persistence_summary)
    print(summary_df.to_string(index=False))
    print("=" * 110)


if __name__ == "__main__":
    run_analysis()
