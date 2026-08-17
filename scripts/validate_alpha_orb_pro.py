"""
Ashva Quantitative Alpha 02 Validation Engine: Auction ORB Pro
Executes the institutional research lifecycle:
1. Ingests clean Angel One SmartAPI 15m candles across 11 liquid NSE stocks.
2. Backtests Auction ORB Pro with Risk-Budget Sizing and exact Indian Regulatory Costs.
3. Evaluates 5-Tier Slippage Stress Matrix (1 to 20 bps).
4. Runs full 4-Gate Statistical Validation (CPCV, DSR, 5000-run Monte Carlo, Net PF).
5. Outputs executive institutional scorecard.
"""

from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.strategies.alpha_orb_pro import AlphaAuctionORBPro
from src.backtest.engine import BacktestEngine
from src.research.validator import StatisticalValidator
from src.analytics.indian_costs import IndianCostModel, Segment
from src.research.experiment_ledger import ResearchExperimentLedger


def run_auction_orb_validation():
    print("=" * 95)
    print("[*] ASHVA RESEARCH LAB: ALPHA 02 (AUCTION ORB PRO) INSTITUTIONAL VALIDATION")
    print("=" * 95)

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    engine = BacktestEngine(initial_capital=500000.0, cost_model=cost_model, segment=Segment.EQUITY_INTRADAY)
    ledger = ResearchExperimentLedger()
    validator = StatisticalValidator(cost_model=cost_model, experiment_ledger=ledger)

    strat = AlphaAuctionORBPro()

    universe = [
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

    results_table = []
    total_initial_capital = 500000.0
    combined_net_pnl = 0.0
    total_trades_count = 0
    total_wins = 0

    print(f"\n[+] STEP 1: BACKTESTING ACROSS {len(universe)} NSE BLUE-CHIPS (ANGEL ONE 15M CANDLES)")
    print(f"    - Risk Budget per Trade: 0.50% (Rs 2,500) | Sizing: Quantity = Rs 2,500 / Risk_Dist")
    print(f"    - Full Indian Regulatory Costs Deducted (Brokerage, STT, GST, Exchange, SEBI, Stamp Duty)\n")
    print(f"{'SYMBOL':<12} | {'NET P&L (INR)':<14} | {'NET ROI':<10} | {'WIN RATE':<9} | {'NET PF':<8} | {'TRADES':<7} | {'MAX DD':<8} | {'STATUS'}")
    print("-" * 95)

    valid_dfs = {}

    for sym in universe:
        try:
            df = lake.load_bars(sym, timeframe="15m")
            if df.empty or len(df) < 100:
                print(f"{sym:<12} | {'NO DATA':<14} | {'N/A':<10} | {'N/A':<9} | {'N/A':<8} | {'0':<7} | {'N/A':<8} | SKIP")
                continue

            valid_dfs[sym] = df
            signals_df = strat.generate_signals(df)

            # Run Backtest with 0.50% Risk per trade
            res = engine.run(
                signals_df,
                symbol=sym,
                strategy_id="Auction_ORB_Pro",
                risk_per_trade_pct=0.005,  # 0.50% risk budget
                capital_per_trade_pct=0.25, # 25% max capital cap
            )

            status_str = "PROFITABLE" if res.total_net_pnl > 0 and res.net_profit_factor >= 1.20 else "UNPROFITABLE"
            print(
                f"{sym:<12} | Rs {res.total_net_pnl:>10,.2f} | {res.net_roi_pct:>8.2f}% | "
                f"{res.win_rate_pct:>7.1f}% | {res.net_profit_factor:>7.2f} | {res.total_trades:>6} | "
                f"{res.max_drawdown_pct:>6.2f}% | {status_str}"
            )

            combined_net_pnl += res.total_net_pnl
            total_trades_count += res.total_trades
            total_wins += res.winning_trades

            results_table.append({
                "symbol": sym,
                "net_pnl": res.total_net_pnl,
                "roi_pct": res.net_roi_pct,
                "win_rate": res.win_rate_pct,
                "net_pf": res.net_profit_factor,
                "trades": res.total_trades,
                "max_dd": res.max_drawdown_pct,
                "sharpe": res.sharpe_ratio,
            })
        except Exception as e:
            print(f"{sym:<12} | ERROR: {e}")

    print("-" * 95)
    overall_roi = (combined_net_pnl / total_initial_capital) * 100.0
    overall_wr = (total_wins / total_trades_count * 100.0) if total_trades_count > 0 else 0.0
    print(f"{'PORTFOLIO':<12} | Rs {combined_net_pnl:>10,.2f} | {overall_roi:>8.2f}% | {overall_wr:>7.1f}% | {'--':<8} | {total_trades_count:>6} | {'--':<8} | {'PORTFOLIO TOTAL'}")
    print("=" * 95)

    # -------------------------------------------------------------------------
    # STEP 2: 5-TIER SLIPPAGE STRESS TESTING (TOP ASSET)
    # -------------------------------------------------------------------------
    if valid_dfs and results_table:
        best_sym = max(results_table, key=lambda x: x["net_pnl"])["symbol"]
        print(f"\n[+] STEP 2: 5-TIER SLIPPAGE STRESS TESTING (TOP ASSET: {best_sym})")
        top_signals = strat.generate_signals(valid_dfs[best_sym])
        stress_matrix = engine.run_slippage_stress_matrix(top_signals, symbol=best_sym, strategy_id="Auction_ORB_Pro")
        print(stress_matrix.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 3: 4-GATE STATISTICAL VALIDATION REPORT
    # -------------------------------------------------------------------------
    print(f"\n[+] STEP 3: EXECUTING 4-GATE STATISTICAL ALPHA VALIDATION (CPCV + DSR + MONTE CARLO)")
    eval_sym = max(results_table, key=lambda x: x["net_pnl"])["symbol"] if results_table else "INFY"
    if eval_sym in valid_dfs:
        val_report = validator.validate_hypothesis(strat, valid_dfs[eval_sym])
        print(f"    - Target Benchmark Symbol : {eval_sym}")
        print(f"    - Hypothesis ID           : {val_report.hypothesis_id}")
        print(f"    - In-Sample Sharpe        : {val_report.in_sample_sharpe:.2f}")
        print(f"    - CPCV OOS Mean Sharpe    : {val_report.cpcv_mean_sharpe:.2f} (Degradation: {val_report.cpcv_degradation_pct:.1f}%)")
        print(f"    - Deflated Sharpe p-value : {val_report.deflated_sharpe_p_value:.4f}")
        print(f"    - Monte Carlo 95th MaxDD  : {val_report.monte_carlo_95_max_dd_pct:.2f}%")
        print(f"    - Post-Tax Net PF         : {val_report.net_profit_factor_post_tax:.2f}")
        print(f"    - FINAL VERDICT           : {val_report.status.value}")
        if val_report.rejection_reasons:
            print(f"    - Gate Findings           : {val_report.rejection_reasons}")

    print("\n" + "=" * 95)
    print("[*] VALIDATION PIPELINE COMPLETED")
    print("=" * 95)


if __name__ == "__main__":
    run_auction_orb_validation()
