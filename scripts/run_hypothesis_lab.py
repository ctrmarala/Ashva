"""
Ashva Quantitative Hypothesis Lab CLI
Formulates, backtests, and validates alpha hypotheses using Marcos López de Prado's DSR, CPCV, and Monte Carlo tests.

Usage:
    python scripts/run_hypothesis_lab.py --symbol RELIANCE --timeframe 5m --hypothesis orb
    python scripts/run_hypothesis_lab.py --symbol RELIANCE --timeframe 5m --hypothesis regime
"""

import argparse
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.yfinance_loader import YFinanceLoader
from src.research.hypothesis_factory import HypothesisFactory
from src.research.validator import StatisticalValidator
from src.strategies.alpha_orb import AlphaInstitutionalORB
from src.strategies.alpha_regime import AlphaRegimeAdaptiveMR
from src.backtest.engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="Ashva Quantitative Alpha Hypothesis Lab")
    parser.add_argument("--symbol", type=str, default="RELIANCE", help="Symbol to test on (e.g. RELIANCE, TCS, INFY)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe (5m, 15m, 1h)")
    parser.add_argument("--hypothesis", type=str, choices=["orb", "regime", "all"], default="all", help="Hypothesis to test")
    parser.add_argument("--period", type=str, default="1mo", help="Lookback period (1mo, 3mo, 6mo, 1y)")

    args = parser.parse_args()
    data_lake = DataLake()
    loader = YFinanceLoader(data_lake=data_lake)

    print("=" * 80)
    print(f"[*] ASHVA QUANTITATIVE ALPHA RESEARCH LAB")
    print(f"[*] Target Symbol: {args.symbol} | Timeframe: {args.timeframe} | Period: {args.period}")
    print("=" * 80)

    # 1. Ensure Data Loaded
    df = data_lake.load_bars(args.symbol, args.timeframe)
    if df.empty or len(df) < 50:
        print(f"[*] Downloading market data for {args.symbol}...")
        df = loader.fetch_and_store(symbol=args.symbol, timeframe=args.timeframe, period=args.period)

    print(f"[+] Loaded {len(df)} bars for analysis (From {df.index[0]} to {df.index[-1]}).\n")

    factory = HypothesisFactory()
    hypotheses_to_test = []

    if args.hypothesis in ["orb", "all"]:
        hypotheses_to_test.append((AlphaInstitutionalORB, AlphaInstitutionalORB.DEFAULT_METADATA))
    if args.hypothesis in ["regime", "all"]:
        hypotheses_to_test.append((AlphaRegimeAdaptiveMR, AlphaRegimeAdaptiveMR.DEFAULT_METADATA))

    # 2. Run Hypothesis Space Validation
    for hyp_cls, metadata in hypotheses_to_test:
        print(f"\n--- [Testing Hypothesis: {metadata.name}] ---")
        print(f"[RATIONALE] {metadata.economic_rationale}\n")
        
        reports = factory.evaluate_hypothesis_space(
            hypothesis_cls=hyp_cls,
            metadata=metadata,
            df=df,
        )

        for i, rep in enumerate(reports):
            status_tag = "[ACCEPTED]" if rep.is_accepted() else "[REJECTED]"
            print(f"Trial #{i+1}: {status_tag}")
            print(f"  - In-Sample Sharpe: {rep.in_sample_sharpe:.2f} | Out-Of-Sample Sharpe: {rep.out_of_sample_sharpe:.2f}")
            print(f"  - Deflated Sharpe p-value: {rep.deflated_sharpe_p_value:.4f}")
            print(f"  - CPCV Degradation: {rep.cpcv_degradation_pct:.1f}%")
            print(f"  - Monte Carlo 95th Percentile Max Drawdown: {rep.monte_carlo_95_max_dd_pct:.1f}%")
            print(f"  - Post-Tax Net Profit Factor: {rep.net_profit_factor_post_tax:.2f}")
            if rep.rejection_reasons:
                print(f"  [!] Rejection Reasons:")
                for r in rep.rejection_reasons:
                    print(f"     * {r}")
            print()

    # 3. Backtest Simulation on Best Variant
    print("=" * 80)
    print("[*] RUNNING DETAILED INSTITUTIONAL BACKTEST ON CANDIDATE (WITH INDIAN TAXES)")
    print("=" * 80)
    orb_strat = AlphaInstitutionalORB()
    signals_df = orb_strat.generate_signals(df)
    
    engine = BacktestEngine(initial_capital=500000.0)
    bt_result = engine.run(signals_df, symbol=args.symbol, strategy_id=orb_strat.strategy_id)
    
    summary = bt_result.summary()
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
