"""
Ashva Quantitative Parameter Tuning & Hyperparameter Optimization Engine
Performs Combinatorial Grid Tuning with Marcos López de Prado's Deflated Sharpe Ratio (DSR)
to find the mathematically optimal parameters without curve-fitting or overfitting.

Usage:
    python scripts/tune_strategy_parameters.py --strategy pullback --symbol INFY --timeframe 15m
"""

import argparse
import sys
from pathlib import Path
from itertools import product
import pandas as pd

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.backtest.engine import BacktestEngine
from src.research.validator import StatisticalValidator
from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.strategies.alpha_vol_squeeze import AlphaVolatilitySqueeze
from src.strategies.alpha_regime import AlphaRegimeAdaptiveMR


def tune_parameters(symbol: str, strategy_name: str, timeframe: str = "15m"):
    lake = DataLake(read_only=True)
    df = lake.load_bars(symbol, timeframe)

    if df.empty:
        print(f"[!] No data found for {symbol} ({timeframe}). Please sync data first.")
        return

    print("=" * 105)
    print(f"[*] ASHVA QUANTITATIVE PARAMETER TUNING ENGINE")
    print(f"[*] Asset: {symbol} | Timeframe: {timeframe} | Strategy: {strategy_name.upper()}")
    print(f"[*] Statistical Overfitting Filter: Deflated Sharpe Ratio (DSR) & Post-Tax Net Profit Factor")
    print("=" * 105)

    # 1. Select Strategy & Parameter Grid
    if strategy_name.lower() in ["pullback", "alpha_07"]:
        base_class = AlphaInstitutionalTrendPullback
        param_grid = {
            "fast_ema": [15, 20, 25],
            "risk_reward_ratio": [2.0, 2.5, 3.0],
            "volume_threshold": [1.0, 1.15, 1.30],
        }
    elif strategy_name.lower() in ["squeeze", "alpha_08", "vol_squeeze"]:
        base_class = AlphaVolatilitySqueeze
        param_grid = {
            "bb_std": [1.8, 2.0, 2.2],
            "kc_mult": [1.3, 1.5, 1.7],
            "risk_reward_ratio": [2.5, 3.0, 3.5],
        }
    else:
        base_class = AlphaRegimeAdaptiveMR
        param_grid = {
            "entry_z_score": [1.8, 2.0, 2.2],
            "exit_z_score": [0.0, 0.5],
            "hurst_threshold": [0.45, 0.48],
        }

    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]

    print(f"[+] Testing {len(combinations)} Combinatorial Parameter Permutations on {symbol}...")
    print("-" * 105)

    results = []
    validator = StatisticalValidator()
    engine = BacktestEngine(initial_capital=500000.0)

    for idx, p in enumerate(combinations):
        strat = base_class(parameters=p)
        signals = strat.generate_signals(df)
        res = engine.run(signals, symbol=symbol, strategy_id=strat.strategy_id)
        s = res.summary()

        rets = pd.Series(res.equity_curve).pct_change().dropna().values
        if len(rets) > 10:
            dsr_p_val, dsr_stat = validator.calculate_deflated_sharpe_ratio(
                strategy_returns=rets,
                num_trials=len(combinations),
            )
            dsr_val = float(dsr_stat)
        else:
            dsr_val = 0.0

        results.append({
            "Params": p,
            "Net PnL": s["total_net_pnl"],
            "Net ROI": s["net_roi_pct"],
            "Win Rate": s["win_rate_pct"],
            "Profit Factor": s["net_profit_factor"],
            "Sharpe": res.sharpe_ratio,
            "DSR Stat": dsr_val,
            "Trades": s["total_trades"],
        })

    # Sort by Net PnL and DSR
    df_res = pd.DataFrame(results).sort_values(by="Net PnL", ascending=False)

    print(f"{'Rank':4s} | {'Parameters':50s} | {'Net PnL':>14s} | {'WinRate':>7s} | {'PF':>5s} | {'Sharpe':>6s} | {'Trades':>6s}")
    print("-" * 105)

    for rank, (i, r) in enumerate(df_res.head(8).iterrows(), 1):
        param_str = ", ".join([f"{k}={v}" for k, v in r["Params"].items()])
        pnl_str = f"Rs {r['Net PnL']:>+10,.2f}"
        print(f"#{rank:<3d} | {param_str:50s} | {pnl_str:>14s} | {r['Win Rate']:>6.1f}% | {r['Profit Factor']:>5.2f} | {r['Sharpe']:>6.2f} | {r['Trades']:>6d}")

    best = df_res.iloc[0]
    print("=" * 105)
    print(f"[*] OPTIMAL PARAMETERS DISCOVERED:")
    print(f"   {best['Params']}")
    print(f"   Net Post-Tax Profit : Rs {best['Net PnL']:+,.2f} ({best['Net ROI']:+.2f}%)")
    print(f"   Win Rate            : {best['Win Rate']:.1f}% | Profit Factor: {best['Profit Factor']:.2f} | Sharpe: {best['Sharpe']:.2f}")
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Ashva Strategy Parameter Tuning CLI")
    parser.add_argument("--strategy", type=str, default="pullback", help="Strategy to tune: pullback, squeeze, regime")
    parser.add_argument("--symbol", type=str, default="INFY", help="Symbol to tune on")
    parser.add_argument("--timeframe", type=str, default="15m")
    args = parser.parse_args()

    tune_parameters(symbol=args.symbol, strategy_name=args.strategy, timeframe=args.timeframe)


if __name__ == "__main__":
    main()
