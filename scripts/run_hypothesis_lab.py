"""
Ashva Master Quantitative Research & Hypothesis Lab CLI
Unified institutional entry-point for Alpha discovery, multi-regime backtesting,
and statistical validation across Indian equity and derivative markets.

Usage:
    # 1. Master audit across all strategies & 14 blue chips (18-month historical data):
    python scripts/run_hypothesis_lab.py --all

    # 2. Test a specific strategy on a single symbol with multi-regime breakdown:
    python scripts/run_hypothesis_lab.py --strategy alpha_02 --symbol TCS --regimes

    # 3. Test a strategy across all 14 symbols with slippage stress test:
    python scripts/run_hypothesis_lab.py --strategy alpha_04 --all-symbols --stress

    # 4. Generate standalone dark-theme HTML tearsheets:
    python scripts/run_hypothesis_lab.py --strategy alpha_02 --symbol TCS --tearsheet
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.research.validator import StatisticalValidator
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.analytics.tearsheet import QuantTearsheetGenerator
from src.research.hypothesis import HypothesisStatus, StrategyHorizon

# Strategy Registry
from src.strategies.alpha_trend_surfer import AlphaTrendSurfer
from src.strategies.alpha_orb_pro import AlphaAuctionORBPro
from src.strategies.alpha_03_vwap_reversion import Alpha03VWAPReversion
from src.strategies.alpha_04_gap_and_go import Alpha04GapAndGo
from src.strategies.alpha_05_opening_drive_pullback import Alpha05OpeningDrivePullback
from src.strategies.alpha_06_pdh_pdl_sweep import Alpha06PDHPDLSweep
from src.strategies.alpha_07_opening_volatility_expansion import Alpha07OpeningVolatilityExpansion
from src.strategies.alpha_08_opening_imbalance import Alpha08OpeningImbalance
from src.strategies.alpha_09_opening_relative_strength import Alpha09OpeningRelativeStrength
from src.strategies.alpha_10_statistical_range_reversion import Alpha10StatisticalRangeReversion
from src.strategies.alpha_11_donchian_breakout import Alpha11DonchianBreakout
from src.strategies.alpha_12_european_open_momentum import Alpha12EuropeanOpenMomentum
from src.strategies.alpha_13_htf_aligned_orb import Alpha13HTFAlignedORB
from src.strategies.alpha_14_gap_momentum_drift import Alpha14GapMomentumDrift
from src.strategies.alpha_15_nr7_volatility_expansion import Alpha15NR7VolatilityExpansion
from src.strategies.alpha_16_inside_day_breakout import Alpha16InsideDayBreakout
from src.strategies.alpha_17_volume_shock_momentum import Alpha17VolumeShockMomentum
from src.strategies.alpha_18_three_day_trend_orb import Alpha18ThreeDayTrendORB
from src.strategies.alpha_19_power_hour_momentum import Alpha19PowerHourMomentum
from src.strategies.alpha_20_vwap_trend_continuation import Alpha20VWAPTrendContinuation
from src.strategies.alpha_21_high_velocity_momentum import Alpha21HighVelocityMomentum
from src.strategies.alpha_22_apex_momentum import Alpha22ApexMomentum
from src.strategies.alpha_23_velocity_50_scanner import Alpha23Velocity50Scanner
from src.strategies.alpha_24_volatility_vacuum_release import Alpha24VolatilityVacuumRelease
from src.strategies.alpha_25_cross_sectional_residual_reversion import Alpha25CrossSectionalResidualReversion

STRATEGY_MAP = {
    "alpha_01": ("ALPHA_01_TRENDSURFER", AlphaTrendSurfer),
    "alpha_02": ("ALPHA_02_AUCTION_ORB", AlphaAuctionORBPro),
    "alpha_03": ("ALPHA_03_VWAP_REVERSION", Alpha03VWAPReversion),
    "alpha_04": ("ALPHA_04_GAP_AND_GO", Alpha04GapAndGo),
    "alpha_05": ("ALPHA_05_OPENING_DRIVE_PULLBACK", Alpha05OpeningDrivePullback),
    "alpha_06": ("ALPHA_06_PDH_PDL_SWEEP", Alpha06PDHPDLSweep),
    "alpha_07": ("ALPHA_07_OPENING_VOLATILITY_EXPANSION", Alpha07OpeningVolatilityExpansion),
    "alpha_08": ("ALPHA_08_OPENING_IMBALANCE", Alpha08OpeningImbalance),
    "alpha_09": ("ALPHA_09_OPENING_RELATIVE_STRENGTH", Alpha09OpeningRelativeStrength),
    "alpha_10": ("ALPHA_10_STATISTICAL_RANGE_REVERSION", Alpha10StatisticalRangeReversion),
    "alpha_11": ("ALPHA_11_DONCHIAN_BREAKOUT", Alpha11DonchianBreakout),
    "alpha_12": ("ALPHA_12_EUROPEAN_OPEN_MOMENTUM", Alpha12EuropeanOpenMomentum),
    "alpha_13": ("ALPHA_13_HTF_ALIGNED_ORB", Alpha13HTFAlignedORB),
    "alpha_14": ("ALPHA_14_GAP_MOMENTUM_DRIFT", Alpha14GapMomentumDrift),
    "alpha_15": ("ALPHA_15_NR7_VOLATILITY_EXPANSION", Alpha15NR7VolatilityExpansion),
    "alpha_16": ("ALPHA_16_INSIDE_DAY_BREAKOUT", Alpha16InsideDayBreakout),
    "alpha_17": ("ALPHA_17_VOLUME_SHOCK_MOMENTUM", Alpha17VolumeShockMomentum),
    "alpha_18": ("ALPHA_18_THREE_DAY_TREND_ORB", Alpha18ThreeDayTrendORB),
    "alpha_19": ("ALPHA_19_POWER_HOUR_MOMENTUM", Alpha19PowerHourMomentum),
    "alpha_20": ("ALPHA_20_VWAP_TREND_CONTINUATION", Alpha20VWAPTrendContinuation),
    "alpha_21": ("ALPHA_21_HIGH_VELOCITY_MOMENTUM", Alpha21HighVelocityMomentum),
    "alpha_22": ("ALPHA_22_APEX_MOMENTUM", Alpha22ApexMomentum),
    "alpha_23": ("ALPHA_23_VELOCITY_50_SCANNER", Alpha23Velocity50Scanner),
    "alpha_24": ("ALPHA_24_VOLATILITY_VACUUM_RELEASE", Alpha24VolatilityVacuumRelease),
    "alpha_25": ("ALPHA_25_CROSS_SECTIONAL_RESIDUAL_REVERSION", Alpha25CrossSectionalResidualReversion),
}

DEFAULT_UNIVERSE = [
    "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
    "BAJFINANCE", "MARUTI", "SUNPHARMA"
]


def run_strategy_backtest(
    strat_id: str,
    strat_obj,
    symbols: List[str],
    lake: DataLake,
    engine: BacktestEngine,
    validator: StatisticalValidator,
    timeframe: str = "15m",
) -> List[Dict]:
    results = []

    # Configure Engine Segment & Timeframe based on Strategy Horizon
    is_swing = False
    if hasattr(strat_obj, "metadata"):
        h = getattr(strat_obj.metadata, "horizon", None)
        if h in [StrategyHorizon.SWING, StrategyHorizon.POSITIONAL] or str(h) in ["SWING", "POSITIONAL", "StrategyHorizon.SWING", "StrategyHorizon.POSITIONAL"]:
            is_swing = True

    engine.segment = Segment.EQUITY_DELIVERY if is_swing else Segment.EQUITY_INTRADAY
    target_tf = getattr(strat_obj.metadata, "timeframe", timeframe) if hasattr(strat_obj, "metadata") else timeframe

    for sym in symbols:
        # Load bars enforcing 540-day research ceiling
        df = lake.load_bars(sym, target_tf, max_lookback_days=540)
        if df.empty or len(df) < 50:
            # Fallback to default timeframe if daily bars not yet ingested
            if target_tf != timeframe:
                df = lake.load_bars(sym, timeframe, max_lookback_days=540)
            if df.empty or len(df) < 50:
                continue

        # 1. Run Baseline Backtest
        signals_df = strat_obj.generate_signals(df)
        res = engine.run(signals_df, symbol=sym, strategy_id=strat_id, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)

        # 2. Run Centralized Statistical Validation (Single Source of Truth with Explicit Symbol)
        report = validator.validate_hypothesis(strat_obj, df, symbol=sym)

        w60 = report.window_metrics.get("60d", {})
        w180 = report.window_metrics.get("180d", {})
        w365 = report.window_metrics.get("365d", {})
        w540 = report.window_metrics.get("540d", {})

        results.append({
            "Strategy": strat_id,
            "Symbol": sym,
            "Net_PnL_INR": round(res.total_net_pnl, 2),
            "Trades": res.total_trades,
            "Tier": report.evidence_tier,
            "Win_Rate": f"{res.win_rate_pct:.1f}%",
            "PF_540d": f"{w540.get('net_pf', res.net_profit_factor):.2f}",
            "PF_365d": f"{w365.get('net_pf', 0.0):.2f}",
            "PF_180d": f"{w180.get('net_pf', 0.0):.2f}",
            "PF_60d": f"{w60.get('net_pf', 0.0):.2f}",
            "Stability": f"{report.regime_stability_score:.0f}%",
            "Regime_60d": f"{report.current_regime_score:.0f}%",
            "Recency_Q": f"{report.recency_weighted_score:+.2f}",
            "Sharpe": round(res.sharpe_ratio, 2),
            "MaxDD": f"{res.max_drawdown_pct:.2f}%",
            "Costs_INR": round(res.total_taxes_paid, 2),
            "Verdict": f"[{report.status.value}]",
            "_result_obj": res,
            "_report": report,
            "_df": df,
            "_signals_df": signals_df,
        })
    return results


def run_slippage_stress(strat_id: str, strat_obj, df: pd.DataFrame, symbol: str):
    print(f"\n[+] 5-TIER SLIPPAGE STRESS MATRIX ({symbol} - 1 to 20 bps)")
    stress_scenarios = [
        ("Optimistic", 1.0),
        ("Base", 3.0),
        ("Conservative", 5.0),
        ("Stress", 10.0),
        ("Extreme", 20.0),
    ]
    signals_df = strat_obj.generate_signals(df)
    rows = []
    for sc_name, slip_bps in stress_scenarios:
        c_model = IndianCostModel(default_slippage_bps=slip_bps)
        eng = BacktestEngine(cost_model=c_model, initial_capital=500000.0)
        r = eng.run(signals_df, symbol=symbol, strategy_id=strat_id, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)
        rows.append({
            "Scenario": sc_name,
            "Slippage_Bps": slip_bps,
            "Net_Pnl_INR": round(r.total_net_pnl, 2),
            "Net_ROI_Pct": round(r.net_roi_pct, 2),
            "Profit_Factor": round(r.net_profit_factor, 2) if r.net_profit_factor < 90 else 99.0,
            "Sharpe": round(r.sharpe_ratio, 2),
            "MaxDD_Pct": round(r.max_drawdown_pct, 2),
            "Total_Taxes_INR": round(r.total_taxes_paid, 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Ashva Quantitative Research & Hypothesis Lab")
    parser.add_argument("--all", action="store_true", help="Run master backtest across all strategies and symbols")
    parser.add_argument("--strategy", type=str, choices=list(STRATEGY_MAP.keys()) + ["all"], default="alpha_02", help="Strategy to test")
    parser.add_argument("--symbol", type=str, default="TCS", help="Target symbol")
    parser.add_argument("--all-symbols", action="store_true", help="Run across all 14 liquid blue chips")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--regimes", action="store_true", help="Run 3-Tier Multi-Regime Persistence Analysis (0-6m, 6-12m, 12-18m)")
    parser.add_argument("--stress", action="store_true", help="Run 5-Tier Slippage Stress Matrix (1-20 bps)")
    parser.add_argument("--no-tearsheet", action="store_true", help="Suppress HTML tearsheet generation")

    args = parser.parse_args()

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    validator = StatisticalValidator(cost_model=cost_model)
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0)
    ts_gen = QuantTearsheetGenerator()

    print("=" * 135)
    print("[*] ASHVA QUANTITATIVE RESEARCH LAB: 540-DAY RESEARCH CANVAS | RECENCY-WEIGHTED MULTI-WINDOW VALIDATION")
    print(f"[*] Framework: 60d (50% Wt), 180d (25% Wt), 365d (15% Wt), 540d (10% Wt) | Regulatory Costs & 3.0 bps Slippage")
    print("=" * 135)

    if args.all or args.strategy == "all":
        # Master Audit across all strategies & symbols
        strategies_to_run = list(STRATEGY_MAP.values())
        target_symbols = DEFAULT_UNIVERSE
    else:
        strategies_to_run = [STRATEGY_MAP[args.strategy]]
        target_symbols = DEFAULT_UNIVERSE if args.all_symbols else [args.symbol.upper()]

    master_results = []
    generated_tearsheets = []
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    for strat_name, strat_cls in strategies_to_run:
        strat_obj = strat_cls()
        print(f"\n" + "#" * 135)
        print(f"[+] MULTI-WINDOW RESEARCH VALIDATION: {strat_name}")
        print("#" * 135)

        results = run_strategy_backtest(strat_name, strat_obj, target_symbols, lake, engine, validator, args.timeframe)
        if not results:
            print("[-] No data found for specified symbols.")
            continue

        disp_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        print(disp_df.to_string(index=False))

        # Portfolio Aggregates
        total_pnl = sum(r["Net_PnL_INR"] for r in results)
        total_trades = sum(r["Trades"] for r in results)
        total_taxes = sum(r["Costs_INR"] for r in results)
        print("-" * 135)
        print(f"[*] {strat_name} 540D PORTFOLIO TOTAL: Net P&L = Rs {total_pnl:+,.2f} | Trades = {total_trades} | Taxes Paid = Rs {total_taxes:,.2f}")

        # Automated Tearsheet Generation for All Evaluated Candidates with Trades
        if not args.no_tearsheet:
            for r in results:
                if r["Trades"] > 0:
                    sym = r["Symbol"]
                    res_obj = r["_result_obj"]
                    try:
                        ts_path = ts_gen.generate_html_tearsheet(res_obj)
                        generated_tearsheets.append((strat_name, sym, ts_path, r["Net_PnL_INR"]))
                    except Exception as e:
                        print(f"[-] Tearsheet error for {strat_name} on {sym}: {e}")

        master_results.extend(results)

        # Detailed Analysis for Single-Symbol Mode or explicit flags
        if len(target_symbols) == 1 or args.regimes or args.stress:
            for r in results:
                sym = r["Symbol"]
                df = r["_df"]

                if args.stress:
                    run_slippage_stress(strat_name, strat_obj, df, sym)

                if args.regimes:
                    print(f"\n--- REGIME PERSISTENCE ANALYSIS (0-6m Current | 6-12m Recent | 12-18m Older): {sym} ({strat_name}) ---")
                    reg_df = validator.evaluate_multi_regime_persistence(strat_obj, df, symbol=sym)
                    if not reg_df.empty:
                        print(reg_df.to_string(index=False))

    if args.all or (len(strategies_to_run) > 1 and len(target_symbols) > 1):
        print("\n" + "=" * 115)
        print("[*] 18-MONTH MASTER VALIDATION LEADERBOARD (SORTED BY NET P&L)")
        print("=" * 115)
        m_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in master_results])
        m_df_sorted = m_df.sort_values(by="Net_PnL_INR", ascending=False)
        print(m_df_sorted.to_string(index=False))
        print("=" * 115)

    if generated_tearsheets:
        print("\n" + "=" * 115)
        print(f"[*] AUTOMATED HTML QUANT TEARSHEETS GENERATED ({len(generated_tearsheets)} Total Saved to data_lake/tearsheets/)")
        print("=" * 115)
        # Sort by Net PnL descending
        generated_tearsheets.sort(key=lambda x: x[3], reverse=True)
        for s_name, sym, path, pnl in generated_tearsheets[:15]:
            print(f"  [+] {s_name:<25} | {sym:<12} | Net P&L: Rs {pnl:+10,.2f} | File: {path}")
        if len(generated_tearsheets) > 15:
            print(f"  ... and {len(generated_tearsheets) - 15} more in data_lake/tearsheets/")
        print("=" * 115)


if __name__ == "__main__":
    main()
