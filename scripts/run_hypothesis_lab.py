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
from src.strategies.alpha_27_sector_momentum_drift import Alpha27SectorMomentumDrift
from src.strategies.alpha_28_value_area_expansion import Alpha28ValueAreaExpansion
from src.strategies.alpha_29_trend_exhaustion_climax import Alpha29TrendExhaustionClimax
from src.strategies.alpha_30_midday_squeeze_trend import Alpha30MiddaySqueezeTrend
from src.strategies.alpha_31_failed_opening_drive_fade import Alpha31FailedOpeningDriveFade
from src.strategies.alpha_35_nr4_opening_expansion import Alpha35NR4OpeningExpansion
from src.strategies.alpha_36_inside_day_expansion import Alpha36InsideDayExpansion
from src.strategies.alpha_37_gap_volume_shock_drift import Alpha37GapVolumeShockDrift
from src.strategies.alpha_38_three_day_trend_expansion import Alpha38ThreeDayTrendExpansion
from src.strategies.alpha_39_open_equals_extreme_drive import Alpha39OpenEqualsExtremeDrive
from src.strategies.alpha_40_double_inside_expansion import Alpha40DoubleInsideExpansion
from src.strategies.alpha_41_opening_marubozu_expansion import Alpha41OpeningMarubozuExpansion
from src.strategies.alpha_42_nr2_gap_breakout import Alpha42NR2GapBreakout
from src.strategies.alpha_43_nr3_gap_breakout import Alpha43NR3GapBreakout
from src.strategies.alpha_44_contraction_open_drive import Alpha44ContractionOpenDrive
from src.strategies.alpha_45_inside_day_gap_momentum import Alpha45InsideDayGapMomentum
from src.strategies.alpha_46_high_conviction_gap_drift import Alpha46HighConvictionGapDrift
from src.strategies.alpha_47_orb30_gap_expansion import Alpha47ORB30GapExpansion
from src.strategies.alpha_48_nr7_gap_breakout import Alpha48NR7GapBreakout
from src.strategies.alpha_49_moderate_gap_volume_shock import Alpha49ModerateGapVolumeShock
from src.strategies.alpha_50_outlier_volume_opening_drive import Alpha50OutlierVolumeOpeningDrive
from src.strategies.alpha_51_trend_aligned_gap_shock import Alpha51TrendAlignedGapShock
from src.strategies.alpha_52_nr5_gap_breakout import Alpha52NR5GapBreakout
from src.strategies.alpha_53_nr7_extreme_gap_expansion import Alpha53NR7ExtremeGapExpansion
from src.strategies.alpha_54_gap_marubozu_momentum import Alpha54GapMarubozuMomentum
from src.strategies.alpha_55_two_day_trend_gap import Alpha55TwoDayTrendGap
from src.strategies.alpha_56_nr4_moderate_gap_shock import Alpha56NR4ModerateGapShock
from src.strategies.alpha_57_nr3_moderate_gap_shock import Alpha57NR3ModerateGapShock
from src.strategies.alpha_58_gap_vwap_momentum import Alpha58GapVWAPMomentum
from src.strategies.alpha_59_nr6_gap_breakout import Alpha59NR6GapBreakout
from src.strategies.alpha_60_ema3_aligned_moderate_gap import Alpha60EMA3AlignedModerateGap
from src.strategies.alpha_61_five_day_hl_moderate_gap import Alpha61FiveDayHLModerateGap
from src.strategies.alpha_62_nr5_moderate_gap_expansion import Alpha62NR5ModerateGapExpansion
from src.strategies.alpha_63_daily_squeeze_gap_expansion import Alpha63DailySqueezeGapExpansion
from src.strategies.alpha_64_opening_range_atr_surge import Alpha64OpeningRangeATRSurge
from src.strategies.alpha_65_nr4_high_volume_gap import Alpha65NR4HighVolumeGap
from src.strategies.alpha_66_two_day_trend_high_vol_gap import Alpha66TwoDayTrendHighVolGap
from src.strategies.alpha_67_ten_day_max_vol_gap import Alpha67TenDayMaxVolGap
from src.strategies.alpha_68_nr5_high_conviction_gap import Alpha68NR5HighConvictionGap
from src.strategies.alpha_69_sub_atr_contraction_gap import Alpha69SubATRContractionGap
from src.strategies.alpha_70_double_inside_target_expansion import Alpha70DoubleInsideTargetExpansion
from src.strategies.alpha_71_outlier_volume_drive import Alpha71OutlierVolumeDrive
from src.strategies.alpha_72_nr3_volatility_expansion import Alpha72NR3VolatilityExpansion
from src.strategies.alpha_73_inside_day_expansion import Alpha73InsideDayExpansion
from src.strategies.alpha_74_gap_marubozu_momentum import Alpha74GapMarubozuMomentum
from src.strategies.alpha_75_morning_drive_continuation import Alpha75MorningDriveContinuation
from src.strategies.alpha_76_two_day_momentum_gap_surge import Alpha76TwoDayMomentumGapSurge
from src.strategies.alpha_77_nr4_shock_expansion import Alpha77NR4ShockExpansion
from src.strategies.alpha_78_double_inside_momentum import Alpha78DoubleInsideMomentum
from src.strategies.alpha_79_high_rr_opening_drive import Alpha79HighRROpeningDrive
from src.strategies.alpha_80_nr7_volatility_expansion import Alpha80NR7VolatilityExpansion
from src.strategies.alpha_81_double_inside_2r_expansion import Alpha81DoubleInside2RExpansion
from src.strategies.alpha_82_double_inside_volume_shock import Alpha82DoubleInsideVolumeShock
from src.strategies.alpha_83_double_inside_gap_drift import Alpha83DoubleInsideGapDrift
from src.strategies.alpha_84_triple_inside_expansion import Alpha84TripleInsideExpansion
from src.strategies.alpha_85_double_inside_225r_expansion import Alpha85DoubleInside225RExpansion
from src.strategies.alpha_86_three_day_trend_surge_2r import Alpha86ThreeDayTrendSurge2R
from src.strategies.alpha_87_two_day_trend_surge_2r import Alpha87TwoDayTrendSurge2R
from src.strategies.alpha_88_three_day_trend_surge_175r import Alpha88ThreeDayTrendSurge175R

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
    "alpha_27": ("ALPHA_27_SECTOR_MOMENTUM_DRIFT", Alpha27SectorMomentumDrift),
    "alpha_28": ("ALPHA_28_VALUE_AREA_EXPANSION", Alpha28ValueAreaExpansion),
    "alpha_29": ("ALPHA_29_TREND_EXHAUSTION_CLIMAX", Alpha29TrendExhaustionClimax),
    "alpha_30": ("ALPHA_30_MIDDAY_SQUEEZE_TREND", Alpha30MiddaySqueezeTrend),
    "alpha_31": ("ALPHA_31_FAILED_OPENING_DRIVE_FADE", Alpha31FailedOpeningDriveFade),
    "alpha_35": ("ALPHA_35_NR4_OPENING_EXPANSION", Alpha35NR4OpeningExpansion),
    "alpha_36": ("ALPHA_36_INSIDE_DAY_EXPANSION", Alpha36InsideDayExpansion),
    "alpha_37": ("ALPHA_37_GAP_VOLUME_SHOCK_DRIFT", Alpha37GapVolumeShockDrift),
    "alpha_38": ("ALPHA_38_THREE_DAY_TREND_EXPANSION", Alpha38ThreeDayTrendExpansion),
    "alpha_39": ("ALPHA_39_OPEN_EQUALS_EXTREME_DRIVE", Alpha39OpenEqualsExtremeDrive),
    "alpha_40": ("ALPHA_40_DOUBLE_INSIDE_EXPANSION", Alpha40DoubleInsideExpansion),
    "alpha_41": ("ALPHA_41_OPENING_MARUBOZU_EXPANSION", Alpha41OpeningMarubozuExpansion),
    "alpha_42": ("ALPHA_42_NR2_GAP_BREAKOUT", Alpha42NR2GapBreakout),
    "alpha_43": ("ALPHA_43_NR3_GAP_BREAKOUT", Alpha43NR3GapBreakout),
    "alpha_44": ("ALPHA_44_CONTRACTION_OPEN_DRIVE", Alpha44ContractionOpenDrive),
    "alpha_45": ("ALPHA_45_INSIDE_DAY_GAP_MOMENTUM", Alpha45InsideDayGapMomentum),
    "alpha_46": ("ALPHA_46_HIGH_CONVICTION_GAP_DRIFT", Alpha46HighConvictionGapDrift),
    "alpha_47": ("ALPHA_47_ORB30_GAP_EXPANSION", Alpha47ORB30GapExpansion),
    "alpha_48": ("ALPHA_48_NR7_GAP_BREAKOUT", Alpha48NR7GapBreakout),
    "alpha_49": ("ALPHA_49_MODERATE_GAP_VOLUME_SHOCK", Alpha49ModerateGapVolumeShock),
    "alpha_50": ("ALPHA_50_OUTLIER_VOLUME_OPENING_DRIVE", Alpha50OutlierVolumeOpeningDrive),
    "alpha_51": ("ALPHA_51_TREND_ALIGNED_GAP_SHOCK", Alpha51TrendAlignedGapShock),
    "alpha_52": ("ALPHA_52_NR5_GAP_BREAKOUT", Alpha52NR5GapBreakout),
    "alpha_53": ("ALPHA_53_NR7_EXTREME_GAP_EXPANSION", Alpha53NR7ExtremeGapExpansion),
    "alpha_54": ("ALPHA_54_GAP_MARUBOZU_MOMENTUM", Alpha54GapMarubozuMomentum),
    "alpha_55": ("ALPHA_55_TWO_DAY_TREND_GAP", Alpha55TwoDayTrendGap),
    "alpha_56": ("ALPHA_56_NR4_MODERATE_GAP_SHOCK", Alpha56NR4ModerateGapShock),
    "alpha_57": ("ALPHA_57_NR3_MODERATE_GAP_SHOCK", Alpha57NR3ModerateGapShock),
    "alpha_58": ("ALPHA_58_GAP_VWAP_MOMENTUM", Alpha58GapVWAPMomentum),
    "alpha_59": ("ALPHA_59_NR6_GAP_BREAKOUT", Alpha59NR6GapBreakout),
    "alpha_60": ("ALPHA_60_EMA3_ALIGNED_MODERATE_GAP", Alpha60EMA3AlignedModerateGap),
    "alpha_61": ("ALPHA_61_FIVE_DAY_HL_MODERATE_GAP", Alpha61FiveDayHLModerateGap),
    "alpha_62": ("ALPHA_62_NR5_MODERATE_GAP_EXPANSION", Alpha62NR5ModerateGapExpansion),
    "alpha_63": ("ALPHA_63_DAILY_SQUEEZE_GAP_EXPANSION", Alpha63DailySqueezeGapExpansion),
    "alpha_64": ("ALPHA_64_OPENING_RANGE_ATR_SURGE", Alpha64OpeningRangeATRSurge),
    "alpha_65": ("ALPHA_65_NR4_HIGH_VOLUME_GAP", Alpha65NR4HighVolumeGap),
    "alpha_66": ("ALPHA_66_TWO_DAY_TREND_HIGH_VOL_GAP", Alpha66TwoDayTrendHighVolGap),
    "alpha_67": ("ALPHA_67_TEN_DAY_MAX_VOL_GAP", Alpha67TenDayMaxVolGap),
    "alpha_68": ("ALPHA_68_NR5_HIGH_CONVICTION_GAP", Alpha68NR5HighConvictionGap),
    "alpha_69": ("ALPHA_69_SUB_ATR_CONTRACTION_GAP", Alpha69SubATRContractionGap),
    "alpha_70": ("ALPHA_70_DOUBLE_INSIDE_TARGET_EXPANSION", Alpha70DoubleInsideTargetExpansion),
    "alpha_71": ("ALPHA_71_OUTLIER_VOLUME_DRIVE", Alpha71OutlierVolumeDrive),
    "alpha_72": ("ALPHA_72_NR3_VOLATILITY_EXPANSION", Alpha72NR3VolatilityExpansion),
    "alpha_73": ("ALPHA_73_INSIDE_DAY_EXPANSION", Alpha73InsideDayExpansion),
    "alpha_74": ("ALPHA_74_GAP_MARUBOZU_MOMENTUM", Alpha74GapMarubozuMomentum),
    "alpha_75": ("ALPHA_75_MORNING_DRIVE_CONTINUATION", Alpha75MorningDriveContinuation),
    "alpha_76": ("ALPHA_76_TWO_DAY_MOMENTUM_GAP_SURGE", Alpha76TwoDayMomentumGapSurge),
    "alpha_77": ("ALPHA_77_NR4_SHOCK_EXPANSION", Alpha77NR4ShockExpansion),
    "alpha_78": ("ALPHA_78_DOUBLE_INSIDE_MOMENTUM", Alpha78DoubleInsideMomentum),
    "alpha_79": ("ALPHA_79_HIGH_RR_OPENING_DRIVE", Alpha79HighRROpeningDrive),
    "alpha_80": ("ALPHA_80_NR7_VOLATILITY_EXPANSION", Alpha80NR7VolatilityExpansion),
    "alpha_81": ("ALPHA_81_DOUBLE_INSIDE_2R_EXPANSION", Alpha81DoubleInside2RExpansion),
    "alpha_82": ("ALPHA_82_DOUBLE_INSIDE_VOLUME_SHOCK", Alpha82DoubleInsideVolumeShock),
    "alpha_83": ("ALPHA_83_DOUBLE_INSIDE_GAP_DRIFT", Alpha83DoubleInsideGapDrift),
    "alpha_84": ("ALPHA_84_TRIPLE_INSIDE_EXPANSION", Alpha84TripleInsideExpansion),
    "alpha_85": ("ALPHA_85_DOUBLE_INSIDE_225R_EXPANSION", Alpha85DoubleInside225RExpansion),
    "alpha_86": ("ALPHA_86_THREE_DAY_TREND_SURGE_2R", Alpha86ThreeDayTrendSurge2R),
    "alpha_87": ("ALPHA_87_TWO_DAY_TREND_SURGE_2R", Alpha87TwoDayTrendSurge2R),
    "alpha_88": ("ALPHA_88_THREE_DAY_TREND_SURGE_175R", Alpha88ThreeDayTrendSurge175R),
}

from scripts.ingest_all_nifty50_timeframes import NIFTY_50_UNIVERSE

DEFAULT_UNIVERSE = NIFTY_50_UNIVERSE


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
