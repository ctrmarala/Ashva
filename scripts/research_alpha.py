"""
Universal Ashva Quantitative Alpha Research & Institutional Validation Runner
Single Authoritative CLI Entrypoint for evaluating ANY alpha strategy against:
- Full 77-Symbol Dynamic Universe
- Full-Universe Multi-Timeframe Discovery with Normalized Empirical Scoring
- Dynamic Real-Market Regime Engine (BULL, BEAR, FLAT)
- Trade-Level, Symbol-Level, and Daily Panel Trade-Return Series Evidence
- Canonical StatisticalValidator (CPCVEngine with purging/embargoing, Bailey-Lopez de Prado DSR, 5,000-Run Monte Carlo)
- Exact Indian Regulatory Cash Taxes (STT, GST, SEBI, Rs 20 Brokerage, Slippage)
- Random Trade Sanity Audits
- Immutable Persistence in SQLite Experiment Ledger
"""

import os
import sys
import json
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.strategies.registry import get_all_strategies, get_strategy_by_name
from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.analytics.metrics import calculate_profit_factor, calculate_trade_level_metrics
from src.backtest.engine import BacktestEngine, BacktestTrade
from src.research.validator import StatisticalValidator, PanelResearchResult
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha
from src.research.alpha_linter import AlphaLinter, AlphaLinterError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AlphaResearchRunner")


class DynamicMarketRegimeEngine:
    """
    Classifies timestamps into BULL, BEAR, or FLAT based on point-in-time benchmark market price structure.
    Constructs an equal-weighted benchmark from top liquid equities and computes EMA50 and trend slope.
    Strictly point-in-time with zero lookahead.
    """
    def __init__(self, lake: DataLake):
        self.lake = lake
        self.benchmark = self._build_benchmark()

    def _build_benchmark(self) -> pd.DataFrame:
        heavyweights = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "LT", "ITC", "AXISBANK"]
        valid_dfs = []
        for sym in heavyweights:
            df = self.lake.load_bars(sym, "15m", max_lookback_days=540)
            if not df.empty and len(df) > 100:
                norm_close = df["close"] / df["close"].iloc[0] * 100.0
                valid_dfs.append(norm_close)

        if valid_dfs:
            bench_df = pd.concat(valid_dfs, axis=1).ffill().mean(axis=1).to_frame(name="bench_close")
            bench_df["ema50"] = bench_df["bench_close"].ewm(span=50, adjust=False).mean()
            bench_df["slope50"] = (bench_df["ema50"] - bench_df["ema50"].shift(5)) / bench_df["ema50"].shift(5)
            return bench_df
        return pd.DataFrame()

    def classify_timestamp(self, ts: pd.Timestamp) -> str:
        if self.benchmark.empty:
            return "FLAT"
        idx = self.benchmark.index.get_indexer([ts], method="pad")[0]
        if idx < 0:
            return "FLAT"
        row = self.benchmark.iloc[idx]
        close = row["bench_close"]
        ema50 = row["ema50"]
        slope = row["slope50"]
        if pd.isna(slope):
            return "FLAT"
        if close > ema50 and slope > 0.0001:
            return "BULL"
        elif close < ema50 and slope < -0.0001:
            return "BEAR"
        else:
            return "FLAT"


def run_full_timeframe_discovery(
    strat_cls: Any,
    lake: DataLake,
    symbols: List[str],
    cost_model: IndianCostModel,
    timeframes: List[str] = ["30m", "15m", "5m", "1m"]
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluates strategy across all symbols for all candidate timeframes.
    Scores timeframes using normalized empirical scoring and selects the preferred timeframe.
    """
    print("\n" + "=" * 80)
    print("STEP 2: FULL-UNIVERSE TIMEFRAME DISCOVERY (77 Stocks x Multi-Timeframe)")
    print("=" * 80)

    tf_results = {}
    for tf in timeframes:
        print(f"\n[>] Backtesting Full Universe ({len(symbols)} stocks) on Candidate Timeframe: {tf}...")
        strat = strat_cls({"timeframe": tf})
        engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY, use_1m_intrabar=False)

        tf_bars = 0
        tf_trades = 0
        tf_wins = 0
        tf_gross = 0.0
        tf_costs = 0.0
        tf_net = 0.0
        syms_evaluated = 0
        positive_syms = 0
        all_tf_net_pnls = []

        for sym in symbols:
            df = lake.load_bars(sym, tf, max_lookback_days=540)
            if df.empty or len(df) < 50:
                continue

            syms_evaluated += 1
            tf_bars += len(df)

            sig_df = strat.generate_signals(df)
            res = engine.run(sig_df, symbol=sym, strategy_id=strat.strategy_id, capital_per_trade_pct=0.25)

            sym_gross = sum(t.gross_pnl for t in res.trade_list)
            sym_costs = res.total_brokerage_paid + res.total_stt_paid + res.total_taxes_paid

            tf_trades += res.total_trades
            tf_wins += res.winning_trades
            tf_gross += sym_gross
            tf_costs += sym_costs
            tf_net += res.total_net_pnl

            for t in res.trade_list:
                all_tf_net_pnls.append(t.net_pnl)

            if res.total_net_pnl > 0:
                positive_syms += 1

        win_rate = (tf_wins / max(1, tf_trades)) * 100.0
        # Canonical Profit Factor calculation: sum(wins) / abs(sum(losses))
        net_pf = calculate_profit_factor(all_tf_net_pnls)
        pos_ratio = positive_syms / max(1, syms_evaluated)
        friction_ratio = tf_costs / max(1.0, abs(tf_gross) + tf_costs)

        # Normalized empirical score:
        # NormPF (0 to 1, capped at PF=2.0) * 0.35 + WinRate (0 to 1) * 0.25 + Breadth (0 to 1) * 0.25 - Friction (0 to 1) * 0.15
        norm_pf = min(1.0, max(0.0, net_pf / 2.0))
        norm_wr = min(1.0, max(0.0, win_rate / 100.0))
        empirical_score = (norm_pf * 0.35) + (norm_wr * 0.25) + (pos_ratio * 0.25) - (friction_ratio * 0.15)

        tf_results[tf] = {
            "timeframe": tf,
            "symbols_evaluated": syms_evaluated,
            "bars_evaluated": tf_bars,
            "trades": tf_trades,
            "win_rate_pct": round(win_rate, 2),
            "gross_pnl": round(tf_gross, 2),
            "total_costs": round(tf_costs, 2),
            "net_pnl": round(tf_net, 2),
            "net_profit_factor": round(net_pf, 2),
            "positive_symbols_count": positive_syms,
            "positive_symbols_ratio": round(pos_ratio * 100.0, 1),
            "friction_ratio": round(friction_ratio, 3),
            "empirical_timeframe_score": round(empirical_score, 4),
        }

        print(f"    TF: {tf:4s} | Stocks: {syms_evaluated:2d} | Bars: {tf_bars:7d} | Trades: {tf_trades:5d} | WR: {win_rate:4.1f}% | Gross: Rs {tf_gross:+10.0f} | Costs: Rs {tf_costs:9.0f} | Net: Rs {tf_net:+10.0f} | Net PF: {net_pf:.2f} | Score: {empirical_score:.4f}")

        # Early Stopping Heuristic: If higher timeframes fail badly, skip lower resolution
        if tf in ["15m", "30m"]:
            if empirical_score <= 0.05 and win_rate < 35.0:
                print(f"[!] Early Stopping Triggered: '{tf}' performed poorly. Pruning lower timeframes to save compute.")
                break

    best_tf = max(tf_results.keys(), key=lambda k: tf_results[k]["empirical_timeframe_score"])
    print(f"\n[+] Empirical Selection Algorithm Result: Preferred Timeframe = '{best_tf}' (Score: {tf_results[best_tf]['empirical_timeframe_score']:.4f})")
    return tf_results, best_tf


def research_single_alpha(strat_id: str, lake: DataLake, symbols: List[str], cost_model: IndianCostModel, ledger: ResearchExperimentLedger, regime_engine: DynamicMarketRegimeEngine, validator: StatisticalValidator):
    print("\n" + "=" * 80)
    print(f"ASHVA RESEARCH LAB — EVALUATING ALPHA: {strat_id}")
    print("=" * 80)

    # 1. Strategy Discovery & Pre-Flight Linting
    strat_cls = get_strategy_by_name(strat_id)
    if strat_cls is None:
        raise ValueError(f"Strategy '{strat_id}' not found in registry.")

    strat_inst = strat_cls()
    print(f"[+] Loaded Strategy: {strat_inst.name} ({strat_inst.strategy_id})")

    print("\n" + "=" * 80)
    print("STEP 1: PRE-FLIGHT ALPHA CONTRACT LINTER")
    print("=" * 80)
    violations = AlphaLinter.lint_strategy_instance(strat_inst)
    if violations:
        print("[FAIL] LINTER FAILED: Strategy violated institutional principles:")
        for v in violations:
            print(f"   - {v}")
        raise AlphaLinterError("Strategy failed pre-flight linter check.")
    print("[PASS] PRE-FLIGHT PASSED: Zero lookahead, dynamic universe binding, and parameter grid verified.")

    # 2. Full Universe Timeframe Discovery
    candidate_timeframes = ["30m", "15m", "5m", "1m"]
    tf_results, preferred_tf = run_full_timeframe_discovery(strat_cls, lake, symbols, cost_model, candidate_timeframes)

    # 3. Full 77-Stock Panel Backtest on Preferred Timeframe
    print("\n" + "=" * 80)
    print(f"STEP 3: FULL UNIVERSE PANEL BACKTEST ({preferred_tf})")
    print("=" * 80)

    strat_pref = strat_cls({"timeframe": preferred_tf})
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)

    all_trades: List[BacktestTrade] = []
    symbol_breakdown = []
    total_gross = 0.0
    total_costs = 0.0
    total_net = 0.0
    total_wins = 0
    evaluated_count = 0

    for sym in symbols:
        df = lake.load_bars(sym, preferred_tf, max_lookback_days=540)
        if df.empty or len(df) < 50:
            continue

        evaluated_count += 1
        sig_df = strat_pref.generate_signals(df)
        res = engine.run(sig_df, symbol=sym, strategy_id=strat_id, capital_per_trade_pct=0.25)

        sym_gross = sum(t.gross_pnl for t in res.trade_list)
        sym_costs = res.total_brokerage_paid + res.total_stt_paid + res.total_taxes_paid
        sym_net_pnls = [t.net_pnl for t in res.trade_list]
        sym_pf = calculate_profit_factor(sym_net_pnls)

        total_gross += sym_gross
        total_costs += sym_costs
        total_net += res.total_net_pnl
        total_wins += res.winning_trades
        all_trades.extend(res.trade_list)

        symbol_breakdown.append({
            "symbol": sym,
            "bars": len(df),
            "trades": res.total_trades,
            "win_rate": round(res.win_rate_pct, 1),
            "gross_pnl": round(sym_gross, 2),
            "costs": round(sym_costs, 2),
            "net_pnl": round(res.total_net_pnl, 2),
            "profit_factor": round(sym_pf, 2),
        })

    df_sym = pd.DataFrame(symbol_breakdown)
    profitable_count = len(df_sym[df_sym["net_pnl"] > 0]) if not df_sym.empty else 0
    panel_win_rate = (total_wins / max(1, len(all_trades))) * 100.0
    panel_net_pf = calculate_profit_factor([t.net_pnl for t in all_trades])

    print(f"[+] Total Stocks Evaluated: {evaluated_count} / {len(symbols)}")
    print(f"[+] Total Panel Trades: {len(all_trades):,d} | Panel Win Rate: {panel_win_rate:.1f}% | Panel Net PF: {panel_net_pf:.2f}")
    print(f"[+] Total Gross PnL: Rs {total_gross:+,.0f} | Total Statutory Costs: Rs {total_costs:,.0f} | Total Net PnL: Rs {total_net:+,.0f}")
    print(f"[+] Profitable Symbols Count: {profitable_count} / {evaluated_count} ({(profitable_count/max(1, evaluated_count))*100:.1f}%)")

    # 4. Dynamic Real-Market Regime Engine (BULL, BEAR, FLAT)
    print("\n" + "=" * 80)
    print("STEP 4: DYNAMIC REAL-MARKET REGIME CHARACTERIZATION")
    print("=" * 80)

    regimes = {
        "BULL": {"trades": 0, "wins": 0, "gross": 0.0, "costs": 0.0, "net": 0.0, "pnls": []},
        "BEAR": {"trades": 0, "wins": 0, "gross": 0.0, "costs": 0.0, "net": 0.0, "pnls": []},
        "FLAT": {"trades": 0, "wins": 0, "gross": 0.0, "costs": 0.0, "net": 0.0, "pnls": []}
    }

    for t in all_trades:
        reg = regime_engine.classify_timestamp(t.entry_time)
        c_tot = t.cost_breakdown.total_tax_and_charges + t.cost_breakdown.brokerage + t.cost_breakdown.slippage_cost
        regimes[reg]["trades"] += 1
        regimes[reg]["gross"] += t.gross_pnl
        regimes[reg]["costs"] += c_tot
        regimes[reg]["net"] += t.net_pnl
        regimes[reg]["pnls"].append(t.net_pnl)
        if t.net_pnl > 0:
            regimes[reg]["wins"] += 1

    for reg_name, stats in regimes.items():
        wr = (stats["wins"] / max(1, stats["trades"])) * 100.0
        pf = calculate_profit_factor(stats["pnls"])
        print(f"    Regime: {reg_name:5s} | Trades: {stats['trades']:5d} | Win Rate: {wr:5.1f}% | Gross: Rs {stats['gross']:+9.0f} | Costs: Rs {stats['costs']:8.0f} | Net: Rs {stats['net']:+9.0f} | Net PF: {pf:.2f}")

    # 5. Canonical Statistical Validation via StatisticalValidator.validate_panel
    print("\n" + "=" * 80)
    print("STEP 5: TRUE 77-STOCK PANEL CPCV & STATISTICAL QUALIFICATION")
    print("=" * 80)

    panel_evidence = PanelResearchResult(
        hypothesis=strat_pref,
        all_trades=all_trades,
        symbol_metrics=symbol_breakdown,
        timeframe_comparison=tf_results,
        parameter_grid=strat_inst.get_parameter_grid() if hasattr(strat_inst, "get_parameter_grid") else {},
        tested_timeframes_count=len(candidate_timeframes),
        initial_capital=500000.0,
        regime_breakdown={k: {k2: v2 for k2, v2 in v.items() if k2 != "pnls"} for k, v in regimes.items()},
        selected_timeframe=preferred_tf,
        symbol_universe=symbols,
    )

    report = validator.validate_panel(panel_evidence)

    print(f"[+] Panel In-Sample Sharpe: {report.in_sample_sharpe:+.2f}")
    print(f"[+] Panel CPCV Out-Of-Sample Sharpe: {report.cpcv_mean_sharpe:+.2f}")
    print(f"[+] Panel Deflated Sharpe Ratio (DSR) p-value: {report.deflated_sharpe_p_value:.4f}")
    print(f"[+] Panel Monte Carlo 95th Percentile Max Drawdown: {report.monte_carlo_95_max_dd_pct:.2f}%")
    print(f"[+] Panel Post-Tax Net Profit Factor: {report.net_profit_factor_post_tax:.2f}")
    print(f"[+] Institutional Qualification Decision: {report.status.value}")

    if report.rejection_reasons:
        print("[-] Rejection Reasons / Gate Failures:")
        for r in report.rejection_reasons:
            print(f"    - {r}")

    # 6. Random Trade Audit
    print("\n" + "=" * 80)
    print("STEP 6: RANDOM TRADE AUDIT & DATA SANITY CROSS-CHECKS")
    print("=" * 80)
    if all_trades:
        np.random.seed(42)
        sample_indices = np.random.choice(len(all_trades), min(5, len(all_trades)), replace=False)
        for idx in sample_indices:
            tr = all_trades[idx]
            tot_c = tr.cost_breakdown.total_tax_and_charges + tr.cost_breakdown.brokerage + tr.cost_breakdown.slippage_cost
            reg_tr = regime_engine.classify_timestamp(tr.entry_time)
            print(f"[*] Trade #{tr.trade_id} on {tr.symbol:10s} | Regime: {reg_tr:5s}")
            print(f"    Side: {tr.side:5s} | Entry: {tr.entry_time} @ Rs {tr.entry_price:.2f} | Exit: {tr.exit_time} @ Rs {tr.exit_price:.2f}")
            print(f"    Gross: Rs {tr.gross_pnl:+.2f} | Statutory Costs: Rs {tot_c:.2f} | Net: Rs {tr.net_pnl:+.2f} | Exit: {tr.exit_reason}")

    return report.status.value


def main():
    parser = argparse.ArgumentParser(description="Universal Ashva Alpha Research Runner")
    parser.add_argument("--alpha-id", type=str, default="1_alpha", help="Strategy ID to evaluate (e.g. 1_alpha)")
    parser.add_argument("--all", action="store_true", help="Evaluate all registered strategies")
    parser.add_argument("--untested-only", action="store_true", help="Evaluate only untested registered strategies")
    args = parser.parse_args()

    lake = DataLake(read_only=True)
    symbols = get_universe_symbols()
    cost_model = IndianCostModel()
    ledger = ResearchExperimentLedger()
    validator = StatisticalValidator(cost_model=cost_model, experiment_ledger=ledger)
    regime_engine = DynamicMarketRegimeEngine(lake)

    all_strategies = get_all_strategies(reload=True)
    print(f"[+] Found {len(all_strategies)} strategies in registry: {list(all_strategies.keys())}")

    targets = []
    if args.all:
        targets = list(all_strategies.keys())
    elif args.untested_only:
        tested_ids = set()
        with sqlite3.connect("data_lake/experiment_ledger.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT strategy_id FROM experiment_journal")
            tested_ids = {r[0].lower() for r in cursor.fetchall()}
        targets = [s for s in all_strategies.keys() if s.lower() not in tested_ids]
    else:
        targets = [args.alpha_id]

    print(f"[+] Target Strategies to Evaluate: {targets}")
    for t_id in targets:
        research_single_alpha(t_id, lake, symbols, cost_model, ledger, regime_engine, validator)


if __name__ == "__main__":
    main()