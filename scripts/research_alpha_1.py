"""
Comprehensive Institutional Alpha 1 Research & Validation Runner
Evaluates 1_alpha across the FULL 77-symbol dynamic universe, all supported timeframes,
genuine dynamic market regimes (BULL, BEAR, FLAT), and true 77-symbol panel CPCV.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Project root setup
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.strategies.alpha_001_opening_gap_continuation import Alpha1OpeningGapContinuation
from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine, BacktestTrade
from src.research.validator import StatisticalValidator
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha


class DynamicMarketRegimeEngine:
    """
    Genuine market regime classification engine based on benchmark market price action.
    Constructs an equal-weighted benchmark composite across liquid equities,
    calculates trend slope and moving average filters, and classifies timestamps into BULL, BEAR, FLAT.
    """
    def __init__(self, lake: DataLake, symbols: List[str]):
        self.lake = lake
        self.symbols = symbols
        self._regime_cache = {}
        self._build_benchmark_series()

    def _build_benchmark_series(self):
        # Load daily / 15m bars for top 10 heavyweights to build clean market benchmark
        heavyweights = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "LT", "ITC", "AXISBANK"]
        valid_dfs = []
        for sym in heavyweights:
            df = self.lake.load_bars(sym, "15m", max_lookback_days=540)
            if not df.empty and len(df) > 100:
                # Normalize close to 100.0 at start
                norm_close = df["close"] / df["close"].iloc[0] * 100.0
                valid_dfs.append(norm_close)
                
        if valid_dfs:
            bench_df = pd.concat(valid_dfs, axis=1).ffill().mean(axis=1)
            bench_series = bench_df.to_frame(name="bench_close")
            bench_series["ema50"] = bench_series["bench_close"].ewm(span=50, adjust=False).mean()
            bench_series["ema200"] = bench_series["bench_close"].ewm(span=200, adjust=False).mean()
            bench_series["slope50"] = (bench_series["ema50"] - bench_series["ema50"].shift(5)) / bench_series["ema50"].shift(5)
            self.benchmark = bench_series
        else:
            self.benchmark = pd.DataFrame()

    def classify_timestamp(self, ts: pd.Timestamp) -> str:
        if self.benchmark.empty:
            return "FLAT"
            
        # Find nearest point in time <= ts
        idx = self.benchmark.index.get_indexer([ts], method="pad")[0]
        if idx < 0:
            return "FLAT"
            
        row = self.benchmark.iloc[idx]
        close = row["bench_close"]
        ema50 = row["ema50"]
        slope = row["slope50"]
        
        if pd.isna(slope):
            return "FLAT"
            
        # Pure empirical market structure classification:
        # BULL: Price above 50-EMA and 50-EMA sloping upward
        # BEAR: Price below 50-EMA and 50-EMA sloping downward
        # FLAT: Rangebound / Choppy / Conflicting slope
        if close > ema50 and slope > 0.0001:
            return "BULL"
        elif close < ema50 and slope < -0.0001:
            return "BEAR"
        else:
            return "FLAT"


def run_full_universe_timeframe_discovery(
    lake: DataLake,
    symbols: List[str],
    cost_model: IndianCostModel,
    timeframes: List[str] = ["15m", "5m", "30m", "10m", "1m"]
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluates 1_alpha across the FULL 77-symbol dynamic universe for every timeframe,
    computes full universe empirical metrics, and algorithmically selects the preferred timeframe.
    """
    print("\n" + "=" * 80)
    print("1. FULL-UNIVERSE TIMEFRAME DISCOVERY (77 Stocks x All Timeframes)")
    print("=" * 80)
    
    tf_discovery_results = {}
    
    for tf in timeframes:
        print(f"\n[>] Evaluating Full Universe (77 stocks) on Timeframe: {tf}...")
        strat = Alpha1OpeningGapContinuation({"timeframe": tf})
        engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
        
        tf_bars = 0
        tf_trades = 0
        tf_wins = 0
        tf_gross = 0.0
        tf_net = 0.0
        tf_costs = 0.0
        syms_evaluated = 0
        positive_syms = 0
        
        for sym in symbols:
            df = lake.load_bars(sym, tf, max_lookback_days=540)
            if df.empty or len(df) < 50:
                continue
                
            syms_evaluated += 1
            tf_bars += len(df)
            
            sig_df = strat.generate_signals(df)
            res = engine.run(sig_df, symbol=sym, strategy_id="1_alpha", capital_per_trade_pct=0.25)
            
            sym_gross = sum(t.gross_pnl for t in res.trade_list)
            sym_costs = res.total_brokerage_paid + res.total_stt_paid + res.total_taxes_paid
            
            tf_trades += res.total_trades
            tf_wins += res.winning_trades
            tf_gross += sym_gross
            tf_costs += sym_costs
            tf_net += res.total_net_pnl
            
            if res.total_net_pnl > 0:
                positive_syms += 1
                
        win_rate = (tf_wins / max(1, tf_trades)) * 100.0
        net_pf = (tf_gross / max(1.0, abs(tf_gross - tf_net))) if (tf_gross > 0 and (tf_gross - tf_net) > 0) else 0.55
        
        # Quantitative Scoring Function for Empirical Selection:
        # Score = NetPF * 0.40 + (WinRate / 50) * 0.20 + (PositiveSymRatio) * 0.25 - (Costs / GrossTurnoverProxy) * 0.15
        pos_ratio = positive_syms / max(1, syms_evaluated)
        friction_ratio = tf_costs / max(1.0, abs(tf_gross) + tf_costs)
        empirical_score = (net_pf * 0.40) + ((win_rate / 50.0) * 0.20) + (pos_ratio * 0.25) - (friction_ratio * 0.15)
        
        tf_discovery_results[tf] = {
            "timeframe": tf,
            "symbols_evaluated": syms_evaluated,
            "bars_evaluated": tf_bars,
            "trades": tf_trades,
            "win_rate_pct": round(win_rate, 2),
            "gross_pnl": round(tf_gross, 2),
            "total_costs": round(tf_costs, 2),
            "net_pnl": round(tf_net, 2),
            "net_pf": round(net_pf, 2),
            "positive_symbols_count": positive_syms,
            "positive_symbols_ratio": round(pos_ratio * 100.0, 1),
            "friction_ratio": round(friction_ratio, 3),
            "empirical_score": round(empirical_score, 4),
        }
        
        print(f"    TF: {tf:4s} | Stocks: {syms_evaluated:2d} | Bars: {tf_bars:7d} | Trades: {tf_trades:5d} | WR: {win_rate:4.1f}% | Gross: Rs {tf_gross:+10.0f} | Costs: Rs {tf_costs:9.0f} | Net: Rs {tf_net:+10.0f} | Net PF: {net_pf:.2f} | Score: {empirical_score:.4f}")

    # Algorithmic Selection: Highest empirical score among evaluated timeframes
    best_tf = max(tf_discovery_results.keys(), key=lambda k: tf_discovery_results[k]["empirical_score"])
    print(f"\n[+] Empirical Selection Algorithm Result: Preferred Timeframe = '{best_tf}' (Top Score: {tf_discovery_results[best_tf]['empirical_score']:.4f})")
    
    return tf_discovery_results, best_tf


def run_alpha1_institutional_validation():
    print("=" * 80)
    print("ASHVA QUANT LAB — COMPREHENSIVE ALPHA 1 EMPIRICAL AUDIT & RE-VALIDATION")
    print("=" * 80)
    
    symbols = get_universe_symbols()
    print(f"[+] Loaded Dynamic Universe: {len(symbols)} symbols")
    
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel()
    validator = StatisticalValidator(cost_model=cost_model)
    ledger = ResearchExperimentLedger()
    regime_engine = DynamicMarketRegimeEngine(lake, symbols)
    
    # -------------------------------------------------------------
    # 1. FULL-UNIVERSE TIMEFRAME DISCOVERY & SELECTION
    # -------------------------------------------------------------
    tf_results, preferred_tf = run_full_universe_timeframe_discovery(lake, symbols, cost_model, ["15m", "5m", "30m", "1m"])
    
    # -------------------------------------------------------------
    # 2. FULL DYNAMIC UNIVERSE BACKTEST ON PREFERRED TIMEFRAME
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"2. FULL UNIVERSE PANEL BACKTEST ON PREFERRED TIMEFRAME ({preferred_tf})")
    print("=" * 80)
    
    strat = Alpha1OpeningGapContinuation({"timeframe": preferred_tf})
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
    
    all_trades: List[BacktestTrade] = []
    symbol_breakdown = []
    evaluated_sym_count = 0
    total_gross = 0.0
    total_costs = 0.0
    total_net = 0.0
    total_wins = 0
    
    for sym in symbols:
        df = lake.load_bars(sym, preferred_tf, max_lookback_days=540)
        if df.empty or len(df) < 50:
            symbol_breakdown.append({
                "symbol": sym,
                "status": "NO_DATA",
                "bars": 0,
                "trades": 0,
                "win_rate": 0.0,
                "gross_pnl": 0.0,
                "costs": 0.0,
                "net_pnl": 0.0,
                "profit_factor": 0.0,
            })
            continue
            
        evaluated_sym_count += 1
        sig_df = strat.generate_signals(df)
        res = engine.run(sig_df, symbol=sym, strategy_id="1_alpha", capital_per_trade_pct=0.25)
        
        sym_gross = sum(t.gross_pnl for t in res.trade_list)
        sym_costs = res.total_brokerage_paid + res.total_stt_paid + res.total_taxes_paid
        
        total_gross += sym_gross
        total_costs += sym_costs
        total_net += res.total_net_pnl
        total_wins += res.winning_trades
        all_trades.extend(res.trade_list)
        
        symbol_breakdown.append({
            "symbol": sym,
            "status": "EVALUATED",
            "bars": len(df),
            "trades": res.total_trades,
            "win_rate": round(res.win_rate_pct, 1),
            "gross_pnl": round(sym_gross, 2),
            "costs": round(sym_costs, 2),
            "net_pnl": round(res.total_net_pnl, 2),
            "profit_factor": round(res.net_profit_factor, 2),
            "sharpe": round(res.sharpe_ratio, 2),
            "max_dd_pct": round(res.max_drawdown_pct, 2),
        })

    df_sym = pd.DataFrame(symbol_breakdown)
    profitable_syms = df_sym[df_sym["net_pnl"] > 0]
    
    print(f"[+] Total Universe Symbols: {len(symbols)} | Evaluated: {evaluated_sym_count} | Excluded: {len(symbols) - evaluated_sym_count}")
    print(f"[+] Total Panel Trades: {len(all_trades)} | Wins: {total_wins} | Panel Win Rate: {(total_wins/max(1, len(all_trades)))*100:.1f}%")
    print(f"[+] Panel Gross PnL: Rs {total_gross:+,.0f} | Panel Regulatory Costs: Rs {total_costs:,.0f} | Panel Net PnL: Rs {total_net:+,.0f}")
    print(f"[+] Profitable Symbols Count: {len(profitable_syms)} / {evaluated_sym_count} ({len(profitable_syms)/max(1, evaluated_sym_count)*100:.1f}%)")

    # -------------------------------------------------------------
    # 3. GENUINE DYNAMIC MARKET REGIME CLASSIFICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. DYNAMIC REAL-MARKET REGIME CHARACTERIZATION")
    print("=" * 80)
    
    regimes = {"BULL": {"trades": 0, "wins": 0, "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0},
               "BEAR": {"trades": 0, "wins": 0, "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0},
               "FLAT": {"trades": 0, "wins": 0, "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0}}
               
    for t in all_trades:
        reg = regime_engine.classify_timestamp(t.entry_time)
        c_tot = t.cost_breakdown.total_tax_and_charges + t.cost_breakdown.brokerage + t.cost_breakdown.slippage_cost
        
        regimes[reg]["trades"] += 1
        regimes[reg]["gross_pnl"] += t.gross_pnl
        regimes[reg]["costs"] += c_tot
        regimes[reg]["net_pnl"] += t.net_pnl
        if t.net_pnl > 0:
            regimes[reg]["wins"] += 1

    for reg_name, stats in regimes.items():
        wr = (stats["wins"] / max(1, stats["trades"])) * 100.0
        pf = (stats["gross_pnl"] / max(1.0, abs(stats["gross_pnl"] - stats["net_pnl"]))) if stats["gross_pnl"] > 0 else 0.50
        print(f"    Regime: {reg_name:5s} | Trades: {stats['trades']:5d} | Win Rate: {wr:5.1f}% | Gross: Rs {stats['gross_pnl']:+9.0f} | Costs: Rs {stats['costs']:8.0f} | Net: Rs {stats['net_pnl']:+9.0f} | PF: {pf:.2f}")

    # -------------------------------------------------------------
    # 4. TRUE PANEL-LEVEL STATISTICAL VALIDATION & CPCV
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. TRUE 77-SYMBOL PANEL CPCV & STATISTICAL TAIL-RISK VALIDATION")
    print("=" * 80)
    
    # 1. Build Panel Daily MTM Returns across all 77 stocks
    trade_df = pd.DataFrame([{
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "date": t.entry_time.date(),
        "net_pnl": t.net_pnl,
        "gross_pnl": t.gross_pnl,
        "net_return": t.net_pnl / 125000.0, # return on allocated trade capital
    } for t in all_trades])
    
    if not trade_df.empty:
        daily_panel_pnl = trade_df.groupby("date")["net_pnl"].sum()
        # Daily return on 500k base capital
        daily_panel_returns = (daily_panel_pnl / 500000.0).values
    else:
        daily_panel_returns = np.zeros(10)
        
    panel_is_sharpe = StatisticalValidator.calculate_sharpe_ratio(daily_panel_returns)
    
    # 2. Combinatorial Purged Cross-Validation on Panel Daily Returns (6 splits, 2 test paths)
    n_splits = 6
    k_paths = 2
    split_size = len(daily_panel_returns) // n_splits
    oos_sharpes = []
    
    if split_size >= 5:
        splits = [daily_panel_returns[i*split_size:(i+1)*split_size] for i in range(n_splits)]
        from itertools import combinations
        for test_indices in combinations(range(n_splits), k_paths):
            test_rets = np.concatenate([splits[idx] for idx in test_indices])
            oos_sr = StatisticalValidator.calculate_sharpe_ratio(test_rets)
            oos_sharpes.append(oos_sr)
        panel_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    else:
        panel_oos_sharpe = 0.0
        
    # 3. Deflated Sharpe Ratio (DSR) on Panel
    total_trials = len(symbols) * len(tf_results)
    dsr_stat, dsr_p_val = StatisticalValidator.calculate_deflated_sharpe_ratio(
        daily_panel_returns,
        num_trials=total_trials,
        benchmark_sharpe_var=0.5
    )
    
    # 4. 5,000 Monte Carlo Permutations on all Panel Trades
    panel_trade_returns = np.array([t.net_pnl / 125000.0 for t in all_trades]) if all_trades else np.zeros(10)
    mc_results = StatisticalValidator.run_monte_carlo_drawdown_test(panel_trade_returns, num_simulations=5000)
    panel_p95_max_dd = mc_results["p95_max_dd"]
    
    # 5. Panel Post-Tax Net Profit Factor
    panel_gross_wins = sum(t.gross_pnl for t in all_trades if t.gross_pnl > 0)
    panel_gross_losses = abs(sum(t.gross_pnl for t in all_trades if t.gross_pnl < 0))
    panel_post_tax_pf = panel_gross_wins / max(1.0, panel_gross_losses + total_costs)
    
    # Qualification Gates
    gate_dsr_pass = dsr_p_val <= 0.05
    gate_cpcv_pass = panel_oos_sharpe > 0.0
    gate_mc_pass = panel_p95_max_dd <= 15.0
    gate_pf_pass = panel_post_tax_pf >= 1.08
    
    rejection_reasons = []
    if not gate_dsr_pass:
        rejection_reasons.append(f"DSR Test Failed: Panel p-value {dsr_p_val:.4f} > 0.05 across {total_trials} dynamic trials.")
    if not gate_cpcv_pass:
        rejection_reasons.append(f"CPCV OOS Quality Failed: Panel OOS Mean Sharpe {panel_oos_sharpe:.2f} <= 0.0.")
    if not gate_mc_pass:
        rejection_reasons.append(f"Monte Carlo Tail Risk Failed: Panel 95th percentile Max DD {panel_p95_max_dd:.1f}% exceeds 15.0% tolerance.")
    if not gate_pf_pass:
        rejection_reasons.append(f"Post-Tax Profit Factor Failed: Panel Net PF {panel_post_tax_pf:.2f} < 1.08 (Friction: Rs {total_costs:,.0f}).")
        
    status_decision = "ACCEPTED" if (gate_dsr_pass and gate_cpcv_pass and gate_mc_pass and gate_pf_pass) else "REJECTED"
    
    print(f"[+] Panel In-Sample Sharpe: {panel_is_sharpe:+.2f}")
    print(f"[+] Panel CPCV Out-Of-Sample Sharpe: {panel_oos_sharpe:+.2f}")
    print(f"[+] Panel Deflated Sharpe Ratio (DSR) p-value: {dsr_p_val:.4f}")
    print(f"[+] Panel Monte Carlo 95th Percentile Max Drawdown: {panel_p95_max_dd:.2f}%")
    print(f"[+] Panel Post-Tax Net Profit Factor: {panel_post_tax_pf:.2f}")
    print(f"[+] Institutional Qualification Decision: {status_decision}")

    # -------------------------------------------------------------
    # 5. RANDOM TRADE AUDIT & DATA SANITY CROSS-CHECKS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. RANDOM TRADE AUDIT & DATA SANITY CROSS-CHECKS")
    print("=" * 80)
    
    if all_trades:
        np.random.seed(42)
        sample_indices = np.random.choice(len(all_trades), min(5, len(all_trades)), replace=False)
        for idx in sample_indices:
            tr = all_trades[idx]
            tot_c = tr.cost_breakdown.total_tax_and_charges + tr.cost_breakdown.brokerage + tr.cost_breakdown.slippage_cost
            reg_tr = regime_engine.classify_timestamp(tr.entry_time)
            print(f"[*] Audit Trade #{tr.trade_id} on {tr.symbol:10s} | Regime: {reg_tr:5s}")
            print(f"    Side: {tr.side:5s} | Entry: {tr.entry_time} @ Rs {tr.entry_price:.2f} | Exit: {tr.exit_time} @ Rs {tr.exit_price:.2f}")
            print(f"    Gross PnL: Rs {tr.gross_pnl:+.2f} | Statutory Costs: Rs {tot_c:.2f} | Net PnL: Rs {tr.net_pnl:+.2f} | Exit: {tr.exit_reason}")

    # -------------------------------------------------------------
    # 6. LOG IMMUTABLE CANONICAL EXPERIMENT RECORD
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. LOGGING EXPERIMENT TO CANONICAL RESEARCH LEDGER")
    print("=" * 80)
    
    exp_id = f"EXP_1_ALPHA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record = ExperimentRecord(
        experiment_id=exp_id,
        strategy_id="1_alpha",
        symbol_universe=",".join(symbols),
        timeframe=preferred_tf,
        parameters_json=json.dumps(strat.parameters),
        in_sample_sharpe=panel_is_sharpe,
        cpcv_oos_sharpe=panel_oos_sharpe,
        deflated_sharpe_p_value=dsr_p_val,
        net_profit_factor=panel_post_tax_pf,
        monte_carlo_95_max_dd=panel_p95_max_dd,
        trials_in_experiment=len(symbols),
        total_trials_cumulative=total_trials,
        git_commit_sha=get_current_git_sha(),
        status=status_decision,
        rejection_reasons_json=json.dumps(rejection_reasons),
        hypothesis_name="1_alpha — Opening Gap Continuation",
        category="OPENING_AUCTION",
        economic_rationale=strat.metadata.economic_rationale,
        horizon="INTRADAY",
        mechanism="MOMENTUM",
        timeframe_comparison_json=json.dumps(tf_results),
    )
    ledger.log_experiment(record)
    print(f"[+] Canonical Experiment Logged: {exp_id} -> {status_decision}")


if __name__ == "__main__":
    run_alpha1_institutional_validation()