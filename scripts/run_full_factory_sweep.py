"""
Ashva Full-Scale Alpha Factory Production Sweep
Orchestrates an end-to-end institutional quantitative sweep across all 76 alpha strategies,
all 50 NIFTY equities, and 4 intraday timeframes (5m, 10m, 15m, 30m).

Pipeline Stages:
1. Cross-Timeframe & Trailing Mode Optimization (76 Strategies x 50 Stocks x 4 Timeframes x 3 Modes)
2. CPCV & Probability of Backtest Overfitting (PBO) Guard (Marcos López de Prado)
3. Market Regime Decomposition & Alpha DNA Card Generation
4. Master Multi-Alpha Ensemble Portfolio Simulation (Concurrent Capital & Sector Caps)
5. Certified Production Model Pack Export (config/production_model_pack.json)
"""

import sys
import os
import json
import time
import importlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel
from src.backtest.engine import BacktestEngine
from src.research.regime_profiler import MarketRegimeProfiler
from src.research.cpcv_engine import CPCVEngine
from src.portfolio.master_portfolio_backtester import MasterPortfolioBacktester
from scripts.ingest_all_nifty50_timeframes import NIFTY_50_UNIVERSE


def load_all_strategy_classes() -> List[Tuple[str, Any]]:
    """Loads all 76 strategy classes dynamically from src/strategies."""
    strat_dir = Path("src/strategies")
    loaded = []
    for p in sorted(strat_dir.glob("alpha_*.py")):
        mod_name = f"src.strategies.{p.stem}"
        try:
            mod = importlib.import_module(mod_name)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and attr.startswith("Alpha") and hasattr(obj, "generate_signals"):
                    loaded.append((attr, obj))
                    break
        except Exception as e:
            print(f"[!] Warning: Could not load {p.name}: {e}")
    return loaded


def run_production_sweep(
    universe: List[str] = NIFTY_50_UNIVERSE,
    timeframes: List[str] = ["5m", "10m", "15m", "30m"],
    exit_modes: List[str] = ["BREAK_EVEN", "STEP_RATCHET"],
    initial_capital: float = 500000.0,
):
    print("=" * 110)
    print("                      ASHVA QUANTITATIVE ALPHA FACTORY: FULL PRODUCTION SWEEP")
    print(f"[*] Universe: {len(universe)} NIFTY Equities | Timeframes: {timeframes} | Capital: Rs {initial_capital:,.2f}")
    print("=" * 110)

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel()
    profiler = MarketRegimeProfiler(lake)
    cpcv_engine = CPCVEngine(n_partitions=6, k_test_partitions=2)

    strategy_classes = load_all_strategy_classes()
    print(f"[+] Successfully loaded {len(strategy_classes)} Alpha Strategy Classes from registry.\n")

    # =========================================================================
    # STAGE 1: Cross-Timeframe & Exit Mode Optimization per Strategy
    # =========================================================================
    print("-" * 110)
    print("STAGE 1: CROSS-TIMEFRAME & EXIT MODE OPTIMIZATION (ALL STRATEGIES x 50 STOCKS)")
    print("-" * 110)

    best_strategy_configs = []
    all_evaluated_records = []

    total_strats = len(strategy_classes)
    for s_idx, (strat_name, strat_cls) in enumerate(strategy_classes, 1):
        print(f"[{s_idx:02d}/{total_strats}] Optimizing {strat_name:35s}...", end="", flush=True)
        strat_best_pnl = -float("inf")
        strat_best_config = None

        for tf in timeframes:
            for mode in exit_modes:
                total_trades = 0
                gross_pnl = 0.0
                net_pnl = 0.0
                wins = 0
                gross_win = 0.0
                gross_loss = 0.0
                trade_returns = []

                for sym in universe:
                    df = lake.load_bars(sym.upper(), tf)
                    if df.empty or len(df) < 500:
                        continue

                    if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp").sort_index()

                    strat = strat_cls()
                    try:
                        df_signals = strat.generate_signals(df)
                    except Exception:
                        continue

                    if "signal" not in df_signals.columns:
                        continue

                    engine = BacktestEngine(cost_model=cost_model, initial_capital=initial_capital)
                    res = engine.run(df_signals, symbol=sym.upper(), strategy_id=strat_name, trailing_mode=mode)

                    total_trades += res.total_trades
                    sym_net = res.total_net_pnl
                    net_pnl += sym_net
                    gross_pnl += (res.final_equity - res.initial_capital)
                    wins += res.winning_trades
                    gross_win += sum([t.net_pnl for t in res.trade_list if t.net_pnl > 0])
                    gross_loss += sum([abs(t.net_pnl) for t in res.trade_list if t.net_pnl < 0])
                    trade_returns.extend([t.net_pnl for t in res.trade_list])

                if total_trades < 15:
                    continue

                win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
                pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
                ret_series = pd.Series(trade_returns)
                sharpe = float((ret_series.mean() / (ret_series.std() + 1e-6)) * np.sqrt(252)) if len(ret_series) > 5 else 0.0

                record = {
                    "strategy_name": strat_name,
                    "strategy_cls": strat_cls,
                    "timeframe": tf,
                    "trailing_mode": mode,
                    "total_trades": total_trades,
                    "net_pnl": round(net_pnl, 2),
                    "win_rate": round(win_rate, 1),
                    "profit_factor": round(pf, 2),
                    "sharpe": round(sharpe, 2),
                    "trade_returns": trade_returns,
                }
                all_evaluated_records.append(record)

                if net_pnl > strat_best_pnl:
                    strat_best_pnl = net_pnl
                    strat_best_config = record

        if strat_best_config and strat_best_config["net_pnl"] > 0:
            print(f" [PASS: TF={strat_best_config['timeframe']}, Mode={strat_best_config['trailing_mode']}, Net PnL=+Rs {strat_best_config['net_pnl']:,.2f}, Sharpe={strat_best_config['sharpe']:.2f}]")
            best_strategy_configs.append(strat_best_config)
        else:
            pnl_str = f"Rs {strat_best_pnl:,.2f}" if strat_best_pnl > -float('inf') else "No Trades"
            print(f" [REJECT: Best Net PnL={pnl_str}]")

    print(f"\n[+] Total Strategies with Positive Net PnL across Universe: {len(best_strategy_configs)}/{total_strats}\n")

    # =========================================================================
    # STAGE 2: CPCV & Statistical Overfitting Verification (PBO <= 30%)
    # =========================================================================
    print("-" * 110)
    print("STAGE 2: COMBINATORIAL PURGED CROSS-VALIDATION (CPCV) & OVERFITTING FILTER")
    print("-" * 110)

    champion_alphas = []
    for candidate in best_strategy_configs:
        strat_name = candidate["strategy_name"]
        trade_returns = candidate["trade_returns"]

        cpcv_res = cpcv_engine.evaluate_trades(trade_returns)
        pbo = cpcv_res.get("pbo", 1.0)
        oos_sharpe = cpcv_res.get("mean_oos_sharpe", 0.0)
        is_overfitted = cpcv_res.get("is_overfitted", True)

        candidate["cpcv"] = cpcv_res

        if not is_overfitted and oos_sharpe >= 0.50 and pbo <= 0.30:
            print(f"[+] CHAMPION ALPHA CERTIFIED: {strat_name:32s} | TF={candidate['timeframe']:3s} | Net PnL=+Rs {candidate['net_pnl']:9.2f} | OOS Sharpe={oos_sharpe:.2f} | PBO={cpcv_res['pbo_pct']}")
            champion_alphas.append(candidate)
        else:
            print(f"[-] OVERFIT REJECTION:        {strat_name:32s} | PBO={cpcv_res.get('pbo_pct', '100%')} | OOS Sharpe={oos_sharpe:.2f}")

    print(f"\n[+] Certified All-Weather Champion Alphas Passing CPCV/PBO: {len(champion_alphas)}\n")

    if not champion_alphas:
        print("[!] No champion alphas passed CPCV. Exiting.")
        return

    # =========================================================================
    # STAGE 3: Market Regime Decomposition & Alpha DNA Cards
    # =========================================================================
    print("-" * 110)
    print("STAGE 3: MARKET REGIME DECOMPOSITION & ALPHA DNA GENERATION")
    print("-" * 110)

    for champ in champion_alphas:
        strat_name = champ["strategy_name"]
        strat_cls = champ["strategy_cls"]
        tf = champ["timeframe"]
        mode = champ["trailing_mode"]

        # Collect discrete trades for regime tagging
        all_trades_raw = []
        for sym in universe:
            df = lake.load_bars(sym.upper(), tf)
            if df.empty or len(df) < 500:
                continue

            if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            strat = strat_cls()
            df_signals = strat.generate_signals(df)
            engine = BacktestEngine(cost_model=cost_model, initial_capital=initial_capital)
            res = engine.run(df_signals, symbol=sym.upper(), strategy_id=strat_name, trailing_mode=mode)
            for t in res.trade_list:
                all_trades_raw.append({
                    "entry_time": t.entry_time,
                    "pnl": t.net_pnl,
                    "gap_pct": float(df_signals.loc[df_signals.index == t.entry_time].iloc[0].get("gap_pct", 0.0)) if t.entry_time in df_signals.index else 0.0
                })

        dna = profiler.profile_trades(all_trades_raw, alpha_id=strat_name)
        champ["dna_card"] = dna
        print(f"[+] Generated DNA Card for {strat_name:32s} | Approved Regimes: {len(dna.get('approved_regimes', []))} | Disabled: {len(dna.get('disabled_regimes', []))}")

    # =========================================================================
    # STAGE 4: Master Multi-Alpha Ensemble Portfolio Simulation
    # =========================================================================
    print("\n" + "-" * 110)
    print("STAGE 4: MASTER ENSEMBLE PORTFOLIO SIMULATION (CONCURRENT CAPITAL & SECTOR CAPS)")
    print("-" * 110)

    tester = MasterPortfolioBacktester(
        data_lake=lake,
        initial_capital=initial_capital,
        risk_per_trade_inr=2500.0,
        max_concurrent_positions=5,
        max_positions_per_sector=2,
        cost_model=cost_model,
    )

    strategies_to_run = [c["strategy_cls"] for c in champion_alphas]
    tf_map = {c["strategy_name"]: c["timeframe"] for c in champion_alphas}
    trail_map = {c["strategy_name"]: c["trailing_mode"] for c in champion_alphas}

    portfolio_res = tester.run_portfolio_backtest(
        strategies=strategies_to_run,
        symbols=universe,
        strategy_timeframe_map=tf_map,
        strategy_trailing_map=trail_map,
        use_regime_filter=True,
    )

    print("\n========================= MASTER PORTFOLIO TRACK RECORD =========================")
    print(f"Initial Capital:         Rs {portfolio_res['initial_capital']:,.2f}")
    print(f"Final Equity:            Rs {portfolio_res['final_equity']:,.2f}")
    print(f"Total Net PnL:           Rs {portfolio_res['total_net_pnl']:,.2f} ({portfolio_res['roi_pct']:+.2f}% Net ROI)")
    print(f"Total Executed Trades:   {portfolio_res['total_trades']:,}")
    print(f"Winning Trades:          {portfolio_res['winning_trades']:,} ({portfolio_res['win_rate']})")
    print(f"Profit Factor:           {portfolio_res['profit_factor']:.2f}")
    print(f"Portfolio Sharpe Ratio:  {portfolio_res['portfolio_sharpe']:.2f}")
    print(f"Portfolio Sortino Ratio: {portfolio_res['portfolio_sortino']:.2f}")
    print(f"Maximum Drawdown:        {portfolio_res['max_drawdown_pct']:.2f}%")
    print("---------------------------------------------------------------------------------")
    print("Top Alpha PnL Contributors:")
    for a_id, pnl in list(portfolio_res["alpha_contribution"].items())[:10]:
        print(f" - {a_id:35s}: Rs {pnl:+10,.2f}")
    print("=================================================================================\n")

    # =========================================================================
    # STAGE 5: Export Certified Production Model Pack
    # =========================================================================
    print("-" * 110)
    print("STAGE 5: EXPORT PRODUCTION MODEL PACK")
    print("-" * 110)

    model_pack = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "universe_size": len(universe),
        "portfolio_metrics": {
            "initial_capital": portfolio_res["initial_capital"],
            "final_equity": portfolio_res["final_equity"],
            "net_pnl": portfolio_res["total_net_pnl"],
            "roi_pct": portfolio_res["roi_pct"],
            "total_trades": portfolio_res["total_trades"],
            "win_rate": portfolio_res["win_rate"],
            "profit_factor": portfolio_res["profit_factor"],
            "portfolio_sharpe": portfolio_res["portfolio_sharpe"],
            "portfolio_sortino": portfolio_res["portfolio_sortino"],
            "max_drawdown_pct": portfolio_res["max_drawdown_pct"],
        },
        "champion_alphas": [
            {
                "strategy_name": c["strategy_name"],
                "optimal_timeframe": c["timeframe"],
                "optimal_trailing_mode": c["trailing_mode"],
                "standalone_net_pnl": c["net_pnl"],
                "standalone_sharpe": c["sharpe"],
                "win_rate": c["win_rate"],
                "profit_factor": c["profit_factor"],
                "pbo": c["cpcv"]["pbo_pct"],
                "mean_oos_sharpe": c["cpcv"]["mean_oos_sharpe"],
                "approved_regimes": c.get("dna_card", {}).get("approved_regimes", []),
                "disabled_regimes": c.get("dna_card", {}).get("disabled_regimes", []),
            }
            for c in champion_alphas
        ]
    }

    out_file = Path("config/production_model_pack.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(model_pack, f, indent=2)
    print(f"[+] Saved Production Model Pack to {out_file} with {len(champion_alphas)} certified alphas!")

    print("\n" + "=" * 110)
    print("                      PRODUCTION SWEEP COMPLETE & CERTIFIED")
    print("=" * 110)


if __name__ == "__main__":
    run_production_sweep()
