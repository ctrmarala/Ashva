"""
Ashva Evidence-Based Dispatcher & Rejection Opportunity Cost Analyzer

Deeply investigates the 87.1% signal rejection problem across 50 proven alphas & 77 liquid stocks:
1. Opportunity Cost of the 2,086 Rejected Trades (Gross/Net PnL, Win Rate %, Money Left on Table).
2. Systematic Alpha Starvation (Which high-edge alphas are getting blocked by slot hogs).
3. Intra-Stock Collision & Symbol Competition (Multiple alphas piling onto the same ticker).
4. Slot Capacity Sensitivity Curve (K = 4 -> 6 -> 8 -> 10 -> 12 -> 16 -> 20 slots).
5. Evidence-Based Dispatcher Policy Benchmarking:
   - Policy 1: Baseline FIFO (Chronological + Alpha ID)
   - Policy 2: Alpha Diversity (Max 1 position per alpha strategy)
   - Policy 3: Symbol Diversity (Max 1 position per stock ticker)
   - Policy 4: Combined Diversity (Max 1 per alpha + Max 1 per symbol)
   - Policy 5: Historical Quality Priority (Sorted by Alpha OOS Profit Factor)
   - Policy 6: Quality Priority + Combined Diversity (Institutional Benchmark)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.registry import get_strategy_by_name
from src.ui.data_access import UIDataAccess
from src.analytics.metrics import calculate_profit_factor


def harvest_all_candidates(lake: DataLake, symbols: list, alpha_ids: list, dal: UIDataAccess, registry_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Harvests all candidate trade signals and pre-computes their standalone economics."""
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    engine = BacktestEngine(
        cost_model=cost_model,
        initial_capital=500000.0,
        segment=Segment.EQUITY_INTRADAY,
        use_1m_intrabar=True,
        data_lake=lake,
    )

    all_candidates = []
    print(f"[*] Harvesting candidate trades for {len(alpha_ids)} alphas across {len(symbols)} symbols...", flush=True)

    df_cache = {}
    reg_lookup = registry_df.set_index("alpha_id")

    for idx, alpha_id in enumerate(alpha_ids, 1):
        strat_cls = get_strategy_by_name(alpha_id)
        if not strat_cls:
            continue

        detail = dal.get_alpha_detail(alpha_id)
        optimal_tf = detail.get("timeframe") or "15m"
        
        # Pull exact registered OOS metrics
        if alpha_id in reg_lookup.index:
            row = reg_lookup.loc[alpha_id]
            oos_pf = float(row.get("profit_factor", 1.5))
            oos_sharpe = float(row.get("oos_sharpe", 1.0))
        else:
            oos_pf = 1.5
            oos_sharpe = 1.0

        strat = strat_cls({"timeframe": optimal_tf})

        alpha_trade_count = 0
        for sym in symbols:
            cache_key = (sym, optimal_tf)
            if cache_key not in df_cache:
                df_cache[cache_key] = lake.load_bars(sym, optimal_tf, max_lookback_days=540)
            df = df_cache[cache_key]

            if df.empty or len(df) < 50:
                continue

            sig_df = strat.generate_signals(df)
            res = engine.run(sig_df, symbol=sym, strategy_id=alpha_id, capital_per_trade_pct=0.25)
            
            for t in res.trade_list:
                alloc_cap = 125000.0
                qty = max(1, int(alloc_cap // t.entry_price))
                is_buy = (t.side == "BUY")
                is_sl = ("STOP" in t.exit_reason.upper() or "SL" in t.exit_reason.upper())
                buy_px = t.entry_price if is_buy else t.exit_price
                sell_px = t.exit_price if is_buy else t.entry_price

                costs = cost_model.calculate_trade_costs(
                    buy_price=buy_px,
                    sell_price=sell_px,
                    quantity=qty,
                    segment=Segment.EQUITY_INTRADAY,
                    is_stop_loss=is_sl,
                )

                all_candidates.append({
                    "alpha_id": alpha_id,
                    "symbol": sym,
                    "side": t.side,
                    "entry_time": pd.to_datetime(t.entry_time),
                    "exit_time": pd.to_datetime(t.exit_time),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "quantity_125k": qty,
                    "gross_pnl_125k": costs.gross_pnl,
                    "costs_125k": costs.total_tax_and_charges,
                    "net_pnl_125k": costs.net_pnl,
                    "oos_pf": oos_pf,
                    "oos_sharpe": oos_sharpe,
                    "optimal_tf": optimal_tf,
                })
                alpha_trade_count += 1

        print(f"  [{idx:02d}/{len(alpha_ids):02d}] {alpha_id:<12} ({optimal_tf:>3}): {alpha_trade_count:4d} signals (Historical OOS PF: {oos_pf:.2f}, Sharpe: {oos_sharpe:.2f})", flush=True)

    print(f"[+] Total raw candidate signals harvested: {len(all_candidates):,}\n", flush=True)
    return all_candidates


def simulate_dispatcher(
    candidates: List[Dict[str, Any]],
    capital: float = 500000.0,
    max_slots: int = 4,
    policy: str = "FIFO",
    cost_model: IndianCostModel = None,
) -> Dict[str, Any]:
    """Simulates portfolio execution under a specific dispatching & prioritization policy."""
    cost_model = cost_model or IndianCostModel(default_slippage_bps=3.0)
    slot_cap = capital / max_slots

    df_cands = pd.DataFrame(candidates).sort_values(by=["entry_time"]).reset_index(drop=True)
    time_groups = df_cands.groupby("entry_time")

    active_positions: List[Dict[str, Any]] = []
    executed_trades = []
    rejected_trades = []

    for entry_time, group in time_groups:
        current_time = entry_time

        # 1. Release exiting positions
        still_active = []
        for pos in active_positions:
            if pos["exit_time"] <= current_time:
                executed_trades.append(pos)
            else:
                still_active.append(pos)
        active_positions = still_active

        # 2. Sort candidate batch arriving at this timestamp according to policy
        cand_list = group.to_dict("records")

        if policy == "FIFO":
            cand_list.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
        elif policy in ["QUALITY_PRIORITY", "QUALITY_COMBINED"]:
            cand_list.sort(key=lambda x: (-x["oos_pf"], -x["oos_sharpe"], x["alpha_id"]))
        elif policy in ["ALPHA_DIVERSE", "SYMBOL_DIVERSE", "COMBINED_DIVERSE"]:
            cand_list.sort(key=lambda x: (-x["oos_pf"], x["alpha_id"]))

        # 3. Evaluate each candidate against active constraints
        for c in cand_list:
            if len(active_positions) >= max_slots:
                rejected_trades.append({**c, "rejection_reason": "CONCURRENCY_LIMIT"})
                continue

            active_alphas = {p["alpha_id"] for p in active_positions}
            active_symbols = {p["symbol"] for p in active_positions}

            if policy in ["ALPHA_DIVERSE", "COMBINED_DIVERSE", "QUALITY_COMBINED"] and c["alpha_id"] in active_alphas:
                rejected_trades.append({**c, "rejection_reason": "ALPHA_CONCURRENCY_CAP"})
                continue

            if policy in ["SYMBOL_DIVERSE", "COMBINED_DIVERSE", "QUALITY_COMBINED"] and c["symbol"] in active_symbols:
                rejected_trades.append({**c, "rejection_reason": "SYMBOL_CONCURRENCY_CAP"})
                continue

            qty = max(1, int(slot_cap // c["entry_price"]))
            is_buy = (c["side"] == "BUY")
            is_sl = ("STOP" in c["exit_reason"].upper() or "SL" in c["exit_reason"].upper())
            buy_px = c["entry_price"] if is_buy else c["exit_price"]
            sell_px = c["exit_price"] if is_buy else c["entry_price"]

            costs = cost_model.calculate_trade_costs(
                buy_price=buy_px,
                sell_price=sell_px,
                quantity=qty,
                segment=Segment.EQUITY_INTRADAY,
                is_stop_loss=is_sl,
            )

            pos_record = {
                **c,
                "quantity": qty,
                "allocated_capital": qty * c["entry_price"],
                "gross_pnl": costs.gross_pnl,
                "costs": costs.total_tax_and_charges,
                "net_pnl": costs.net_pnl,
            }
            active_positions.append(pos_record)

    for pos in active_positions:
        executed_trades.append(pos)

    n_exec = len(executed_trades)
    n_rej = len(rejected_trades)
    tot_gross = sum(t["gross_pnl"] for t in executed_trades)
    tot_costs = sum(t["costs"] for t in executed_trades)
    tot_net = sum(t["net_pnl"] for t in executed_trades)
    wins = [t for t in executed_trades if t["net_pnl"] > 0]
    wr = (len(wins) / max(1, n_exec)) * 100.0
    pf = calculate_profit_factor([t["net_pnl"] for t in executed_trades]) if n_exec > 0 else 0.0
    roi = (tot_net / capital) * 100.0

    return {
        "policy": policy,
        "max_slots": max_slots,
        "executed_count": n_exec,
        "rejected_count": n_rej,
        "acceptance_rate_pct": (n_exec / max(1, n_exec + n_rej)) * 100.0,
        "gross_pnl": tot_gross,
        "costs": tot_costs,
        "net_pnl": tot_net,
        "roi_pct": roi,
        "win_rate_pct": wr,
        "profit_factor": pf,
        "executed_trades": executed_trades,
        "rejected_trades": rejected_trades,
    }


def main():
    lake = DataLake(read_only=True)
    symbols = get_universe_symbols()
    dal = UIDataAccess()

    all_alphas_df = dal.get_alpha_registry_table()
    proven_df = all_alphas_df[all_alphas_df["status"] == "PROVEN"]
    alpha_ids = proven_df["alpha_id"].tolist()

    print("=" * 125)
    print("ASHVA EVIDENCE-BASED DISPATCHER & REJECTION OPPORTUNITY COST RESEARCH")
    print("=" * 125)

    all_candidates = harvest_all_candidates(lake, symbols, alpha_ids, dal, proven_df)

    s_ts = pd.to_datetime("2026-04-19 00:00:00")
    e_ts = pd.to_datetime("2026-09-18 23:59:59")
    window_cands = [c for c in all_candidates if s_ts <= c["entry_time"] <= e_ts]
    print(f"[*] 5-Month Evaluation Window Candidates: {len(window_cands):,} signals (Apr 19 to Sep 18, 2026)\n")

    # PART 1: OPPORTUNITY COST ANALYSIS OF BASELINE FIFO REJECTIONS
    fifo_res = simulate_dispatcher(window_cands, capital=500000.0, max_slots=4, policy="FIFO")
    rej_trades = fifo_res["rejected_trades"]
    exec_trades = fifo_res["executed_trades"]

    print("=" * 125)
    print("PART 1: OPPORTUNITY COST OF THE 2,086 REJECTED TRADES (BASELINE FIFO)")
    print("=" * 125)

    rej_gross = sum(t["gross_pnl_125k"] for t in rej_trades)
    rej_costs = sum(t["costs_125k"] for t in rej_trades)
    rej_net = sum(t["net_pnl_125k"] for t in rej_trades)
    rej_wins = [t for t in rej_trades if t["net_pnl_125k"] > 0]
    rej_wr = (len(rej_wins) / max(1, len(rej_trades))) * 100.0
    rej_pf = calculate_profit_factor([t["net_pnl_125k"] for t in rej_trades])

    print(f"  * Total Rejected Trades:           {len(rej_trades):,} signals ({fifo_res['rejected_count']/(len(window_cands))*100:.1f}% of all signals)")
    print(f"  * Rejected Trades Win Rate:        {rej_wr:.1f}% ({len(rej_wins):,} profitable trades left on the table)")
    print(f"  * Rejected Standalone Gross PnL:   Rs {rej_gross:+12,.2f}")
    print(f"  * Rejected Standalone Taxes/Fees:  Rs {rej_costs:12,.2f}")
    print(f"  * REJECTED STANDALONE NET PNL:     Rs {rej_net:+12,.2f} (Net PF: {rej_pf:.2f})")
    print(f"  * Selected FIFO Net PnL (Actual):  Rs {fifo_res['net_pnl']:+12,.2f} (Win Rate: {fifo_res['win_rate_pct']:.1f}%)")
    print("=" * 125)

    # PART 2: ALPHA STARVATION & CANNIBALIZATION AUDIT
    print("\n" + "=" * 125)
    print("PART 2: SYSTEMATIC ALPHA CANNIBALIZATION & STARVATION AUDIT")
    print("=" * 125)

    df_win = pd.DataFrame(window_cands)
    reg_lookup = proven_df.set_index("alpha_id")
    alpha_summary = []
    for a_id in alpha_ids:
        a_cands = [c for c in window_cands if c["alpha_id"] == a_id]
        a_exec = [c for c in exec_trades if c["alpha_id"] == a_id]
        a_rej = [c for c in rej_trades if c["alpha_id"] == a_id]
        if not a_cands:
            continue
        
        oos_pf = float(reg_lookup.loc[a_id, "profit_factor"]) if a_id in reg_lookup.index else 1.0
        net_edge_left = sum(t["net_pnl_125k"] for t in a_rej)

        alpha_summary.append({
            "alpha_id": a_id,
            "tf": a_cands[0]["optimal_tf"],
            "oos_pf": oos_pf,
            "signals": len(a_cands),
            "executed": len(a_exec),
            "rejected": len(a_rej),
            "rejection_rate": (len(a_rej) / len(a_cands)) * 100.0,
            "net_left_table": net_edge_left,
        })

    df_alpha_audit = pd.DataFrame(alpha_summary).sort_values(by="net_left_table", ascending=False)
    print(f"{'ALPHA ID':<14} | {'TF':<4} | {'OOS PF':<6} | {'SIGNALS':<7} | {'EXECUTED':<8} | {'REJECTED':<8} | {'REJ %':<7} | {'NET PNL LEFT ON TABLE':<22}")
    print("-" * 125)
    for row in df_alpha_audit.head(15).itertuples():
        print(f"{row.alpha_id:<14} | {row.tf:<4} | {row.oos_pf:>6.2f} | {row.signals:>7d} | {row.executed:>8d} | {row.rejected:>8d} | {row.rejection_rate:>6.1f}% | Rs {row.net_left_table:>18,.2f}")
    print("-" * 125)
    print("  * Top 15 highest-edge alphas starved of execution capital by early FIFO arrivals.")

    # PART 3: INTRA-STOCK COMPETITION & SIMULTANEOUS SIGNAL COLLISION
    print("\n" + "=" * 125)
    print("PART 3: INTRA-STOCK COLLISION (MULTIPLE ALPHAS FIRING ON THE SAME TICKER)")
    print("=" * 125)
    sym_collisions = df_win.groupby(["entry_time", "symbol"]).size()
    multi_sym = sym_collisions[sym_collisions > 1]
    print(f"  * Instances where >= 2 alphas fired on the SAME stock at the EXACT same bar: {len(multi_sym):,} occurrences")
    print(f"  * Max alphas firing simultaneously on a single stock in 1 bar:              {sym_collisions.max()} alphas")

    # PART 4: SLOT CAPACITY SCALING SENSITIVITY (K = 4 -> 20)
    print("\n" + "=" * 125)
    print("PART 4: SLOT CAPACITY SCALING SENSITIVITY (FIXED Rs 500,000 CAPITAL DIVIDED BY K SLOTS)")
    print("=" * 125)
    print(f"{'SLOTS (K)':<10} | {'SLOT SIZING':<14} | {'EXECUTED':<8} | {'REJECTED':<8} | {'ACCEPT %':<8} | {'WIN %':<6} | {'GROSS PNL':<12} | {'TAXES/FEES':<11} | {'NET PNL':<12} | {'ROI %':<8} | {'NET PF':<6}")
    print("-" * 125)

    for k in [4, 6, 8, 10, 12, 16, 20]:
        res_k = simulate_dispatcher(window_cands, capital=500000.0, max_slots=k, policy="FIFO")
        slot_sz = 500000.0 / k
        print(f"{k:<10d} | Rs {slot_sz:>10,.0f} | {res_k['executed_count']:>8d} | {res_k['rejected_count']:>8d} | {res_k['acceptance_rate_pct']:>7.1f}% | {res_k['win_rate_pct']:>5.1f}% | Rs {res_k['gross_pnl']:>9,.0f} | Rs {res_k['costs']:>8,.0f} | Rs {res_k['net_pnl']:>9,.0f} | {res_k['roi_pct']:>+6.2f}% | {res_k['profit_factor']:>5.2f}")

    # PART 5: EVIDENCE-BASED DISPATCHER POLICY BENCHMARKING (AT K = 4 SLOTS)
    print("\n" + "=" * 125)
    print("PART 5: EVIDENCE-BASED DISPATCHER POLICY BENCHMARK (FIXED 4 SLOTS @ Rs 125,000 EACH)")
    print("=" * 125)
    print(f"{'DISPATCHER POLICY':<35} | {'EXECUTED':<8} | {'REJECTED':<8} | {'WIN %':<6} | {'GROSS PNL':<12} | {'TAXES/FEES':<11} | {'NET PNL':<12} | {'ROI %':<8} | {'NET PF':<6}")
    print("-" * 125)

    policies = [
        ("1. Baseline FIFO (Chronological)", "FIFO"),
        ("2. Alpha Diversity (Max 1 / Alpha)", "ALPHA_DIVERSE"),
        ("3. Symbol Diversity (Max 1 / Symbol)", "SYMBOL_DIVERSE"),
        ("4. Combined Diversity (Alpha + Sym)", "COMBINED_DIVERSE"),
        ("5. Quality Priority (OOS PF Rank)", "QUALITY_PRIORITY"),
        ("6. Quality + Combined Diversity", "QUALITY_COMBINED"),
    ]

    for label, pol in policies:
        res_p = simulate_dispatcher(window_cands, capital=500000.0, max_slots=4, policy=pol)
        print(f"{label:<35} | {res_p['executed_count']:>8d} | {res_p['rejected_count']:>8d} | {res_p['win_rate_pct']:>5.1f}% | Rs {res_p['gross_pnl']:>9,.0f} | Rs {res_p['costs']:>8,.0f} | Rs {res_p['net_pnl']:>9,.0f} | {res_p['roi_pct']:>+6.2f}% | {res_p['profit_factor']:>5.2f}")

    print("=" * 125)


if __name__ == "__main__":
    main()
