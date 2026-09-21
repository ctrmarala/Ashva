"""
Ashva PCDE Benchmark Suite - Phase 1.5: Capital Sensitivity & Rigorous Anchors

Evaluates the 4 core empirical anchors (FIFO, Symbol-Diverse, Random Nulls, True Clairvoyant MILP)
across 4 distinct capital scaling levels:
- ₹5,00,000  (₹1.25L per slot)
- ₹10,00,000 (₹2.50L per slot)
- ₹25,00,000 (₹6.25L per slot)
- ₹50,00,000 (₹12.50L per slot)

Investigates whether friction drag is a function of under-capitalization (fixed ₹20 brokerage & roundtrip drag)
vs true alpha extraction quality.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.registry import get_strategy_by_name
from src.ui.data_access import UIDataAccess
from src.analytics.metrics import calculate_profit_factor


def harvest_raw_candidates(lake: DataLake, symbols: list, alpha_ids: list, dal: UIDataAccess) -> List[Dict[str, Any]]:
    """Harvests immutable raw candidate signals without precomputing sizing."""
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    engine = BacktestEngine(
        cost_model=cost_model,
        initial_capital=500000.0,
        segment=Segment.EQUITY_INTRADAY,
        use_1m_intrabar=True,
        data_lake=lake,
    )

    all_candidates = []
    print(f"[*] Harvesting raw candidate signals for {len(alpha_ids)} alphas across {len(symbols)} symbols...", flush=True)

    df_cache = {}

    family_map = {}
    for a_id in alpha_ids:
        num = int("".join(filter(str.isdigit, a_id))) if any(c.isdigit() for c in a_id) else 0
        if 105 <= num <= 109 or num in [61, 62, 63, 64, 65]:
            family_map[a_id] = "VWAP_MEAN_REVERSION"
        elif 110 <= num <= 114 or num in [33, 36, 45, 46, 49]:
            family_map[a_id] = "GAP_EXHAUSTION"
        elif 115 <= num <= 119 or num in [54, 55, 56, 57, 58, 59, 60]:
            family_map[a_id] = "INITIAL_BALANCE"
        else:
            family_map[a_id] = "TREND_CONTINUATION"

    for idx, alpha_id in enumerate(alpha_ids, 1):
        strat_cls = get_strategy_by_name(alpha_id)
        if not strat_cls:
            continue

        detail = dal.get_alpha_detail(alpha_id)
        optimal_tf = detail.get("timeframe") or "15m"
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
                all_candidates.append({
                    "alpha_id": alpha_id,
                    "family": family_map.get(alpha_id, "OTHER"),
                    "symbol": sym,
                    "side": t.side,
                    "entry_time": pd.to_datetime(t.entry_time),
                    "exit_time": pd.to_datetime(t.exit_time),
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price),
                    "exit_reason": t.exit_reason,
                })
                alpha_trade_count += 1

    print(f"[+] Total raw candidate signals harvested: {len(all_candidates):,}\n", flush=True)
    return all_candidates


def evaluate_candidate_economics(cand: Dict[str, Any], slot_cap: float, cost_model: IndianCostModel) -> Dict[str, Any]:
    """Calculates trade economics for a specific allocated slot capital without max(1, qty) bug."""
    entry_px = cand["entry_price"]
    exit_px = cand["exit_price"]
    
    qty = int(slot_cap // entry_px)
    if qty <= 0:
        return {**cand, "quantity": 0, "allocated_capital": 0.0, "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0, "turnover": 0.0, "eligible": False}

    is_buy = (cand["side"] == "BUY")
    is_sl = ("STOP" in cand["exit_reason"].upper() or "SL" in cand["exit_reason"].upper())
    buy_px = entry_px if is_buy else exit_px
    sell_px = exit_px if is_buy else entry_px

    costs = cost_model.calculate_trade_costs(
        buy_price=buy_px,
        sell_price=sell_px,
        quantity=qty,
        segment=Segment.EQUITY_INTRADAY,
        is_stop_loss=is_sl,
    )

    return {
        **cand,
        "quantity": qty,
        "allocated_capital": qty * entry_px,
        "gross_pnl": costs.gross_pnl,
        "costs": costs.total_tax_and_charges,
        "net_pnl": costs.net_pnl,
        "turnover": costs.total_turnover,
        "eligible": True,
    }


def compute_metrics(executed_trades: List[Dict[str, Any]], capital: float, days: int = 105) -> Dict[str, Any]:
    """Computes standardized metrics from executed trades."""
    if not executed_trades:
        return {
            "trades": 0, "win_rate": 0.0, "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0,
            "roi_pct": 0.0, "profit_factor": 0.0, "max_dd_pct": 0.0, "friction_ratio": 0.0,
            "symbol_hhi": 0.0, "sharpe": 0.0,
        }

    df = pd.DataFrame(executed_trades).sort_values(by="exit_time").reset_index(drop=True)
    n_trades = len(df)
    tot_gross = df["gross_pnl"].sum()
    tot_costs = df["costs"].sum()
    tot_net = df["net_pnl"].sum()
    tot_turnover = df["turnover"].sum()
    wins = df[df["net_pnl"] > 0]
    wr = (len(wins) / n_trades) * 100.0
    pf = calculate_profit_factor(df["net_pnl"].tolist())
    roi = (tot_net / capital) * 100.0
    friction_ratio = tot_costs / max(1.0, abs(tot_gross))

    # Realized equity curve & drawdown
    df["cum_net"] = df["net_pnl"].cumsum()
    df["equity"] = capital + df["cum_net"]
    df["peak"] = df["equity"].cummax()
    df["dd_inr"] = df["equity"] - df["peak"]
    df["dd_pct"] = (df["dd_inr"] / df["peak"]) * 100.0
    max_dd_pct = abs(float(df["dd_pct"].min()))

    # Daily Sharpe
    df["date"] = df["exit_time"].dt.date
    daily_pnl = df.groupby("date")["net_pnl"].sum()
    daily_rets = daily_pnl / capital
    std = float(daily_rets.std()) if len(daily_rets) > 1 else 0.0
    mean = float(daily_rets.mean()) if len(daily_rets) > 0 else 0.0
    sharpe = (mean / std * np.sqrt(252)) if std > 1e-7 else 0.0

    # Symbol HHI
    sym_turnover = df.groupby("symbol")["turnover"].sum()
    sym_shares = sym_turnover / max(1.0, tot_turnover)
    symbol_hhi = float((sym_shares ** 2).sum() * 10000.0)

    return {
        "trades": n_trades,
        "win_rate": wr,
        "gross_pnl": tot_gross,
        "costs": tot_costs,
        "net_pnl": tot_net,
        "roi_pct": roi,
        "profit_factor": pf,
        "max_dd_pct": max_dd_pct,
        "friction_ratio": friction_ratio,
        "symbol_hhi": symbol_hhi,
        "sharpe": sharpe,
    }


def solve_milp_clairvoyant(candidates_evaluated: List[Dict[str, Any]], max_slots: int = 4, enforce_symbol_diversity: bool = True) -> List[Dict[str, Any]]:
    """Global Binary Integer Linear Programming Upper Bound Optimizer with Skip."""
    pos_cands = [c for c in candidates_evaluated if c["eligible"] and c["net_pnl"] > 0]
    if not pos_cands:
        return []

    N = len(pos_cands)
    c = np.array([-c_item["net_pnl"] for c_item in pos_cands], dtype=np.float64)

    all_events = sorted(list(set([c_item["entry_time"] for c_item in pos_cands] + [c_item["exit_time"] for c_item in pos_cands])))

    row_list = []
    rhs_list = []

    for t in all_events:
        active_indices = [i for i, c_item in enumerate(pos_cands) if c_item["entry_time"] <= t < c_item["exit_time"]]
        if len(active_indices) > max_slots:
            row = np.zeros(N, dtype=np.float64)
            for idx in active_indices:
                row[idx] = 1.0
            row_list.append(row)
            rhs_list.append(float(max_slots))

        if enforce_symbol_diversity:
            sym_map = {}
            for idx in active_indices:
                sym = pos_cands[idx]["symbol"]
                sym_map.setdefault(sym, []).append(idx)
            for sym, sym_indices in sym_map.items():
                if len(sym_indices) > 1:
                    row = np.zeros(N, dtype=np.float64)
                    for idx in sym_indices:
                        row[idx] = 1.0
                    row_list.append(row)
                    rhs_list.append(1.0)

    if row_list:
        A_ub = np.array(row_list)
        b_ub = np.array(rhs_list)
        constraints = LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    else:
        constraints = None

    integrality = np.ones(N, dtype=int)
    bounds = (np.zeros(N), np.ones(N))

    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
    if not res.success:
        return []

    x_sol = res.x.round().astype(int)
    return [pos_cands[i] for i in range(N) if x_sol[i] == 1]


def run_capital_level(raw_window_cands: List[Dict[str, Any]], capital: float, cost_model: IndianCostModel) -> Dict[str, Any]:
    """Evaluates all 4 Phase 1.5 benchmarks under a specific portfolio capital scale."""
    slot_cap = capital / 4.0
    cands_eval = [evaluate_candidate_economics(c, slot_cap, cost_model) for c in raw_window_cands]
    valid_cands = [c for c in cands_eval if c["eligible"]]
    infeasible_count = len(cands_eval) - len(valid_cands)

    df_cands = pd.DataFrame(valid_cands).sort_values(by=["entry_time"]).reset_index(drop=True)
    time_groups = list(df_cands.groupby("entry_time"))

    # 1. Test A: Baseline FIFO
    active_pos = []
    exec_a = []
    for entry_time, group in time_groups:
        active_pos = [p for p in active_pos if p["exit_time"] > entry_time]
        cand_list = group.to_dict("records")
        cand_list.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
        for c in cand_list:
            if len(active_pos) >= 4:
                continue
            active_pos.append(c)
            exec_a.append(c)
    mA = compute_metrics(exec_a, capital)

    # 2. Test C: Symbol Diversity (Max 1 per symbol)
    active_pos = []
    exec_c = []
    for entry_time, group in time_groups:
        active_pos = [p for p in active_pos if p["exit_time"] > entry_time]
        cand_list = group.to_dict("records")
        cand_list.sort(key=lambda x: (x["alpha_id"], x["symbol"]))
        for c in cand_list:
            if len(active_pos) >= 4:
                continue
            active_syms = {p["symbol"] for p in active_pos}
            if c["symbol"] in active_syms:
                continue
            active_pos.append(c)
            exec_c.append(c)
    mC = compute_metrics(exec_c, capital)

    # 3. Test I-C: Random Null (500 Seeds for speed)
    random_ic_nets = []
    for seed in range(1, 501):
        rng = random.Random(seed)
        active_pos = []
        exec_ic = []
        for entry_time, group in time_groups:
            active_pos = [p for p in active_pos if p["exit_time"] > entry_time]
            cand_list = group.to_dict("records")
            rng.shuffle(cand_list)
            for c in cand_list:
                if len(active_pos) >= 4:
                    continue
                active_syms = {p["symbol"] for p in active_pos}
                if c["symbol"] in active_syms:
                    continue
                active_pos.append(c)
                exec_ic.append(c)
        m_ic = compute_metrics(exec_ic, capital)
        random_ic_nets.append(m_ic["net_pnl"])

    arr_ic = np.array(random_ic_nets)
    mean_ic = float(np.mean(arr_ic))
    p_val_c = float((1.0 + (arr_ic >= mC["net_pnl"]).sum()) / (1.0 + len(arr_ic)))

    # 4. Test J1: True Clairvoyant Upper Bound (MILP)
    exec_j1 = solve_milp_clairvoyant(valid_cands, max_slots=4, enforce_symbol_diversity=True)
    mJ1 = compute_metrics(exec_j1, capital)

    # Opportunity Capture
    denom_capture = mJ1["net_pnl"] - mA["net_pnl"]
    cap_rate_c = ((mC["net_pnl"] - mA["net_pnl"]) / denom_capture * 100.0) if denom_capture > 0 else 0.0

    return {
        "capital": capital,
        "slot_cap": slot_cap,
        "infeasible_count": infeasible_count,
        "feasible_count": len(valid_cands),
        "A_FIFO": mA,
        "C_SYMBOL_DIVERSE": mC,
        "I_C_NULL_MEAN": mean_ic,
        "P_VAL_C": p_val_c,
        "J1_CLAIRVOYANT": mJ1,
        "OPP_CAPTURE_C": cap_rate_c,
    }


def main():
    lake = DataLake(read_only=True)
    symbols = get_universe_symbols()
    dal = UIDataAccess()
    cost_model = IndianCostModel(default_slippage_bps=3.0)

    all_alphas_df = dal.get_alpha_registry_table()
    proven_df = all_alphas_df[all_alphas_df["status"] == "PROVEN"]
    alpha_ids = proven_df["alpha_id"].tolist()

    print("=" * 140)
    print("ASHVA PCDE BENCHMARK - PHASE 1.5: CAPITAL SENSITIVITY & RIGOROUS ANCHORS")
    print("=" * 140)

    raw_candidates = harvest_raw_candidates(lake, symbols, alpha_ids, dal)

    # 5-Month Evaluation Window (Apr 19 to Sep 18, 2026)
    s_ts = pd.to_datetime("2026-04-19 00:00:00")
    e_ts = pd.to_datetime("2026-09-18 23:59:59")
    window_raw = [c for c in raw_candidates if s_ts <= c["entry_time"] <= e_ts]
    print(f"[*] Evaluation Window: 2026-04-19 to 2026-09-18 (5 Months | {len(window_raw):,} Raw Candidate Signals)\n")

    capital_levels = [500000.0, 1000000.0, 2500000.0, 5000000.0]
    results = []

    for cap in capital_levels:
        print(f"[>] Simulating Capital Scale: Rs {cap:,.0f} (Slot Capital: Rs {cap/4:,.0f})...", flush=True)
        res_cap = run_capital_level(window_raw, cap, cost_model)
        results.append(res_cap)
        mC = res_cap["C_SYMBOL_DIVERSE"]
        mJ1 = res_cap["J1_CLAIRVOYANT"]
        print(f"    • Symbol-Diverse Net: Rs {mC['net_pnl']:+9,.2f} ({mC['roi_pct']:+5.2f}% ROI) | J1 Upper Bound: Rs {mJ1['net_pnl']:+9,.2f} ({mJ1['roi_pct']:+5.2f}% ROI)", flush=True)

    print("\n" + "=" * 145)
    print("CAPITAL SENSITIVITY SCORECARD (4 SLOTS | 50 ALPHAS | 77 STOCKS | 5 MONTHS)")
    print("=" * 145)
    print(f"{'PORTFOLIO CAPITAL':<18} | {'SLOT CAP':<12} | {'INFEASIBLE':<11} | {'TEST A (FIFO)':<18} | {'TEST C (SymDiv)':<18} | {'NULL I-C (Mean)':<18} | {'TEST J1 (Clairvoyant)':<24} | {'J1 ROI %':<10} | {'OPP CAPTURE %':<12}")
    print("-" * 145)

    for r in results:
        cap_str = f"Rs {r['capital']:>10,.0f}"
        slot_str = f"Rs {r['slot_cap']:>8,.0f}"
        infeas_str = f"{r['infeasible_count']:>4d} cands"
        a_str = f"Rs {r['A_FIFO']['net_pnl']:>12,.0f}"
        c_str = f"Rs {r['C_SYMBOL_DIVERSE']['net_pnl']:>12,.0f}"
        ic_str = f"Rs {r['I_C_NULL_MEAN']:>12,.0f}"
        j1_str = f"Rs {r['J1_CLAIRVOYANT']['net_pnl']:>18,.0f}"
        j1_roi_str = f"{r['J1_CLAIRVOYANT']['roi_pct']:>+8.2f}%"
        cap_rate_str = f"{r['OPP_CAPTURE_C']:>10.2f}%"
        print(f"{cap_str:<18} | {slot_str:<12} | {infeas_str:<11} | {a_str:<18} | {c_str:<18} | {ic_str:<18} | {j1_str:<24} | {j1_roi_str:<10} | {cap_rate_str:<12}")

    print("=" * 145)

    print("\n[*] DETAILED CLAIRVOYANT UPPER BOUND (TEST J1) PROFILE ACROSS CAPITAL:")
    for r in results:
        mJ1 = r["J1_CLAIRVOYANT"]
        mA = r["A_FIFO"]
        mC = r["C_SYMBOL_DIVERSE"]
        print(f"\n  • Portfolio Capital: Rs {r['capital']:,.0f} (Slot Capital: Rs {r['slot_cap']:,.0f})")
        print(f"    - J1 Trades Selected:      {mJ1['trades']:d} trades (Win Rate: {mJ1['win_rate']:.1f}%, Profit Factor: {mJ1['profit_factor']:.2f})")
        print(f"    - J1 Gross P&L:            Rs {mJ1['gross_pnl']:+12,.2f}")
        print(f"    - J1 Statutory Costs:      Rs {mJ1['costs']:12,.2f} (Friction Ratio: {mJ1['friction_ratio']:.2f}x)")
        print(f"    - J1 NET P&L:              Rs {mJ1['net_pnl']:+12,.2f} ({mJ1['roi_pct']:+.2f}% Net ROI)")
        print(f"    - FIFO vs Symbol-Diverse:  FIFO Net: Rs {mA['net_pnl']:+10,.2f} vs Symbol-Diverse Net: Rs {mC['net_pnl']:+10,.2f}")


if __name__ == "__main__":
    main()
