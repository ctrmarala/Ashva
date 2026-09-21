"""
Harvest and Cache Immutable Raw Candidate Signals for PCDE Research.
Saves raw candidates to data_lake/pcde_raw_candidates.parquet.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from src.data.data_lake import DataLake
from src.core.universe_manager import get_universe_symbols
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.registry import get_strategy_by_name
from src.ui.data_access import UIDataAccess


def main():
    lake = DataLake(read_only=True)
    symbols = get_universe_symbols()
    dal = UIDataAccess()

    all_alphas_df = dal.get_alpha_registry_table()
    proven_df = all_alphas_df[all_alphas_df["status"] == "PROVEN"]
    alpha_ids = proven_df["alpha_id"].tolist()

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

    print(f"[*] Harvesting {len(alpha_ids)} proven alphas across {len(symbols)} symbols...", flush=True)

    cost_model = IndianCostModel(default_slippage_bps=3.0)
    engine = BacktestEngine(
        cost_model=cost_model,
        initial_capital=500000.0,
        segment=Segment.EQUITY_INTRADAY,
        use_1m_intrabar=True,
        data_lake=lake,
    )

    all_candidates = []
    df_cache = {}

    for idx, alpha_id in enumerate(alpha_ids, 1):
        strat_cls = get_strategy_by_name(alpha_id)
        if not strat_cls:
            print(f"[!] Strategy {alpha_id} not found in registry!", flush=True)
            continue

        detail = dal.get_alpha_detail(alpha_id)
        optimal_tf = detail.get("timeframe") or "15m"
        strat = strat_cls({"timeframe": optimal_tf})

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
        print(f"  [{idx:02d}/{len(alpha_ids)}] {alpha_id:<12} | Candidates so far: {len(all_candidates):,}", flush=True)

    df_res = pd.DataFrame(all_candidates)
    print(f"\n[+] Total raw candidate signals harvested: {len(df_res):,}", flush=True)

    out_path = ROOT_DIR / "data_lake" / "pcde_raw_candidates.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_parquet(out_path, index=False)
    print(f"[+] Successfully cached {len(df_res)} candidates to {out_path}", flush=True)


if __name__ == "__main__":
    main()
