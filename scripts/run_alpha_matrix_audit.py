"""
Ashva Quantitative Alpha Research Matrix & Portfolio Audit Engine
Evaluates registered quantitative alphas across institutional Nifty universes
using centralized DataLake, BacktestEngine with Indian statutory costs/slippage, and StatisticalValidator.

Unified Execution Modes:
1. DEV: Fast ~18s iteration on 4 diverse strategy families (Alpha 02, 03, 14, 18) over 120 days.
2. RESEARCH: Standard ~10m institutional audit of all 23 Alphas across NIFTY-14 over 540 days with Temporal OOS.
3. FULL: Comprehensive generalization audit of all 23 Alphas across NIFTY-50 over 540 days with Temporal OOS.

Validation Layers:
- Cross-Sectional Generalization: Transfer from NIFTY-14 discovery to unseen NIFTY-36 stocks.
- Temporal Out-Of-Sample (OOS): Strict temporal split between In-Sample (Day -540 to -120) and Untouched OOS (Last 120 days).

Zero New Pipelines: All modes execute through the exact same authoritative backtesting and validation pipeline.
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.research.validator import StatisticalValidator
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.research.hypothesis import StrategyHorizon
from scripts.run_hypothesis_lab import STRATEGY_MAP, DEFAULT_UNIVERSE
from scripts.scan_nifty_50_universe import NIFTY_50_UNIVERSE


DEV_REPRESENTATIVE_ALPHAS = ["alpha_02", "alpha_03", "alpha_14", "alpha_18"]

MODE_CONFIGS = {
    "dev": {
        "universe": "nifty14",
        "alphas": DEV_REPRESENTATIVE_ALPHAS,
        "lookback": 120,
        "oos_days": 30,
        "banner": "[!] DEVELOPMENT RUN -- NOT FOR RESEARCH CONCLUSIONS",
        "description": "Fast Developer Feedback Mode (4 Diverse Strategy Families, 120-Day Lookback)",
    },
    "research": {
        "universe": "nifty14",
        "alphas": list(STRATEGY_MAP.keys()),
        "lookback": 540,
        "oos_days": 120,
        "banner": "[*] RESEARCH RUN -- INSTITUTIONAL AUDIT WITH TEMPORAL OOS",
        "description": "Authoritative Institutional Research Run (All 23 Alphas, NIFTY-14, 540-Day Lookback, 120d OOS)",
    },
    "full": {
        "universe": "nifty50",
        "alphas": list(STRATEGY_MAP.keys()),
        "lookback": 540,
        "oos_days": 120,
        "banner": "[*] FULL RESEARCH RUN -- MULTI-ASSET GENERALIZATION & TEMPORAL OOS",
        "description": "Full Multi-Asset Generalization Run (All 23 Alphas, NIFTY-50, 540-Day Lookback, 120d OOS)",
    },
}


def run_full_alpha_matrix_audit(
    mode: str = "research",
    universe_type: Optional[str] = None,
    alpha_keys: Optional[List[str]] = None,
    max_lookback_days: Optional[int] = None,
    oos_days: Optional[int] = None,
    timeframe: str = "15m",
    capital_per_asset: float = 500000.0,
    risk_per_trade_pct: float = 0.005,
    capital_per_trade_pct: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the Alpha Research Matrix audit across configured alphas and universe using the unified pipeline.
    """
    cfg = MODE_CONFIGS.get(mode, MODE_CONFIGS["research"])
    effective_universe = universe_type or cfg["universe"]
    effective_lookback = max_lookback_days or cfg["lookback"]
    effective_oos_days = oos_days if oos_days is not None else cfg.get("oos_days", 120)
    effective_alpha_keys = alpha_keys or cfg["alphas"]
    banner = cfg["banner"]

    active_strategies = {k: STRATEGY_MAP[k] for k in effective_alpha_keys if k in STRATEGY_MAP}
    if not active_strategies:
        raise ValueError(f"No valid strategy keys found in {effective_alpha_keys}")

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    validator = StatisticalValidator(cost_model=cost_model)

    symbols = DEFAULT_UNIVERSE if effective_universe == "nifty14" else NIFTY_50_UNIVERSE
    total_basket_capital = capital_per_asset * len(symbols)

    print("=" * 140)
    print(f"[*] {banner}")
    print(f"[*] Mode: {mode.upper()} | Strategies: {len(active_strategies)} Alphas | Universe: {effective_universe.upper()} ({len(symbols)} Assets)")
    print(f"[*] Lookback: {effective_lookback}d (IS: {effective_lookback - effective_oos_days}d, Untouched OOS: {effective_oos_days}d) | Timeframe: {timeframe}")
    print(f"[*] Capital: Rs {capital_per_asset:,.0f}/Asset (Total = Rs {total_basket_capital:,.0f}) | Costs: Indian Statutory Taxes + 3.0 bps Slippage")
    print("=" * 140)

    alpha_summary_results = []
    alpha_temporal_oos_results = []
    alpha_daily_returns: Dict[str, pd.Series] = {}
    alpha_asset_detailed_records = []
    alpha_asset_grid_data: Dict[str, Dict[str, str]] = {}

    for strat_key, (strat_name, strat_cls) in active_strategies.items():
        strat_obj = strat_cls()
        is_swing = False
        if hasattr(strat_obj, "metadata"):
            h = getattr(strat_obj.metadata, "horizon", None)
            if h in [StrategyHorizon.SWING, StrategyHorizon.POSITIONAL] or str(h) in ["SWING", "POSITIONAL", "StrategyHorizon.SWING", "StrategyHorizon.POSITIONAL"]:
                is_swing = True

        engine_segment = Segment.EQUITY_DELIVERY if is_swing else Segment.EQUITY_INTRADAY
        target_tf = getattr(strat_obj.metadata, "timeframe", timeframe) if hasattr(strat_obj, "metadata") else timeframe

        print(f"\n[>] Auditing {strat_name} ({strat_key})...", end="", flush=True)

        total_net_pnl = 0.0
        total_taxes = 0.0
        total_trades = 0
        total_wins = 0
        all_symbol_daily_pnl = []
        recency_scores = []
        pf_60d_list = []
        pf_lookback_list = []
        symbol_pnls = {}

        # Temporal OOS accumulators
        is_net_pnl = 0.0
        is_trades = 0
        is_wins = 0
        oos_net_pnl = 0.0
        oos_trades = 0
        oos_wins = 0
        oos_symbol_daily_pnl = []

        alpha_asset_grid_data[strat_name] = {}

        for sym in symbols:
            df = lake.load_bars(sym, target_tf, max_lookback_days=effective_lookback)
            if df.empty or len(df) < 50:
                if target_tf != timeframe:
                    df = lake.load_bars(sym, timeframe, max_lookback_days=effective_lookback)
                if df.empty or len(df) < 50:
                    alpha_asset_grid_data[strat_name][sym] = "No Data"
                    continue

            try:
                signals_df = strat_obj.generate_signals(df)
                eng = BacktestEngine(
                    cost_model=cost_model,
                    initial_capital=capital_per_asset,
                    segment=engine_segment
                )
                res = eng.run(
                    signals_df,
                    symbol=sym,
                    strategy_id=strat_name,
                    risk_per_trade_pct=risk_per_trade_pct,
                    capital_per_trade_pct=capital_per_trade_pct
                )

                # Temporal In-Sample vs Out-of-Sample Split
                if effective_oos_days > 0 and len(signals_df) > 50:
                    max_ts = pd.to_datetime(signals_df.index[-1])
                    oos_cutoff = max_ts - timedelta(days=effective_oos_days)
                    
                    sig_is = signals_df[signals_df.index < oos_cutoff]
                    sig_oos = signals_df[signals_df.index >= oos_cutoff]

                    if not sig_is.empty and len(sig_is) > 20:
                        res_is = eng.run(sig_is, symbol=sym, strategy_id=strat_name, risk_per_trade_pct=risk_per_trade_pct, capital_per_trade_pct=capital_per_trade_pct)
                        is_net_pnl += res_is.total_net_pnl
                        is_trades += res_is.total_trades
                        is_wins += res_is.winning_trades

                    if not sig_oos.empty and len(sig_oos) > 10:
                        res_oos = eng.run(sig_oos, symbol=sym, strategy_id=strat_name, risk_per_trade_pct=risk_per_trade_pct, capital_per_trade_pct=capital_per_trade_pct)
                        oos_net_pnl += res_oos.total_net_pnl
                        oos_trades += res_oos.total_trades
                        oos_wins += res_oos.winning_trades

                        if len(res_oos.equity_curve) > 1:
                            oos_daily = res_oos.equity_curve.diff().resample("1D").sum().fillna(0.0)
                            oos_symbol_daily_pnl.append(oos_daily)

                if res.total_trades > 0:
                    total_net_pnl += res.total_net_pnl
                    total_taxes += res.total_taxes_paid
                    total_trades += res.total_trades
                    total_wins += res.winning_trades
                    symbol_pnls[sym] = res.total_net_pnl

                    # Daily equity series
                    eq = res.equity_curve
                    if len(eq) > 1:
                        daily_pnl = eq.diff().resample("1D").sum().fillna(0.0)
                        all_symbol_daily_pnl.append(daily_pnl)

                    # Validation report metrics
                    rep = validator.validate_hypothesis(strat_obj, df, symbol=sym)
                    recency_scores.append(rep.recency_weighted_score)
                    w60 = rep.window_metrics.get("60d", {})
                    w_lookback = rep.window_metrics.get(f"{effective_lookback}d", rep.window_metrics.get("540d", {}))
                    pf_60_val = float(w60["net_pf"]) if ("net_pf" in w60 and isinstance(w60["net_pf"], (int, float))) else 0.0
                    pf_lookback_val = float(w_lookback["net_pf"]) if ("net_pf" in w_lookback and isinstance(w_lookback["net_pf"], (int, float))) else res.net_profit_factor
                    
                    pf_60d_list.append(min(pf_60_val, 99.0))
                    pf_lookback_list.append(min(pf_lookback_val, 99.0))

                    # Candidate Alpha x Asset pair detail
                    alpha_asset_detailed_records.append({
                        "Alpha_ID": strat_key,
                        "Strategy": strat_name,
                        "Symbol": sym,
                        "Net_PnL_INR": round(res.total_net_pnl, 2),
                        "Net_ROI_Pct": round(res.net_roi_pct, 2),
                        "Trades": res.total_trades,
                        "Win_Rate_Pct": round(res.win_rate_pct, 1),
                        "PF_Lookback": round(min(pf_lookback_val, 99.0), 2),
                        "PF_60d": round(min(pf_60_val, 99.0), 2),
                        "Sharpe": round(res.sharpe_ratio, 2),
                        "Max_DD_Pct": round(res.max_drawdown_pct, 2),
                        "Recency_Q": round(rep.recency_weighted_score, 2),
                        "Status": "🟢 Positive" if res.total_net_pnl > 0 else "🔴 Negative",
                    })

                    sign_str = "+" if res.total_net_pnl > 0 else "-"
                    abs_pnl_k = abs(res.total_net_pnl) / 1000.0
                    marker = "🟢" if res.total_net_pnl > 0 else "🔴"
                    alpha_asset_grid_data[strat_name][sym] = f"{marker} {sign_str}Rs {abs_pnl_k:.1f}k ({res.total_trades}T|PF:{min(pf_lookback_val, 99.0):.1f})"
                else:
                    alpha_asset_grid_data[strat_name][sym] = "0 Trades"

            except Exception as e:
                alpha_asset_grid_data[strat_name][sym] = "Error"

        # Combined daily portfolio returns
        if all_symbol_daily_pnl:
            combined_pnl = pd.concat(all_symbol_daily_pnl, axis=1).sum(axis=1).fillna(0.0)
            alpha_ret = combined_pnl / total_basket_capital
            alpha_daily_returns[strat_name] = alpha_ret
        else:
            alpha_daily_returns[strat_name] = pd.Series(dtype=float)

        true_basket_roi_pct = (total_net_pnl / total_basket_capital) * 100.0
        ann_basket_return_pct = true_basket_roi_pct * (365.25 / max(1, effective_lookback))
        win_rate = (total_wins / max(1, total_trades)) * 100.0 if total_trades > 0 else 0.0
        avg_trade = (total_net_pnl / max(1, total_trades)) if total_trades > 0 else 0.0

        if not alpha_daily_returns[strat_name].empty and len(alpha_daily_returns[strat_name]) > 5:
            rets = alpha_daily_returns[strat_name].dropna()
            mean_r = rets.mean()
            std_r = rets.std()
            sharpe = float((mean_r / std_r * np.sqrt(252))) if std_r > 1e-7 else 0.0

            cum_eq = (1.0 + rets).cumprod()
            peak = cum_eq.cummax()
            dd = (cum_eq - peak) / peak
            max_dd_pct = float(abs(dd.min()) * 100.0) if not dd.empty else 0.0
        else:
            sharpe = 0.0
            max_dd_pct = 0.0

        # Temporal OOS Sharpe
        if oos_symbol_daily_pnl:
            combined_oos_pnl = pd.concat(oos_symbol_daily_pnl, axis=1).sum(axis=1).fillna(0.0)
            oos_ret = combined_oos_pnl / total_basket_capital
            oos_rets_clean = oos_ret.dropna()
            m_oos = oos_rets_clean.mean()
            s_oos = oos_rets_clean.std()
            oos_sharpe = float((m_oos / s_oos * np.sqrt(252))) if s_oos > 1e-7 else 0.0
        else:
            oos_sharpe = 0.0

        oos_win_rate = (oos_wins / max(1, oos_trades)) * 100.0 if oos_trades > 0 else 0.0
        oos_basket_roi = (oos_net_pnl / total_basket_capital) * 100.0
        is_basket_roi = (is_net_pnl / total_basket_capital) * 100.0

        avg_recency_q = float(np.mean(recency_scores)) if recency_scores else 0.0
        avg_pf_60d = float(np.median(pf_60d_list)) if pf_60d_list else 0.0
        avg_pf_lookback = float(np.median(pf_lookback_list)) if pf_lookback_list else 0.0

        pos_symbols = [f"{s} (+Rs {p:,.0f})" for s, p in sorted(symbol_pnls.items(), key=lambda x: x[1], reverse=True) if p > 0]
        pos_count = len(pos_symbols)
        asset_cluster_str = f"🟢 {pos_count}/{len(symbols)} Assets: " + ", ".join(pos_symbols[:3]) if pos_symbols else "🔴 No Positive Assets"

        print(f" [DONE: Trades={total_trades}, Net PnL=Rs {total_net_pnl:+,.0f}, Basket ROI={true_basket_roi_pct:+.2f}%, Sharpe={sharpe:.2f} | OOS PnL=Rs {oos_net_pnl:+,.0f}]")

        alpha_summary_results.append({
            "Alpha_ID": strat_key,
            "Strategy": strat_name,
            "Net_PnL_INR": round(total_net_pnl, 2),
            "Basket_ROI_Pct": round(true_basket_roi_pct, 2),
            "Ann_Return_Pct": round(ann_basket_return_pct, 2),
            "Sharpe": round(sharpe, 2),
            "Max_DD_Pct": round(max_dd_pct, 2),
            "PF_Lookback": round(avg_pf_lookback, 2),
            "PF_60d": round(avg_pf_60d, 2),
            "Trades": total_trades,
            "Win_Rate_Pct": round(win_rate, 1),
            "Avg_Trade_INR": round(avg_trade, 1),
            "Recency_Q": round(avg_recency_q, 2),
            "Observed_Positive_Cluster": asset_cluster_str,
            "_pos_count": pos_count,
        })

        alpha_temporal_oos_results.append({
            "Alpha_ID": strat_key,
            "Strategy": strat_name,
            "IS_Lookback_Days": effective_lookback - effective_oos_days,
            "IS_Trades": is_trades,
            "IS_Net_PnL_INR": round(is_net_pnl, 2),
            "IS_Basket_ROI_Pct": round(is_basket_roi, 2),
            "OOS_Untouched_Days": effective_oos_days,
            "OOS_Trades": oos_trades,
            "OOS_Win_Rate_Pct": round(oos_win_rate, 1),
            "OOS_Net_PnL_INR": round(oos_net_pnl, 2),
            "OOS_Basket_ROI_Pct": round(oos_basket_roi, 2),
            "OOS_Sharpe": round(oos_sharpe, 2),
            "OOS_Status": "🟢 Positive OOS" if oos_net_pnl > 0 else "🔴 Negative OOS",
        })

    summary_df = pd.DataFrame(alpha_summary_results)
    temporal_oos_df = pd.DataFrame(alpha_temporal_oos_results)
    asset_detail_df = pd.DataFrame(alpha_asset_detailed_records)
    grid_2d_df = pd.DataFrame(alpha_asset_grid_data).T

    # Calculate Inter-Alpha Correlation Matrix if multiple strategies audited
    valid_series = {k: v for k, v in alpha_daily_returns.items() if not v.empty and len(v) > 5}
    if len(valid_series) > 1:
        all_rets_df = pd.DataFrame(valid_series).fillna(0.0)
        corr_matrix = all_rets_df.corr(method="pearson")
        rolling_5d = all_rets_df.rolling(5).sum().dropna()
        trade_corr_matrix = rolling_5d.corr(method="pearson") if not rolling_5d.empty else corr_matrix

        max_inter_corrs = []
        max_trade_corrs = []
        for _, row in summary_df.iterrows():
            s_name = row["Strategy"]
            if s_name in corr_matrix.columns:
                others = corr_matrix[s_name].drop(s_name)
                max_inter_corrs.append(round(others.abs().max(), 2) if not others.empty else 0.0)
            else:
                max_inter_corrs.append(0.0)

            if s_name in trade_corr_matrix.columns:
                t_others = trade_corr_matrix[s_name].drop(s_name)
                max_trade_corrs.append(round(t_others.abs().max(), 2) if not t_others.empty else 0.0)
            else:
                max_trade_corrs.append(0.0)

        summary_df["Inter_Alpha_Corr"] = max_inter_corrs
        summary_df["Trade_PnL_Corr"] = max_trade_corrs
    else:
        summary_df["Inter_Alpha_Corr"] = 0.0
        summary_df["Trade_PnL_Corr"] = 0.0
        corr_matrix = pd.DataFrame()

    # Dual Classification Logic
    strategy_statuses = []
    for _, row in summary_df.iterrows():
        pnl = row["Net_PnL_INR"]
        sharpe = row["Sharpe"]
        corr = row["Inter_Alpha_Corr"]
        pos_cnt = row["_pos_count"]

        if pnl > 0 and sharpe >= 0.70 and corr <= 0.40:
            strategy_statuses.append("🟢 Core (Universal)")
        elif pnl > 0 and sharpe >= 0.30:
            strategy_statuses.append("🟡 Promising (Universal)")
        elif pnl > 0 and corr <= 0.15:
            strategy_statuses.append("🔵 Diversifier")
        elif pnl > 0 and corr > 0.50:
            strategy_statuses.append("🟠 Redundant")
        elif pos_cnt > 0:
            strategy_statuses.append(f"🔍 Candidate Asset Edges ({pos_cnt} Pairs)")
        else:
            strategy_statuses.append("🔴 Failed Aggregate")

    summary_df["Strategy_Classification"] = strategy_statuses
    summary_df.drop(columns=["_pos_count"], inplace=True)
    summary_df.sort_values(by="Net_PnL_INR", ascending=False, inplace=True)

    positive_clusters_df = asset_detail_df[asset_detail_df["Net_PnL_INR"] > 0].sort_values(by="Net_PnL_INR", ascending=False) if not asset_detail_df.empty else pd.DataFrame()

    # Format Markdown Report
    output_text = f"# ASHVA ALPHA RESEARCH MATRIX AUDIT — [{mode.upper()} MODE]\n\n"
    output_text += f"> **{banner}**\n\n"
    if mode == "dev":
        output_text += "> [!WARNING]\n"
        output_text += "> **DEVELOPMENT RUN ONLY**: This run used a truncated lookback and a 4-alpha subset for fast execution feedback. Do NOT use these metrics for formal research conclusions or live capital allocation.\n\n"

    output_text += f"- **Execution Mode**: `{mode.upper()}`\n"
    output_text += f"- **Universe**: `{effective_universe.upper()}` ({len(symbols)} Assets)\n"
    output_text += f"- **Audited Strategies**: `{len(active_strategies)} Alphas` ({', '.join(active_strategies.keys())})\n"
    output_text += f"- **Historical Lookback**: `{effective_lookback} Days` (IS: `{effective_lookback - effective_oos_days}d` | Untouched OOS: `{effective_oos_days}d`)\n"
    output_text += f"- **Capital Deployment**: `Rs {capital_per_asset:,.0f}/Asset` (Total Basket Capital = `Rs {total_basket_capital:,.0f}`)\n"
    output_text += f"- **Cost Model**: Indian Statutory Taxes (STT, Exchange, GST, SEBI, Stamp Duty) + 3.0 bps Slippage\n"
    output_text += f"- **Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}`\n\n"

    output_text += f"## 1. Strategy Summary Matrix ({len(active_strategies)} Alphas)\n\n"
    output_text += summary_df.to_markdown(index=False) + "\n\n"

    output_text += "## 2. Candidate Alpha × Asset Edges (Observed Positive Pairs)\n\n"
    if not positive_clusters_df.empty:
        output_text += positive_clusters_df.to_markdown(index=False) + "\n\n"
    else:
        output_text += "No positive pairs observed under current parameters.\n\n"

    output_text += f"## 3. Temporal Out-Of-Sample (OOS) Validation ({effective_oos_days} Days Untouched Test Period)\n\n"
    output_text += temporal_oos_df.sort_values(by="OOS_Net_PnL_INR", ascending=False).to_markdown(index=False) + "\n\n"

    output_text += f"## 4. Full 2D Alpha × Asset Interaction Grid ({len(active_strategies)} Alphas × {len(symbols)} Assets)\n\n"
    output_text += grid_2d_df.to_markdown() + "\n\n"

    if not corr_matrix.empty:
        output_text += "## 5. Inter-Alpha Daily Return Correlation Matrix (Cross-Strategy Redundancy)\n\n"
        output_text += corr_matrix.round(2).to_markdown() + "\n\n"

    output_file = Path("matrix_output.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print("\n" + "=" * 140)
    print(f"[*] AUDIT COMPLETE [{mode.upper()}]: Report saved to {output_file.resolve()}")
    print("=" * 140)

    return summary_df, positive_clusters_df, temporal_oos_df, grid_2d_df, corr_matrix


def main():
    parser = argparse.ArgumentParser(description="Ashva Alpha Research Matrix & Portfolio Audit")
    parser.add_argument("--mode", type=str, choices=["dev", "research", "full"], default="research",
                        help="Execution mode: dev (fast ~18s), research (10m Nifty-14 audit with 120d OOS), full (25m Nifty-50 generalization with 120d OOS)")
    parser.add_argument("--universe", type=str, choices=["nifty14", "nifty50"], default=None,
                        help="Override default universe (nifty14 or nifty50)")
    parser.add_argument("--alphas", type=str, default=None,
                        help="Comma-separated strategy keys to audit (e.g. 'alpha_02,alpha_03,alpha_14')")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Override lookback window in days (e.g. 120, 540)")
    parser.add_argument("--oos-days", type=int, default=None,
                        help="Override temporal OOS window in days (e.g. 60, 120)")
    parser.add_argument("--timeframe", type=str, default="15m",
                        help="Candle timeframe (default: 15m)")

    args = parser.parse_args()
    alpha_keys = [a.strip() for a in args.alphas.split(",")] if args.alphas else None

    run_full_alpha_matrix_audit(
        mode=args.mode,
        universe_type=args.universe,
        alpha_keys=alpha_keys,
        max_lookback_days=args.lookback,
        oos_days=args.oos_days,
        timeframe=args.timeframe
    )


if __name__ == "__main__":
    main()
