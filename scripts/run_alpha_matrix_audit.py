"""
Ashva Quantitative Alpha Research Matrix & Portfolio Audit Engine
Evaluates all registered quantitative alphas (01 to 23) across the institutional Nifty universe
using centralized DataLake, BacktestEngine with Indian statutory costs/slippage, and StatisticalValidator.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
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


def run_full_alpha_matrix_audit(
    universe_type: str = "nifty14",
    max_lookback_days: int = 540,
    timeframe: str = "15m",
    capital_per_alpha: float = 500000.0,
    risk_per_trade_pct: float = 0.005,
    capital_per_trade_pct: float = 0.25,
) -> pd.DataFrame:
    """
    Executes the comprehensive 12-metric Alpha Research Matrix audit across all registered alphas.
    """
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    validator = StatisticalValidator(cost_model=cost_model)

    symbols = DEFAULT_UNIVERSE if universe_type == "nifty14" else NIFTY_50_UNIVERSE

    print("=" * 135)
    print(f"[*] ASHVA MASTER ALPHA RESEARCH MATRIX AUDIT (ALL {len(STRATEGY_MAP)} REGISTERED ALPHAS)")
    print(f"[*] Universe: {universe_type.upper()} ({len(symbols)} Assets) | Lookback: {max_lookback_days}d | Timeframe: {timeframe}")
    print(f"[*] Regulatory Costs: STT, Exchange, GST, SEBI, Stamp Duty | Slippage: 3.0 bps")
    print("=" * 135)

    alpha_results = []
    alpha_daily_returns: Dict[str, pd.Series] = {}

    for strat_key, (strat_name, strat_cls) in STRATEGY_MAP.items():
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
        pf_540d_list = []
        symbol_pnls = {}

        for sym in symbols:
            df = lake.load_bars(sym, target_tf, max_lookback_days=max_lookback_days)
            if df.empty or len(df) < 50:
                if target_tf != timeframe:
                    df = lake.load_bars(sym, timeframe, max_lookback_days=max_lookback_days)
                if df.empty or len(df) < 50:
                    continue

            try:
                signals_df = strat_obj.generate_signals(df)
                eng = BacktestEngine(
                    cost_model=cost_model,
                    initial_capital=capital_per_alpha,
                    segment=engine_segment
                )
                res = eng.run(
                    signals_df,
                    symbol=sym,
                    strategy_id=strat_name,
                    risk_per_trade_pct=risk_per_trade_pct,
                    capital_per_trade_pct=capital_per_trade_pct
                )

                if res.total_trades > 0:
                    total_net_pnl += res.total_net_pnl
                    total_taxes += res.total_taxes_paid
                    total_trades += res.total_trades
                    total_wins += res.winning_trades
                    symbol_pnls[sym] = res.total_net_pnl

                    # Extract daily equity / PnL series
                    eq = res.equity_curve
                    if len(eq) > 1:
                        daily_pnl = eq.diff().resample("1D").sum().fillna(0.0)
                        all_symbol_daily_pnl.append(daily_pnl)

                    # Validation report metrics
                    rep = validator.validate_hypothesis(strat_obj, df, symbol=sym)
                    recency_scores.append(rep.recency_weighted_score)
                    w60 = rep.window_metrics.get("60d", {})
                    w540 = rep.window_metrics.get("540d", {})
                    if "net_pf" in w60 and isinstance(w60["net_pf"], (int, float)):
                        pf_60d_list.append(min(float(w60["net_pf"]), 99.0))
                    if "net_pf" in w540 and isinstance(w540["net_pf"], (int, float)):
                        pf_540d_list.append(min(float(w540["net_pf"]), 99.0))

            except Exception as e:
                pass

        # Aggregate daily portfolio returns for this Alpha
        if all_symbol_daily_pnl:
            combined_pnl = pd.concat(all_symbol_daily_pnl, axis=1).sum(axis=1).fillna(0.0)
            alpha_ret = combined_pnl / capital_per_alpha
            alpha_daily_returns[strat_name] = alpha_ret
        else:
            alpha_daily_returns[strat_name] = pd.Series(dtype=float)

        # Calculate metrics
        net_roi = (total_net_pnl / capital_per_alpha) * 100.0
        days_span = max_lookback_days
        ann_return = net_roi * (365.25 / max(1, days_span))
        win_rate = (total_wins / max(1, total_trades)) * 100.0 if total_trades > 0 else 0.0
        avg_trade = (total_net_pnl / max(1, total_trades)) if total_trades > 0 else 0.0

        # Sharpe & Drawdown from aggregated daily return series
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

        avg_recency_q = float(np.mean(recency_scores)) if recency_scores else 0.0
        avg_pf_60d = float(np.median(pf_60d_list)) if pf_60d_list else 0.0
        avg_pf_540d = float(np.median(pf_540d_list)) if pf_540d_list else 0.0

        # Top Performing Symbols for this Alpha
        pos_symbols = [f"{s} (+Rs {p:,.0f})" for s, p in sorted(symbol_pnls.items(), key=lambda x: x[1], reverse=True) if p > 0]
        top_syms_str = ", ".join(pos_symbols[:3]) if pos_symbols else "None"

        print(f" [DONE: Trades={total_trades}, Net PnL=Rs {total_net_pnl:+,.0f}, Sharpe={sharpe:.2f}, WinRate={win_rate:.1f}%]")

        alpha_results.append({
            "Alpha_ID": strat_key,
            "Strategy": strat_name,
            "Net_ROI_Pct": round(net_roi, 2),
            "Ann_Return_Pct": round(ann_return, 2),
            "Sharpe": round(sharpe, 2),
            "Max_DD_Pct": round(max_dd_pct, 2),
            "PF_540d": round(avg_pf_540d, 2),
            "Trades": total_trades,
            "Win_Rate_Pct": round(win_rate, 1),
            "Avg_Trade_INR": round(avg_trade, 1),
            "Recency_Q": round(avg_recency_q, 2),
            "PF_60d": round(avg_pf_60d, 2),
            "Net_PnL_INR": round(total_net_pnl, 2),
            "Top_Profitable_Assets": top_syms_str,
        })

    matrix_df = pd.DataFrame(alpha_results)

    # Calculate Inter-Alpha Correlation Matrix
    valid_series = {k: v for k, v in alpha_daily_returns.items() if not v.empty and len(v) > 10}
    if valid_series:
        all_rets_df = pd.DataFrame(valid_series).fillna(0.0)
        corr_matrix = all_rets_df.corr(method="pearson")

        # 5-day rolling trade correlation matrix
        rolling_5d = all_rets_df.rolling(5).sum().dropna()
        trade_corr_matrix = rolling_5d.corr(method="pearson") if not rolling_5d.empty else corr_matrix

        max_inter_corrs = []
        max_trade_corrs = []
        for _, row in matrix_df.iterrows():
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

        matrix_df["Inter_Alpha_Corr"] = max_inter_corrs
        matrix_df["Trade_PnL_Corr"] = max_trade_corrs
    else:
        matrix_df["Inter_Alpha_Corr"] = 0.0
        matrix_df["Trade_PnL_Corr"] = 0.0
        corr_matrix = pd.DataFrame()

    # Classification logic (Question A Intrinsic Quality vs Question B Portfolio Utility)
    statuses = []
    for _, row in matrix_df.iterrows():
        pnl = row["Net_PnL_INR"]
        sharpe = row["Sharpe"]
        corr = row["Inter_Alpha_Corr"]
        rec_q = row["Recency_Q"]
        trades = row["Trades"]
        has_pos_assets = row["Top_Profitable_Assets"] != "None"

        if pnl > 0 and sharpe >= 0.70 and corr <= 0.40 and trades >= 10:
            statuses.append("🟢 Core")
        elif (pnl > 0 or has_pos_assets) and (rec_q > 0.10 or sharpe >= 0.30 or trades > 15):
            statuses.append("🟡 Promising")
        elif pnl > 0 and corr <= 0.15:
            statuses.append("🔵 Diversifier")
        elif pnl > 0 and corr > 0.50:
            statuses.append("🟠 Redundant")
        else:
            statuses.append("🔴 Failed")

    matrix_df["Classification"] = statuses

    # Sort matrix by Net PnL descending
    matrix_df.sort_values(by="Net_PnL_INR", ascending=False, inplace=True)

    # Format Markdown Report
    output_text = "# ASHVA INSTITUTIONAL ALPHA RESEARCH MATRIX AUDIT\n\n"
    output_text += f"**Universe**: {universe_type.upper()} ({len(symbols)} Assets) | **Historical Lookback**: {max_lookback_days} Days | **Timeframe**: {timeframe}\n"
    output_text += f"**Cost Model**: Indian Statutory Costs (STT, Exchange, GST, SEBI, Stamp Duty) + 3.0 bps Slippage\n\n"
    output_text += "## 1. Complete Alpha Research Matrix (All 23 Alphas)\n\n"
    output_text += matrix_df.to_markdown(index=False) + "\n\n"

    if not corr_matrix.empty:
        output_text += "## 2. Inter-Alpha Daily Return Correlation Matrix (Cross-Strategy Redundancy)\n\n"
        output_text += corr_matrix.round(2).to_markdown() + "\n\n"

    # Save to matrix_output.md
    output_file = Path("matrix_output.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print("\n" + "=" * 135)
    print(f"[*] AUDIT COMPLETE: Full 23-Alpha Matrix saved to {output_file.resolve()}")
    print("=" * 135)

    return matrix_df


def main():
    parser = argparse.ArgumentParser(description="Ashva Alpha Research Matrix & Portfolio Audit")
    parser.add_argument("--universe", type=str, choices=["nifty14", "nifty50"], default="nifty14", help="Asset universe")
    parser.add_argument("--lookback", type=int, default=540, help="Lookback window in days (default: 540)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe (default: 15m)")

    args = parser.parse_args()
    run_full_alpha_matrix_audit(
        universe_type=args.universe,
        max_lookback_days=args.lookback,
        timeframe=args.timeframe
    )


if __name__ == "__main__":
    main()
