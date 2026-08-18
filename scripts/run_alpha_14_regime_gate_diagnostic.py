"""
Ashva Quantitative Alpha 14 Regime Gate Diagnostic Engine
Tests Macro Regime Gate (Gate A) and Asset-Level Multi-Day Trend Gate (Gate B)
independently and in combination on Alpha 14 to determine whether ex-ante market conditions
can eliminate negative drag from choppy regimes without starving opportunities.

Protocols:
1. Baseline: Unfiltered Alpha 14 (Frozen Parameters)
2. Gate A Only: Macro Market Breadth / Trend Alignment
3. Gate B Only: Asset-Level Multi-Day Trend Alignment
4. Gate A + Gate B: Combined Condition Gate

Evaluates:
- Opportunities Retained (% of total baseline trades)
- Full 540d Economics (Net PnL after Indian costs + 3.0 bps slippage, PF, Win Rate, Sharpe, MaxDD)
- Window 1 Preservation (Does it preserve edge in favorable expansion periods?)
- Windows 3 & 4 Drag Elimination (Does it eliminate losses in choppy regimes?)
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.strategies.alpha_14_gap_momentum_drift import Alpha14GapMomentumDrift
from src.features.indicators import TechnicalIndicators as TI
from scripts.run_hypothesis_lab import DEFAULT_UNIVERSE
from scripts.scan_nifty_50_universe import NIFTY_50_UNIVERSE


def build_market_breadth_series(raw_bars: Dict[str, pd.DataFrame], sma_period: int = 20) -> pd.Series:
    """
    Computes daily market breadth (% of universe stocks closing above their daily 20-day SMA).
    Zero look-ahead: Shifted by 1 day so that at 09:15 AM on day d, only day d-1 close is used.
    """
    daily_closes = {}
    for sym, df in raw_bars.items():
        if df.empty:
            continue
        d_close = df["close"].resample("1D").last().dropna()
        if len(d_close) >= sma_period:
            d_sma = d_close.rolling(window=sma_period).mean()
            # 1.0 if close > sma, else 0.0
            above_sma = (d_close > d_sma).astype(float)
            daily_closes[sym] = above_sma

    if not daily_closes:
        return pd.Series(dtype=float)

    breadth_df = pd.DataFrame(daily_closes)
    # Average breadth across universe (range: 0.0 to 1.0)
    breadth_pct = breadth_df.mean(axis=1)
    # Shift by 1 day so day d uses day d-1 breadth
    prior_breadth = breadth_pct.shift(1).dropna()
    return prior_breadth


def generate_gated_signals(
    df: pd.DataFrame,
    symbol: str,
    market_breadth_daily: pd.Series,
    gate_mode: str = "baseline",  # "baseline", "gate_a", "gate_b", "gate_ab"
    breadth_threshold: float = 0.50,
    asset_sma_period: int = 20,
) -> pd.DataFrame:
    """
    Generates Alpha 14 signals with optional ex-ante regime gates applied strictly at decision time (09:30 AM).
    """
    strat = Alpha14GapMomentumDrift()
    base_sig = strat.generate_signals(df)

    if gate_mode == "baseline" or base_sig.empty:
        return base_sig

    out_sig = base_sig.copy()

    # Precompute daily stock 20-day SMA (shifted by 1 day for zero lookahead)
    d_close = df["close"].resample("1D").last().dropna()
    d_sma = d_close.rolling(window=asset_sma_period).mean()
    prior_d_close = d_close.shift(1)
    prior_d_sma = d_sma.shift(1)
    stock_trend_bullish = (prior_d_close > prior_d_sma)  # True = Bullish, False = Bearish

    # Map daily gates to intraday timestamps
    dates = pd.Series(out_sig.index.date, index=out_sig.index)
    date_to_breadth = market_breadth_daily.to_dict()
    date_to_stock_trend = stock_trend_bullish.to_dict()

    for i in range(len(out_sig)):
        sig_val = out_sig["signal"].iloc[i]
        if sig_val != 0.0:
            d_key = pd.to_datetime(dates.iloc[i])

            # Gate A: Macro Market Regime Gate
            # - Long allowed only if prior-day Market Breadth >= 50%
            # - Short allowed only if prior-day Market Breadth < 50%
            if gate_mode in ["gate_a", "gate_ab"]:
                m_breadth = date_to_breadth.get(d_key, np.nan)
                if not np.isnan(m_breadth):
                    if sig_val > 0 and m_breadth < breadth_threshold:
                        out_sig.iloc[i, out_sig.columns.get_loc("signal")] = 0.0
                        out_sig.iloc[i, out_sig.columns.get_loc("rationale")] = "FILTERED_BY_GATE_A_MACRO_BREADTH"
                    elif sig_val < 0 and m_breadth >= breadth_threshold:
                        out_sig.iloc[i, out_sig.columns.get_loc("signal")] = 0.0
                        out_sig.iloc[i, out_sig.columns.get_loc("rationale")] = "FILTERED_BY_GATE_A_MACRO_BREADTH"

            # Gate B: Asset-Level Multi-Day Trend Gate
            # - Long allowed only if stock's prior-day close > 20-day SMA
            # - Short allowed only if stock's prior-day close < 20-day SMA
            if gate_mode in ["gate_b", "gate_ab"]:
                is_bullish = date_to_stock_trend.get(d_key, np.nan)
                if not np.isnan(is_bullish):
                    if sig_val > 0 and not is_bullish:
                        out_sig.iloc[i, out_sig.columns.get_loc("signal")] = 0.0
                        out_sig.iloc[i, out_sig.columns.get_loc("rationale")] = "FILTERED_BY_GATE_B_ASSET_TREND"
                    elif sig_val < 0 and is_bullish:
                        out_sig.iloc[i, out_sig.columns.get_loc("signal")] = 0.0
                        out_sig.iloc[i, out_sig.columns.get_loc("rationale")] = "FILTERED_BY_GATE_B_ASSET_TREND"

    return out_sig


def evaluate_configuration_across_windows(
    universe_type: str = "nifty14",
    total_lookback_days: int = 540,
    n_windows: int = 5,
    capital_per_asset: float = 500000.0,
    risk_per_trade_pct: float = 0.005,
    capital_per_trade_pct: float = 0.25,
) -> pd.DataFrame:
    """
    Evaluates Baseline, Gate A, Gate B, and Gate A+B across 5 chronological windows and full 540d.
    """
    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    symbols = DEFAULT_UNIVERSE if universe_type == "nifty14" else NIFTY_50_UNIVERSE
    total_basket_capital = capital_per_asset * len(symbols)
    window_days = total_lookback_days // n_windows

    print("=" * 140)
    print(f"[*] ASHVA ALPHA 14 REGIME GATE DIAGNOSTIC & COMPARATIVE ANALYSIS")
    print(f"[*] Universe: {universe_type.upper()} ({len(symbols)} Assets) | Lookback: {total_lookback_days} Days")
    print(f"[*] Testing: [1] Baseline -> [2] Gate A (Macro) -> [3] Gate B (Asset Trend) -> [4] Gate A+B (Combined)")
    print(f"[*] Costs: Indian Statutory Taxes + 3.0 bps Slippage | Capital: Rs {capital_per_asset:,.0f}/Asset")
    print("=" * 140)

    # 1. Load raw bars for all symbols
    raw_bars = {}
    all_timestamps = []
    for sym in symbols:
        df = lake.load_bars(sym, "15m", max_lookback_days=total_lookback_days)
        if not df.empty and len(df) > 50:
            raw_bars[sym] = df
            all_timestamps.extend(df.index.tolist())

    max_global_ts = max(all_timestamps)
    min_global_ts = min(all_timestamps)

    # 2. Build Market Breadth Series
    market_breadth = build_market_breadth_series(raw_bars, sma_period=20)

    # 3. Build window boundaries
    window_boundaries = []
    for w_idx in range(n_windows):
        start_day_offset = total_lookback_days - (w_idx * window_days)
        end_day_offset = total_lookback_days - ((w_idx + 1) * window_days)
        w_start_ts = max_global_ts - timedelta(days=start_day_offset)
        w_end_ts = max_global_ts - timedelta(days=end_day_offset)
        if w_idx == 0:
            w_start_ts = min_global_ts
        if w_idx == n_windows - 1:
            w_end_ts = max_global_ts + timedelta(hours=1)
        window_boundaries.append((f"W{w_idx+1}", w_start_ts, w_end_ts))

    gates_to_test = [
        ("1. Baseline (No Gates)", "baseline"),
        ("2. Gate A Only (Macro Regime)", "gate_a"),
        ("3. Gate B Only (Asset Trend)", "gate_b"),
        ("4. Gate A + Gate B (Combined)", "gate_ab"),
    ]

    gate_results = []
    baseline_total_trades = 0

    for gate_label, gate_mode in gates_to_test:
        print(f"\n[>] Testing {gate_label}...", end="", flush=True)

        # Generate signals for this gate mode across all assets
        gated_signals = {}
        for sym, df in raw_bars.items():
            sig_df = generate_gated_signals(df, sym, market_breadth, gate_mode=gate_mode)
            gated_signals[sym] = sig_df

        # Run Full 540d Backtest
        full_net_pnl = 0.0
        full_gross_profit = 0.0
        full_gross_loss = 0.0
        full_trades = 0
        full_wins = 0
        full_daily_pnl = []

        window_metrics = {}

        for w_label, w_start, w_end in window_boundaries:
            w_trades = 0
            w_wins = 0
            w_net_pnl = 0.0
            w_gp = 0.0
            w_gl = 0.0

            for sym, sig_df in gated_signals.items():
                w_sig = sig_df[(sig_df.index >= w_start) & (sig_df.index < w_end)]
                if w_sig.empty or len(w_sig) < 10:
                    continue

                eng = BacktestEngine(cost_model=cost_model, initial_capital=capital_per_asset, segment=Segment.EQUITY_INTRADAY)
                res = eng.run(w_sig, symbol=sym, strategy_id="ALPHA_14", risk_per_trade_pct=risk_per_trade_pct, capital_per_trade_pct=capital_per_trade_pct)

                if res.total_trades > 0:
                    w_trades += res.total_trades
                    w_wins += res.winning_trades
                    w_net_pnl += res.total_net_pnl
                    w_gp += sum(t.gross_pnl for t in res.trade_list if t.gross_pnl > 0)
                    w_gl += sum(abs(t.gross_pnl) for t in res.trade_list if t.gross_pnl < 0)

                    if len(res.equity_curve) > 1:
                        d_pnl = res.equity_curve.diff().resample("1D").sum().fillna(0.0)
                        full_daily_pnl.append(d_pnl)

            w_pf = (w_gp / max(1.0, w_gl)) if w_gl > 0 else (99.0 if w_gp > 0 else 0.0)
            window_metrics[w_label] = {
                "trades": w_trades,
                "net_pnl": w_net_pnl,
                "pf": min(w_pf, 99.0),
                "win_rate": (w_wins / max(1, w_trades)) * 100.0 if w_trades > 0 else 0.0
            }

            full_trades += w_trades
            full_wins += w_wins
            full_net_pnl += w_net_pnl
            full_gross_profit += w_gp
            full_gross_loss += w_gl

        if gate_mode == "baseline":
            baseline_total_trades = full_trades

        retained_pct = (full_trades / max(1, baseline_total_trades)) * 100.0 if baseline_total_trades > 0 else 100.0
        win_rate = (full_wins / max(1, full_trades)) * 100.0 if full_trades > 0 else 0.0
        net_pf = (full_gross_profit / max(1.0, full_gross_loss)) if full_gross_loss > 0 else (99.0 if full_gross_profit > 0 else 0.0)
        basket_roi_pct = (full_net_pnl / total_basket_capital) * 100.0

        if full_daily_pnl:
            comb = pd.concat(full_daily_pnl, axis=1).sum(axis=1).fillna(0.0)
            rets = (comb / total_basket_capital).dropna()
            m_r = rets.mean()
            s_r = rets.std()
            sharpe = float((m_r / s_r * np.sqrt(252))) if s_r > 1e-7 else 0.0

            cum = (1.0 + rets).cumprod()
            pk = cum.cummax()
            dd = (cum - pk) / pk
            max_dd_pct = float(abs(dd.min()) * 100.0) if not dd.empty else 0.0
        else:
            sharpe = 0.0
            max_dd_pct = 0.0

        w1 = window_metrics.get("W1", {})
        w2 = window_metrics.get("W2", {})
        w3 = window_metrics.get("W3", {})
        w4 = window_metrics.get("W4", {})
        w5 = window_metrics.get("W5", {})

        w3_4_pnl = w3.get("net_pnl", 0.0) + w4.get("net_pnl", 0.0)
        w3_4_trades = w3.get("trades", 0) + w4.get("trades", 0)

        print(f" [DONE: Trades={full_trades} ({retained_pct:.1f}%), PnL=Rs {full_net_pnl:+,.0f}, PF={net_pf:.2f}, Sharpe={sharpe:.2f} | W1={w1.get('net_pnl', 0):+,.0f}, W3+W4={w3_4_pnl:+,.0f}]")

        gate_results.append({
            "Configuration": gate_label,
            "Trades_540d": full_trades,
            "Retained_Pct": round(retained_pct, 1),
            "Net_PnL_540d_INR": round(full_net_pnl, 2),
            "Basket_ROI_Pct": round(basket_roi_pct, 2),
            "Net_PF": round(min(net_pf, 99.0), 2),
            "Win_Rate_Pct": round(win_rate, 1),
            "Sharpe": round(sharpe, 2),
            "Max_DD_Pct": round(max_dd_pct, 2),
            "W1_Favorable_PnL": round(w1.get("net_pnl", 0.0), 2),
            "W1_PF": round(w1.get("pf", 0.0), 2),
            "W2_PnL": round(w2.get("net_pnl", 0.0), 2),
            "W3_W4_Choppy_PnL": round(w3_4_pnl, 2),
            "W3_W4_Trades": w3_4_trades,
            "W5_OOS_PnL": round(w5.get("net_pnl", 0.0), 2),
            "W5_PF": round(w5.get("pf", 0.0), 2),
        })

    report_df = pd.DataFrame(gate_results)

    # Format Markdown Report
    out_md = f"# ASHVA ALPHA 14 REGIME GATE DIAGNOSTIC REPORT\n\n"
    out_md += f"> **Strategy**: `ALPHA_14_GAP_MOMENTUM_DRIFT` (Frozen Execution & Sizing Parameters)\n"
    out_md += f"> **Universe**: `{universe_type.upper()}` ({len(symbols)} Assets) | **Lookback**: `{total_lookback_days} Days`\n"
    out_md += f"> **Cost Engine**: Indian Statutory Taxes + 3.0 bps Slippage\n\n"

    out_md += "## 1. Comparative Gate Performance Matrix (Baseline vs Gate A vs Gate B vs Gate A+B)\n\n"
    out_md += report_df.to_markdown(index=False) + "\n\n"

    out_md += "## 2. Key Diagnostic Findings\n\n"
    out_md += "- **Gate A (Macro Regime Gate)**: Requires prior-day Nifty Market Breadth (% of stocks > 20d SMA) >= 50% for Longs and < 50% for Shorts.\n"
    out_md += "- **Gate B (Asset Multi-Day Trend Gate)**: Requires the target stock's prior-day close to be > 20d SMA for Longs and < 20d SMA for Shorts.\n"
    out_md += "- **Gate A+B (Combined Gate)**: Both macro and micro trend alignment required at 09:30 AM.\n\n"

    output_path = Path("alpha_14_gate_diagnostic_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out_md)

    print("\n" + "=" * 140)
    print(f"[*] DIAGNOSTIC COMPLETE: Report saved to {output_path.resolve()}")
    print("=" * 140)

    return report_df


def main():
    parser = argparse.ArgumentParser(description="Alpha 14 Regime Gate Diagnostic")
    parser.add_argument("--universe", type=str, choices=["nifty14", "nifty50"], default="nifty14",
                        help="Universe to evaluate (default: nifty14)")
    parser.add_argument("--lookback", type=int, default=540,
                        help="Total historical lookback in days (default: 540)")
    parser.add_argument("--windows", type=int, default=5,
                        help="Number of chronological windows (default: 5)")

    args = parser.parse_args()
    evaluate_configuration_across_windows(
        universe_type=args.universe,
        total_lookback_days=args.lookback,
        n_windows=args.windows
    )


if __name__ == "__main__":
    main()
