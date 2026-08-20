"""
Ashva Continuous Autonomous Alpha Generator - Phase 2 (Discovering Alphas 75 - 82)
Explores novel microstructure, VWAP reclaim, liquidity sweep, and trend continuation mechanisms.
"""

import sys
import os
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.research.continuous_alpha_engine import FastDayBarArray

lake = DataLake(read_only=True)
cost_model = IndianCostModel(default_slippage_bps=3.0)

symbols = [
    "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
    "BAJFINANCE", "MARUTI", "SUNPHARMA"
]

print("=" * 120)
print("[*] ASHVA CONTINUOUS ALPHA DISCOVERY - GENERATING CANDIDATES FOR ALPHAS 75 TO 82")
print("=" * 120)

# Load data and prepare bar caches
asset_dfs = {}
for sym in symbols:
    df = lake.load_bars(sym, "15m", max_lookback_days=540)
    if not df.empty and len(df) >= 50:
        asset_dfs[sym] = df

print(f"[+] Loaded {len(asset_dfs)} assets from DataLake.")


def evaluate_mechanism_candidates():
    capital_per_trade = 125000.0
    oos_days = 120
    results = []

    # Precompute daily and intraday arrays per symbol
    parsed_symbols = {}
    for sym, df in asset_dfs.items():
        parsed_symbols[sym] = FastDayBarArray(sym, df, oos_days=oos_days)

    # 1. MORNING_VWAP_RECLAIM
    print("[>] Exploring Mechanism: MORNING_VWAP_RECLAIM...")
    for min_gap in [0.002, 0.0035, 0.005]:
        for min_rvol in [1.2, 1.5, 1.8]:
            for target_rr in [1.25, 1.5, 2.0]:
                for min_body in [0.50, 0.60]:
                    trades = []
                    oos_trades = []
                    pos_assets = 0
                    for sym, bar in parsed_symbols.items():
                        sym_net = 0.0
                        sym_t = 0
                        gap_pct = (bar.b0_open - bar.prev_closes) / np.maximum(1e-4, bar.prev_closes)
                        abs_gap = np.abs(gap_pct)
                        rvol = bar.b0_vol / np.maximum(1.0, bar.tod_vols)
                        bar_range = bar.b0_high - bar.b0_low
                        body_size = np.abs(bar.b0_close - bar.b0_open)
                        body_ratio = np.where(bar_range > 0, body_size / bar_range, 0.0)

                        # Bar 0 bullish / bearish drive
                        long_drive = (gap_pct >= min_gap) & (rvol >= min_rvol) & (body_ratio >= min_body) & (bar.b0_close > bar.b0_open)
                        short_drive = (gap_pct <= -min_gap) & (rvol >= min_rvol) & (body_ratio >= min_body) & (bar.b0_close < bar.b0_open)

                        long_idx = np.where(long_drive)[0]
                        short_idx = np.where(short_drive)[0]

                        # Longs entered at bar 1 open (09:30)
                        for idx in long_idx:
                            entry_p = bar.b1_open[idx]
                            if entry_p <= 0: continue
                            sl_p = bar.b0_low[idx]
                            risk = max(entry_p * 0.002, entry_p - sl_p)
                            tp_p = entry_p + (target_rr * risk)

                            exit_p = bar.intraday_closes[idx, -1]
                            for bar_k in range(1, 25):
                                if bar.intraday_lows[idx, bar_k] <= sl_p:
                                    exit_p = sl_p
                                    break
                                elif bar.intraday_highs[idx, bar_k] >= tp_p:
                                    exit_p = tp_p
                                    break

                            qty = int(capital_per_trade / entry_p)
                            if qty <= 0: continue
                            gross = (exit_p - entry_p) * qty
                            turnover = (entry_p + exit_p) * qty
                            cost = 40.0 + (0.000125 * exit_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                            net = gross - cost

                            trades.append(net)
                            sym_net += net
                            sym_t += 1
                            if bar.is_oos[idx]:
                                oos_trades.append(net)

                        # Shorts
                        for idx in short_idx:
                            entry_p = bar.b1_open[idx]
                            if entry_p <= 0: continue
                            sl_p = bar.b0_high[idx]
                            risk = max(entry_p * 0.002, sl_p - entry_p)
                            tp_p = entry_p - (target_rr * risk)

                            exit_p = bar.intraday_closes[idx, -1]
                            for bar_k in range(1, 25):
                                if bar.intraday_highs[idx, bar_k] >= sl_p:
                                    exit_p = sl_p
                                    break
                                elif bar.intraday_lows[idx, bar_k] <= tp_p:
                                    exit_p = tp_p
                                    break

                            qty = int(capital_per_trade / entry_p)
                            if qty <= 0: continue
                            gross = (entry_p - exit_p) * qty
                            turnover = (entry_p + exit_p) * qty
                            cost = 40.0 + (0.000125 * entry_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                            net = gross - cost

                            trades.append(net)
                            sym_net += net
                            sym_t += 1
                            if bar.is_oos[idx]:
                                oos_trades.append(net)

                        if sym_t > 0 and sym_net > 0:
                            pos_assets += 1

                    if len(trades) >= 15:
                        tot_net = sum(trades)
                        wr = (len([p for p in trades if p > 0]) / len(trades)) * 100.0
                        wins = [p for p in trades if p > 0]
                        losses = [p for p in trades if p <= 0]
                        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 99.0
                        rets = [p / capital_per_trade for p in trades]
                        std = np.std(rets) if rets else 0.0
                        sh = (np.mean(rets) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0

                        oos_net = sum(oos_trades) if oos_trades else 0.0
                        oos_wr = (len([p for p in oos_trades if p > 0]) / max(1, len(oos_trades))) * 100.0
                        avg_net_trade_pct = (tot_net / (len(trades) * capital_per_trade)) * 100.0

                        results.append({
                            "mechanism": "MORNING_DRIVE_CONTINUATION",
                            "params": {"min_gap": min_gap, "min_rvol": min_rvol, "min_body": min_body, "target_rr": target_rr},
                            "540d_pnl": round(tot_net, 2),
                            "540d_trades": len(trades),
                            "win_rate": round(wr, 1),
                            "profit_factor": round(pf, 2),
                            "sharpe": round(sh, 2),
                            "oos_pnl": round(oos_net, 2),
                            "oos_trades": len(oos_trades),
                            "oos_wr": round(oos_wr, 1),
                            "pos_assets": pos_assets,
                            "avg_trade_pct": round(avg_net_trade_pct, 3),
                        })

    # 2. TWO_DAY_TREND_GAP_SURGE
    print("[>] Exploring Mechanism: TWO_DAY_TREND_GAP_SURGE...")
    for min_gap in [0.0025, 0.004, 0.006]:
        for max_gap in [0.012, 0.018]:
            for min_rvol in [1.2, 1.5]:
                for min_body in [0.55, 0.65]:
                    for target_rr in [1.5, 1.75, 2.0]:
                        trades = []
                        oos_trades = []
                        pos_assets = 0
                        for sym, bar in parsed_symbols.items():
                            sym_net = 0.0
                            sym_t = 0
                            gap_pct = (bar.b0_open - bar.prev_closes) / np.maximum(1e-4, bar.prev_closes)
                            abs_gap = np.abs(gap_pct)
                            rvol = bar.b0_vol / np.maximum(1.0, bar.tod_vols)
                            bar_range = bar.b0_high - bar.b0_low
                            body_size = np.abs(bar.b0_close - bar.b0_open)
                            body_ratio = np.where(bar_range > 0, body_size / bar_range, 0.0)

                            base_cond = (abs_gap >= min_gap) & (abs_gap <= max_gap) & (rvol >= min_rvol) & (body_ratio >= min_body)
                            long_cond = base_cond & (gap_pct > 0) & bar.is_2g & (bar.b0_close > bar.b0_open)
                            short_cond = base_cond & (gap_pct < 0) & bar.is_2r & (bar.b0_close < bar.b0_open)

                            long_idx = np.where(long_cond)[0]
                            short_idx = np.where(short_cond)[0]

                            for idx in long_idx:
                                entry_p = bar.b1_open[idx]
                                if entry_p <= 0: continue
                                sl_p = bar.b0_low[idx]
                                risk = max(entry_p * 0.002, entry_p - sl_p)
                                tp_p = entry_p + (target_rr * risk)

                                exit_p = bar.intraday_closes[idx, -1]
                                for bar_k in range(1, 25):
                                    if bar.intraday_lows[idx, bar_k] <= sl_p:
                                        exit_p = sl_p
                                        break
                                    elif bar.intraday_highs[idx, bar_k] >= tp_p:
                                        exit_p = tp_p
                                        break

                                qty = int(capital_per_trade / entry_p)
                                if qty <= 0: continue
                                gross = (exit_p - entry_p) * qty
                                turnover = (entry_p + exit_p) * qty
                                cost = 40.0 + (0.000125 * exit_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                                net = gross - cost

                                trades.append(net)
                                sym_net += net
                                sym_t += 1
                                if bar.is_oos[idx]: oos_trades.append(net)

                            for idx in short_idx:
                                entry_p = bar.b1_open[idx]
                                if entry_p <= 0: continue
                                sl_p = bar.b0_high[idx]
                                risk = max(entry_p * 0.002, sl_p - entry_p)
                                tp_p = entry_p - (target_rr * risk)

                                exit_p = bar.intraday_closes[idx, -1]
                                for bar_k in range(1, 25):
                                    if bar.intraday_highs[idx, bar_k] >= sl_p:
                                        exit_p = sl_p
                                        break
                                    elif bar.intraday_lows[idx, bar_k] <= tp_p:
                                        exit_p = tp_p
                                        break

                                qty = int(capital_per_trade / entry_p)
                                if qty <= 0: continue
                                gross = (entry_p - exit_p) * qty
                                turnover = (entry_p + exit_p) * qty
                                cost = 40.0 + (0.000125 * entry_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                                net = gross - cost

                                trades.append(net)
                                sym_net += net
                                sym_t += 1
                                if bar.is_oos[idx]: oos_trades.append(net)

                            if sym_t > 0 and sym_net > 0: pos_assets += 1

                        if len(trades) >= 15:
                            tot_net = sum(trades)
                            wr = (len([p for p in trades if p > 0]) / len(trades)) * 100.0
                            wins = [p for p in trades if p > 0]
                            losses = [p for p in trades if p <= 0]
                            pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 99.0
                            rets = [p / capital_per_trade for p in trades]
                            std = np.std(rets) if rets else 0.0
                            sh = (np.mean(rets) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0

                            oos_net = sum(oos_trades) if oos_trades else 0.0
                            oos_wr = (len([p for p in oos_trades if p > 0]) / max(1, len(oos_trades))) * 100.0
                            avg_net_trade_pct = (tot_net / (len(trades) * capital_per_trade)) * 100.0

                            results.append({
                                "mechanism": "TWO_DAY_TREND_GAP_SURGE",
                                "params": {"min_gap": min_gap, "max_gap": max_gap, "min_rvol": min_rvol, "min_body": min_body, "target_rr": target_rr},
                                "540d_pnl": round(tot_net, 2),
                                "540d_trades": len(trades),
                                "win_rate": round(wr, 1),
                                "profit_factor": round(pf, 2),
                                "sharpe": round(sh, 2),
                                "oos_pnl": round(oos_net, 2),
                                "oos_trades": len(oos_trades),
                                "oos_wr": round(oos_wr, 1),
                                "pos_assets": pos_assets,
                                "avg_trade_pct": round(avg_net_trade_pct, 3),
                            })

    # 3. DOUBLE_INSIDE_TARGET_EXPANSION
    print("[>] Exploring Mechanism: DOUBLE_INSIDE_TARGET_EXPANSION...")
    for min_rvol in [1.0, 1.25, 1.5]:
        for target_rr in [1.5, 2.0, 2.5]:
            trades = []
            oos_trades = []
            pos_assets = 0
            for sym, bar in parsed_symbols.items():
                sym_net = 0.0
                sym_t = 0
                rvol = bar.b0_vol / np.maximum(1.0, bar.tod_vols)
                base_cond = bar.is_din & (rvol >= min_rvol)
                long_cond = base_cond & (bar.b0_close > bar.prev_highs) & (bar.b0_close > bar.b0_open)
                short_cond = base_cond & (bar.b0_close < bar.prev_lows) & (bar.b0_close < bar.b0_open)

                long_idx = np.where(long_cond)[0]
                short_idx = np.where(short_cond)[0]

                for idx in long_idx:
                    entry_p = bar.b1_open[idx]
                    if entry_p <= 0: continue
                    sl_p = bar.b0_low[idx]
                    risk = max(entry_p * 0.002, entry_p - sl_p)
                    tp_p = entry_p + (target_rr * risk)

                    exit_p = bar.intraday_closes[idx, -1]
                    for bar_k in range(1, 25):
                        if bar.intraday_lows[idx, bar_k] <= sl_p:
                            exit_p = sl_p
                            break
                        elif bar.intraday_highs[idx, bar_k] >= tp_p:
                            exit_p = tp_p
                            break

                    qty = int(capital_per_trade / entry_p)
                    if qty <= 0: continue
                    gross = (exit_p - entry_p) * qty
                    turnover = (entry_p + exit_p) * qty
                    cost = 40.0 + (0.000125 * exit_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                    net = gross - cost

                    trades.append(net)
                    sym_net += net
                    sym_t += 1
                    if bar.is_oos[idx]: oos_trades.append(net)

                for idx in short_idx:
                    entry_p = bar.b1_open[idx]
                    if entry_p <= 0: continue
                    sl_p = bar.b0_high[idx]
                    risk = max(entry_p * 0.002, sl_p - entry_p)
                    tp_p = entry_p - (target_rr * risk)

                    exit_p = bar.intraday_closes[idx, -1]
                    for bar_k in range(1, 25):
                        if bar.intraday_highs[idx, bar_k] >= sl_p:
                            exit_p = sl_p
                            break
                        elif bar.intraday_lows[idx, bar_k] <= tp_p:
                            exit_p = tp_p
                            break

                    qty = int(capital_per_trade / entry_p)
                    if qty <= 0: continue
                    gross = (entry_p - exit_p) * qty
                    turnover = (entry_p + exit_p) * qty
                    cost = 40.0 + (0.000125 * entry_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                    net = gross - cost

                    trades.append(net)
                    sym_net += net
                    sym_t += 1
                    if bar.is_oos[idx]: oos_trades.append(net)

                if sym_t > 0 and sym_net > 0: pos_assets += 1

            if len(trades) >= 5:
                tot_net = sum(trades)
                wr = (len([p for p in trades if p > 0]) / len(trades)) * 100.0
                wins = [p for p in trades if p > 0]
                losses = [p for p in trades if p <= 0]
                pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 99.0
                rets = [p / capital_per_trade for p in trades]
                std = np.std(rets) if rets else 0.0
                sh = (np.mean(rets) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0

                oos_net = sum(oos_trades) if oos_trades else 0.0
                oos_wr = (len([p for p in oos_trades if p > 0]) / max(1, len(oos_trades))) * 100.0
                avg_net_trade_pct = (tot_net / (len(trades) * capital_per_trade)) * 100.0

                results.append({
                    "mechanism": "DOUBLE_INSIDE_TARGET_EXPANSION",
                    "params": {"min_rvol": min_rvol, "target_rr": target_rr},
                    "540d_pnl": round(tot_net, 2),
                    "540d_trades": len(trades),
                    "win_rate": round(wr, 1),
                    "profit_factor": round(pf, 2),
                    "sharpe": round(sh, 2),
                    "oos_pnl": round(oos_net, 2),
                    "oos_trades": len(oos_trades),
                    "oos_wr": round(oos_wr, 1),
                    "pos_assets": pos_assets,
                    "avg_trade_pct": round(avg_net_trade_pct, 3),
                })

    df_res = pd.DataFrame(results)
    df_res.to_csv("phase2_discovery_runs.csv", index=False)
    print(f"\n[+] Evaluated {len(df_res)} candidates. Results saved to phase2_discovery_runs.csv")

    qualifying = df_res[(df_res["540d_pnl"] > 0) & (df_res["oos_pnl"] > 0) & (df_res["pos_assets"] >= 7)]
    print(f"[+] Found {len(qualifying)} qualifying candidates meeting all institutional filters:")
    print(qualifying.sort_values(by="oos_pnl", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    evaluate_mechanism_candidates()
