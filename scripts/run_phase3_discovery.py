"""
Ashva Continuous Autonomous Alpha Generator - Phase 3 (Archetypes 76 - 85)
Evaluates Opening 30m ORB, VWAP Reversion from 2.5x ATR, NR7 Squeeze, Liquidity Sweeps, and Delivery Swings.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment

lake = DataLake(read_only=True)
cost_model = IndianCostModel(default_slippage_bps=3.0)

symbols = [
    "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
    "BAJFINANCE", "MARUTI", "SUNPHARMA"
]

print("=" * 120)
print("[*] ASHVA CONTINUOUS ALPHA DISCOVERY - PHASE 3 (EXPANDED DIVERSE ARCHETYPES)")
print("=" * 120)

asset_dfs = {}
for sym in symbols:
    df = lake.load_bars(sym, "15m", max_lookback_days=540)
    if not df.empty and len(df) >= 50:
        asset_dfs[sym] = df

print(f"[+] Loaded {len(asset_dfs)} assets from DataLake.")


def run_phase3():
    capital_per_trade = 125000.0
    oos_days = 120
    results = []

    # 1. ARCHETYPE A: 30-MINUTE OPENING RANGE BREAKOUT (Bar 0 + Bar 1 ORB broken on Bar 2 @ 09:45)
    print("\n[>] Testing Archetype: 30M_OPENING_RANGE_BREAKOUT...")
    for min_orb_range_pct in [0.003, 0.005, 0.008]:
        for min_b2_rvol in [1.2, 1.5, 2.0]:
            for target_rr in [1.25, 1.5, 2.0]:
                trades = []
                oos_trades = []
                pos_assets = 0
                for sym, df in asset_dfs.items():
                    sym_net = 0.0
                    sym_t = 0

                    df = df.copy()
                    ts = pd.to_datetime(df.index)
                    dates = ts.date
                    times = ts.time
                    df["time_str"] = [t.strftime("%H:%M") for t in times]
                    df["bar_date"] = dates

                    # Tod mean vol
                    df["tod_mean_vol"] = df.groupby("time_str")["volume"].transform(
                        lambda s: s.shift(1).rolling(20, min_periods=5).mean()
                    ).fillna(df["volume"])

                    max_dt = ts[-1]
                    oos_cutoff = (max_dt - pd.Timedelta(days=oos_days)).date()

                    # Group by day
                    for d, group in df.groupby("bar_date"):
                        if len(group) < 10: continue
                        t_strs = list(group["time_str"].values)
                        if "09:15" not in t_strs or "09:30" not in t_strs or "09:45" not in t_strs:
                            continue

                        b0 = group[group["time_str"] == "09:15"].iloc[0]
                        b1 = group[group["time_str"] == "09:30"].iloc[0]
                        b2 = group[group["time_str"] == "09:45"].iloc[0]

                        orb30_high = max(b0["high"], b1["high"])
                        orb30_low = min(b0["low"], b1["low"])
                        orb_range = orb30_high - orb30_low
                        mid = (orb30_high + orb30_low) / 2.0

                        if orb_range / mid < min_orb_range_pct:
                            continue

                        b2_rvol = b2["volume"] / max(1.0, b2["tod_mean_vol"])
                        if b2_rvol < min_b2_rvol:
                            continue

                        # Breakout condition on bar 2 (09:45)
                        # Long: bar 2 closes > orb30_high
                        # Short: bar 2 closes < orb30_low
                        side = 0
                        if b2["close"] > orb30_high and b2["close"] > b2["open"]:
                            side = 1
                            sl_p = orb30_low
                        elif b2["close"] < orb30_low and b2["close"] < b2["open"]:
                            side = -1
                            sl_p = orb30_high

                        if side == 0:
                            continue

                        # Entry at bar 3 open (10:00)
                        b3_sub = group[group["time_str"] == "10:00"]
                        if b3_sub.empty: continue
                        entry_p = b3_sub.iloc[0]["open"]
                        if entry_p <= 0: continue

                        if side == 1:
                            risk = max(entry_p * 0.002, entry_p - sl_p)
                            tp_p = entry_p + (target_rr * risk)
                            # evaluate bars from 10:00 to 15:15
                            post_sub = group[group["time_str"] >= "10:00"]
                            exit_p = post_sub.iloc[-1]["close"]
                            for _, prow in post_sub.iterrows():
                                if prow["low"] <= sl_p:
                                    exit_p = sl_p
                                    break
                                elif prow["high"] >= tp_p:
                                    exit_p = tp_p
                                    break
                            qty = int(capital_per_trade / entry_p)
                            if qty <= 0: continue
                            gross = (exit_p - entry_p) * qty
                            turnover = (entry_p + exit_p) * qty
                            cost = 40.0 + (0.000125 * exit_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                            net = gross - cost

                        else:
                            risk = max(entry_p * 0.002, sl_p - entry_p)
                            tp_p = entry_p - (target_rr * risk)
                            post_sub = group[group["time_str"] >= "10:00"]
                            exit_p = post_sub.iloc[-1]["close"]
                            for _, prow in post_sub.iterrows():
                                if prow["high"] >= sl_p:
                                    exit_p = sl_p
                                    break
                                elif prow["low"] <= tp_p:
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
                        if d >= oos_cutoff:
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
                        "mechanism": "30M_OPENING_RANGE_BREAKOUT",
                        "params": {"min_orb_pct": min_orb_range_pct, "min_b2_rvol": min_b2_rvol, "target_rr": target_rr},
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

    # 2. ARCHETYPE B: NR7 VOLATILITY CONTRACTION OPENING GAP BREAKOUT
    print("\n[>] Testing Archetype: NR7_OPENING_GAP_EXPANSION...")
    for min_gap in [0.002, 0.0035, 0.005]:
        for min_rvol in [1.25, 1.5]:
            for target_rr in [1.5, 2.0, 2.5]:
                trades = []
                oos_trades = []
                pos_assets = 0
                for sym, df in asset_dfs.items():
                    sym_net = 0.0
                    sym_t = 0
                    ts = pd.to_datetime(df.index)
                    dates = ts.date
                    times = ts.time
                    df = df.copy()
                    df["time_str"] = [t.strftime("%H:%M") for t in times]
                    df["bar_date"] = dates

                    daily_summary = df.groupby(dates).agg(
                        day_high=("high", "max"),
                        day_low=("low", "min"),
                        day_close=("close", "last"),
                    )
                    daily_range = daily_summary["day_high"] - daily_summary["day_low"]
                    prev_close = daily_summary["day_close"].shift(1)
                    prev_high = daily_summary["day_high"].shift(1)
                    prev_low = daily_summary["day_low"].shift(1)

                    is_nr7_series = pd.Series(False, index=daily_summary.index)
                    for d_i in range(7, len(daily_summary)):
                        if daily_range.iloc[d_i - 1] < daily_range.iloc[d_i - 7:d_i - 1].min():
                            is_nr7_series.iloc[d_i] = True

                    df["is_nr7"] = pd.Series(dates, index=df.index).map(is_nr7_series).ffill().fillna(False)
                    df["prev_day_close"] = pd.Series(dates, index=df.index).map(prev_close).ffill()
                    df["prev_day_high"] = pd.Series(dates, index=df.index).map(prev_high).ffill()
                    df["prev_day_low"] = pd.Series(dates, index=df.index).map(prev_low).ffill()
                    df["tod_mean_vol"] = df.groupby("time_str")["volume"].transform(
                        lambda s: s.shift(1).rolling(20, min_periods=5).mean()
                    ).fillna(df["volume"])

                    max_dt = ts[-1]
                    oos_cutoff = (max_dt - pd.Timedelta(days=oos_days)).date()

                    for d, group in df.groupby("bar_date"):
                        if len(group) < 5: continue
                        t_strs = list(group["time_str"].values)
                        if "09:15" not in t_strs or "09:30" not in t_strs: continue

                        b0 = group[group["time_str"] == "09:15"].iloc[0]
                        b1 = group[group["time_str"] == "09:30"].iloc[0]

                        if not b0["is_nr7"] or pd.isna(b0["prev_day_close"]) or b0["prev_day_close"] <= 0:
                            continue

                        gap_pct = (b0["open"] - b0["prev_day_close"]) / b0["prev_day_close"]
                        abs_gap = abs(gap_pct)
                        rvol = b0["volume"] / max(1.0, b0["tod_mean_vol"])

                        if abs_gap < min_gap or rvol < min_rvol:
                            continue

                        side = 0
                        if gap_pct > 0 and b0["close"] > b0["prev_day_high"] and b0["close"] > b0["open"]:
                            side = 1
                            sl_p = b0["low"]
                        elif gap_pct < 0 and b0["close"] < b0["prev_day_low"] and b0["close"] < b0["open"]:
                            side = -1
                            sl_p = b0["high"]

                        if side == 0: continue
                        entry_p = b1["open"]
                        if entry_p <= 0: continue

                        post_sub = group[group["time_str"] >= "09:30"]
                        exit_p = post_sub.iloc[-1]["close"]

                        if side == 1:
                            risk = max(entry_p * 0.002, entry_p - sl_p)
                            tp_p = entry_p + (target_rr * risk)
                            for _, prow in post_sub.iterrows():
                                if prow["low"] <= sl_p:
                                    exit_p = sl_p
                                    break
                                elif prow["high"] >= tp_p:
                                    exit_p = tp_p
                                    break
                            qty = int(capital_per_trade / entry_p)
                            if qty <= 0: continue
                            gross = (exit_p - entry_p) * qty
                            turnover = (entry_p + exit_p) * qty
                            cost = 40.0 + (0.000125 * exit_p * qty) + (0.0003 * turnover) + (0.000035 * turnover)
                            net = gross - cost
                        else:
                            risk = max(entry_p * 0.002, sl_p - entry_p)
                            tp_p = entry_p - (target_rr * risk)
                            for _, prow in post_sub.iterrows():
                                if prow["high"] >= sl_p:
                                    exit_p = sl_p
                                    break
                                elif prow["low"] <= tp_p:
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
                        if d >= oos_cutoff: oos_trades.append(net)

                    if sym_t > 0 and sym_net > 0: pos_assets += 1

                if len(trades) >= 10:
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
                        "mechanism": "NR7_OPENING_GAP_EXPANSION",
                        "params": {"min_gap": min_gap, "min_rvol": min_rvol, "target_rr": target_rr},
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

    df_all = pd.DataFrame(results)
    df_all.to_csv("phase3_discovery_runs.csv", index=False)
    print(f"\n[+] Evaluated Phase 3 runs. Total evaluated: {len(df_all)}")
    qualifying = df_all[(df_all["540d_pnl"] > 0) & (df_all["oos_pnl"] > 0) & (df_all["pos_assets"] >= 7)]
    print(f"[+] Found {len(qualifying)} qualifying Phase 3 candidates:")
    print(qualifying.sort_values(by="oos_pnl", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    run_phase3()
