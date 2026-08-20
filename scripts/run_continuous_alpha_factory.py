"""
Ashva Continuous Autonomous Alpha Discovery & Portfolio Engineering Engine
Multi-hour iterative quantitative discovery, fine-tuning, and validation for Indian Intraday Cash Equities.
"""

import sys
import os
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    HypothesisStatus,
    StrategyHorizon,
    MarketMechanism,
)
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha

lake = DataLake(read_only=True)
cost_model = IndianCostModel()
engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
ledger = ResearchExperimentLedger()
git_sha = get_current_git_sha()

symbols = [
    "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
    "BAJFINANCE", "MARUTI", "SUNPHARMA"
]

print("=" * 115)
print("ASHVA QUANTITATIVE RESEARCH FACTORY: CONTINUOUS AUTONOMOUS DISCOVERY & PORTFOLIO SCALING")
print("=" * 115)

# Pre-cache bars and daily features for rapid multi-hypothesis iteration
cached_data = {}
for sym in symbols:
    df = lake.load_bars(sym, "15m", max_lookback_days=540)
    if df.empty: continue
    
    timestamps = pd.to_datetime(df.index)
    dates = timestamps.date
    times = timestamps.time
    df["time_str"] = [t.strftime("%H:%M") for t in times]
    
    daily_summary = df.groupby(dates).agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
        day_open=("open", "first"),
        day_vol=("volume", "sum")
    )
    
    daily_range = daily_summary["day_high"] - daily_summary["day_low"]
    prev_close = daily_summary["day_close"].shift(1)
    prev_high = daily_summary["day_high"].shift(1)
    prev_low = daily_summary["day_low"].shift(1)
    prev_open = daily_summary["day_open"].shift(1)
    
    tr1 = daily_range
    tr2 = (daily_summary["day_high"] - prev_close).abs()
    tr3 = (daily_summary["day_low"] - prev_close).abs()
    daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)
    
    # Precompute range contraction series (NR3..NR7)
    nr_series = {}
    for k in range(3, 8):
        is_nr = pd.Series(False, index=daily_summary.index)
        for d_i in range(k, len(daily_summary)):
            if daily_range.iloc[d_i - 1] < daily_range.iloc[d_i - k:d_i - 1].min():
                is_nr.iloc[d_i] = True
        nr_series[f"nr{k}"] = is_nr
        
    # Double inside
    h = daily_summary["day_high"]
    l = daily_summary["day_low"]
    is_d_in = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2)) & (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))
    
    # 2-day and 3-day trend alignment
    is_2g = (daily_summary["day_close"].shift(1) > daily_summary["day_open"].shift(1)) & (daily_summary["day_close"].shift(2) > daily_summary["day_open"].shift(2))
    is_2r = (daily_summary["day_close"].shift(1) < daily_summary["day_open"].shift(1)) & (daily_summary["day_close"].shift(2) < daily_summary["day_open"].shift(2))
    
    # 10d max opening volume
    bar1_mask = df["time_str"] == "09:15"
    b1_vols = df.loc[bar1_mask, "volume"]
    b1_dates = df.loc[bar1_mask].index.date
    b1_series = pd.Series(b1_vols.values, index=b1_dates)
    max_b1_10d = b1_series.rolling(10, min_periods=5).max().shift(1)
    
    # Time of day rolling mean volume
    tod_vol = df.groupby("time_str")["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean()
    ).fillna(df["volume"])
    
    df["prev_day_close"] = pd.Series(dates, index=df.index).map(prev_close).ffill()
    df["prev_day_high"] = pd.Series(dates, index=df.index).map(prev_high).ffill()
    df["prev_day_low"] = pd.Series(dates, index=df.index).map(prev_low).ffill()
    df["daily_atr"] = pd.Series(dates, index=df.index).map(daily_atr14).ffill()
    df["tod_mean_vol"] = tod_vol
    df["bar1_10d_max_vol"] = pd.Series(dates, index=df.index).map(max_b1_10d).ffill()
    df["is_double_inside"] = pd.Series(dates, index=df.index).map(is_d_in).ffill().fillna(False)
    df["is_2green"] = pd.Series(dates, index=df.index).map(is_2g).ffill().fillna(False)
    df["is_2red"] = pd.Series(dates, index=df.index).map(is_2r).ffill().fillna(False)
    
    for k in range(3, 8):
        df[f"is_nr{k}"] = pd.Series(dates, index=df.index).map(nr_series[f"nr{k}"]).ffill().fillna(False)
        
    cached_data[sym] = df

print(f"[+] Successfully loaded and enriched 540-day feature cache for {len(cached_data)} symbols.")

# Research Hypothesis Generator across quantitative parameter space
def evaluate_hypothesis_signals(
    hypo_type: str,
    params: Dict[str, Any]
) -> Tuple[float, float, float, int, float, float, List[Any]]:
    all_trades = []
    total_gross = 0.0
    total_net = 0.0
    total_costs = 0.0
    active_days = set()
    
    t_0915 = pd.to_datetime("09:15:00").time()
    t_1515 = pd.to_datetime("15:15:00").time()
    
    min_gap = params.get("min_gap", 0.0035)
    max_gap = params.get("max_gap", 0.0120)
    min_rvol = params.get("min_rvol", 1.75)
    min_body = params.get("min_body", 0.60)
    target_rr = params.get("target_rr", 1.50)
    nr_k = params.get("nr_k", 4)
    
    for sym, df_orig in cached_data.items():
        df = df_orig.copy()
        n = len(df)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n
        
        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values
        tod_vols = df["tod_mean_vol"].values
        prev_closes = df["prev_day_close"].values
        prev_highs = df["prev_day_high"].values
        prev_lows = df["prev_day_low"].values
        daily_atrs = df["daily_atr"].values
        dates = pd.to_datetime(df.index).date
        times = pd.to_datetime(df.index).time
        
        if hypo_type == "NR_GAP_BREAKOUT":
            nr_flags = df[f"is_nr{nr_k}"].values
        elif hypo_type == "DOUBLE_INSIDE":
            nr_flags = df["is_double_inside"].values
        elif hypo_type == "TWO_DAY_TREND":
            g_flags = df["is_2green"].values
            r_flags = df["is_2red"].values
        elif hypo_type == "VOL_OUTLIER":
            vol_outliers = df["bar1_10d_max_vol"].values
            
        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""
        
        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]
            
            if bar_date != current_day:
                current_day = bar_date
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""
                
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                continue
                
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue
                
            if traded_today or pd.isna(prev_closes[i]) or prev_closes[i] <= 0:
                continue
                
            if bar_time == t_0915:
                gap_pct = (opens[i] - prev_closes[i]) / prev_closes[i]
                abs_gap = abs(gap_pct)
                rvol = volumes[i] / max(1.0, tod_vols[i])
                bar_range = highs[i] - lows[i]
                body_size = abs(closes[i] - opens[i])
                body_ratio = (body_size / bar_range) if bar_range > 0 else 0.0
                
                # Condition 1: NR Range Compression + Moderate Gap
                if hypo_type == "NR_GAP_BREAKOUT":
                    if nr_flags[i] and min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= min_body:
                        if gap_pct > 0 and closes[i] > prev_highs[i] and closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True
                        elif gap_pct < 0 and closes[i] < prev_lows[i] and closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = -1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True
                            
                # Condition 2: 2-Day Trend Alignment + Moderate Gap
                elif hypo_type == "TWO_DAY_TREND":
                    if min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= min_body:
                        if gap_pct > 0 and g_flags[i] and closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True
                        elif gap_pct < 0 and r_flags[i] and closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = -1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True
                            
                # Condition 3: Volume Outlier Drive (10d max volume)
                elif hypo_type == "VOL_OUTLIER":
                    if min_gap <= abs_gap <= max_gap and not pd.isna(vol_outliers[i]) and volumes[i] >= (params.get("vol_mult", 1.20) * vol_outliers[i]) and body_ratio >= min_body:
                        if gap_pct > 0 and closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True
                        elif gap_pct < 0 and closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = -1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True

                # Condition 4: Opening Marubozu Momentum (Body >= 70%)
                elif hypo_type == "GAP_MARUBOZU":
                    if min_gap <= abs_gap <= max_gap and rvol >= min_rvol and body_ratio >= 0.70:
                        if gap_pct > 0 and closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True
                        elif gap_pct < 0 and closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = -1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True

                # Condition 5: Double Inside Day Range Expansion
                elif hypo_type == "DOUBLE_INSIDE":
                    if nr_flags[i] and rvol >= min_rvol:
                        if closes[i] > prev_highs[i] and closes[i] > opens[i]:
                            curr_state = 1.0
                            sl = prev_lows[i]
                            risk = closes[i] - sl
                            tp = closes[i] + (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = 1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True
                        elif closes[i] < prev_lows[i] and closes[i] < opens[i]:
                            curr_state = -1.0
                            sl = prev_highs[i]
                            risk = sl - closes[i]
                            tp = closes[i] - (target_rr * risk)
                            curr_sl = sl
                            curr_tp = tp
                            signals[i] = -1.0
                            stop_loss[i] = curr_sl
                            take_profit[i] = curr_tp
                            traded_today = True

        df["signal"] = signals
        df["stop_loss"] = stop_loss
        df["take_profit"] = take_profit
        df["entry_rationale"] = rationales
        
        res = engine.run(df, symbol=sym, strategy_id="eval", capital_per_trade_pct=0.25, risk_per_trade_pct=0.005)
        if res.total_trades > 0:
            for t in res.trade_list:
                all_trades.append(t)
                active_days.add(pd.to_datetime(t.entry_time).date())
                total_gross += t.gross_pnl
                total_net += t.net_pnl
                total_costs += (t.gross_pnl - t.net_pnl)
                
    n_t = len(all_trades)
    if n_t == 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0, []
        
    wins = [t for t in all_trades if t.net_pnl > 0]
    losses = [t for t in all_trades if t.net_pnl <= 0]
    win_rate = (len(wins) / n_t) * 100.0
    
    net_wins = sum(t.net_pnl for t in wins)
    net_losses = abs(sum(t.net_pnl for t in losses))
    net_pf = (net_wins / net_losses) if net_losses > 0 else (99.0 if net_wins > 0 else 0.0)
    
    all_sorted = sorted(all_trades, key=lambda t: t.exit_time)
    cum_eq = np.cumsum([t.net_pnl for t in all_sorted])
    running_max = np.maximum.accumulate(cum_eq)
    max_dd = abs(min(cum_eq - running_max)) if len(cum_eq) > 0 else 0.0
    max_dd_pct = (max_dd / 500000.0) * 100.0
    
    rets = [t.net_pnl / 125000.0 for t in all_trades]
    std = np.std(rets)
    sharpe = (np.mean(rets) / std * np.sqrt(252 * 6.25 / 9.6)) if std > 0 else 0.0
    
    return total_net, net_pf, sharpe, n_t, win_rate, max_dd_pct, all_trades

print("[+] Signal evaluation kernel compiled and ready for parameter grid search.")
