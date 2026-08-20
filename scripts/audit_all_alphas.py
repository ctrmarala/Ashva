"""
Ashva Complete 66-Alpha Comprehensive Portfolio Audit
Audits all registered strategies in STRATEGY_MAP across NIFTY-14 over 540 days (IS: 420d, Untouched OOS: 120d).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.research.hypothesis import StrategyHorizon
from scripts.run_hypothesis_lab import STRATEGY_MAP, DEFAULT_UNIVERSE

lake = DataLake(read_only=True)
cost_model = IndianCostModel(default_slippage_bps=3.0)
symbols = DEFAULT_UNIVERSE
lookback_days = 540
oos_days = 120
capital_per_asset = 500000.0
total_basket_capital = capital_per_asset * len(symbols)

print("=" * 130)
print(f"[*] AUDITING ALL {len(STRATEGY_MAP)} REGISTERED STRATEGIES ACROSS {len(symbols)} ASSETS (540d / 120d OOS)")
print("=" * 130)

results = []

for strat_id, (strat_name, strat_cls) in STRATEGY_MAP.items():
    strat_obj = strat_cls()
    is_swing = False
    if hasattr(strat_obj, "metadata"):
        h = getattr(strat_obj.metadata, "horizon", None)
        if h in [StrategyHorizon.SWING, StrategyHorizon.POSITIONAL] or str(h) in ["SWING", "POSITIONAL", "StrategyHorizon.SWING", "StrategyHorizon.POSITIONAL"]:
            is_swing = True

    engine_segment = Segment.EQUITY_DELIVERY if is_swing else Segment.EQUITY_INTRADAY
    target_tf = getattr(strat_obj.metadata, "timeframe", "15m") if hasattr(strat_obj, "metadata") else "15m"

    total_net_pnl = 0.0
    total_trades = 0
    total_wins = 0
    oos_net_pnl = 0.0
    oos_trades = 0
    oos_wins = 0
    pos_assets = 0
    asset_pnls = {}

    all_daily_pnl = []
    oos_daily_pnl = []

    for sym in symbols:
        df = lake.load_bars(sym, target_tf, max_lookback_days=lookback_days)
        if df.empty or len(df) < 50:
            if target_tf != "15m":
                df = lake.load_bars(sym, "15m", max_lookback_days=lookback_days)
            if df.empty or len(df) < 50:
                continue

        try:
            sig_df = strat_obj.generate_signals(df)
            eng = BacktestEngine(cost_model=cost_model, initial_capital=capital_per_asset, segment=engine_segment)
            res = eng.run(sig_df, symbol=sym, strategy_id=strat_id, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)

            # Temporal OOS
            if oos_days > 0 and len(sig_df) > 50:
                max_ts = pd.to_datetime(sig_df.index[-1])
                oos_cutoff = max_ts - timedelta(days=oos_days)
                sig_oos = sig_df[sig_df.index >= oos_cutoff]
                if not sig_oos.empty and len(sig_oos) > 10:
                    res_oos = eng.run(sig_oos, symbol=sym, strategy_id=strat_id, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)
                    oos_net_pnl += res_oos.total_net_pnl
                    oos_trades += res_oos.total_trades
                    oos_wins += res_oos.winning_trades
                    if len(res_oos.equity_curve) > 1:
                        oos_daily_pnl.append(res_oos.equity_curve.diff().resample("1D").sum().fillna(0.0))

            if res.total_trades > 0:
                total_net_pnl += res.total_net_pnl
                total_trades += res.total_trades
                total_wins += res.winning_trades
                asset_pnls[sym] = res.total_net_pnl
                if res.total_net_pnl > 0:
                    pos_assets += 1
                if len(res.equity_curve) > 1:
                    all_daily_pnl.append(res.equity_curve.diff().resample("1D").sum().fillna(0.0))

        except Exception as e:
            pass

    if all_daily_pnl:
        combined_pnl = pd.concat(all_daily_pnl, axis=1).sum(axis=1).fillna(0.0)
        rets = combined_pnl / total_basket_capital
        mean_r = rets.mean()
        std_r = rets.std()
        sharpe = float((mean_r / std_r * np.sqrt(252))) if std_r > 1e-7 else 0.0
    else:
        sharpe = 0.0

    if oos_daily_pnl:
        combined_oos = pd.concat(oos_daily_pnl, axis=1).sum(axis=1).fillna(0.0)
        oos_rets = combined_oos / total_basket_capital
        m_oos = oos_rets.mean()
        s_oos = oos_rets.std()
        oos_sharpe = float((m_oos / s_oos * np.sqrt(252))) if s_oos > 1e-7 else 0.0
    else:
        oos_sharpe = 0.0

    win_rate = (total_wins / max(1, total_trades)) * 100.0 if total_trades > 0 else 0.0
    oos_win_rate = (oos_wins / max(1, oos_trades)) * 100.0 if oos_trades > 0 else 0.0

    results.append({
        "Alpha_ID": strat_id,
        "Strategy_Name": strat_name,
        "540d_Net_PnL": round(total_net_pnl, 2),
        "540d_Trades": total_trades,
        "540d_Win_Rate": round(win_rate, 1),
        "540d_Sharpe": round(sharpe, 2),
        "120d_OOS_Net_PnL": round(oos_net_pnl, 2),
        "120d_OOS_Trades": oos_trades,
        "120d_OOS_Win_Rate": round(oos_win_rate, 1),
        "120d_OOS_Sharpe": round(oos_sharpe, 2),
        "Pos_Assets": f"{pos_assets}/{len(symbols)}",
    })
    
    status_icon = "PASS " if total_net_pnl > 0 and oos_net_pnl > 0 else ("MIXED" if total_net_pnl > 0 or oos_net_pnl > 0 else "FAIL ")
    print(f"[{status_icon}] {strat_id:<10} | {strat_name:<42} | 540d: Rs {total_net_pnl:+10,.0f} ({total_trades:4d}T, {win_rate:4.1f}%W, Sh:{sharpe:+4.2f}) | OOS: Rs {oos_net_pnl:+9,.0f} ({oos_trades:3d}T, {oos_win_rate:4.1f}%W, Sh:{oos_sharpe:+4.2f}) | Pos: {pos_assets:2d}/{len(symbols)}")

df_res = pd.DataFrame(results)
df_res.sort_values(by="120d_OOS_Net_PnL", ascending=False, inplace=True)
df_res.to_csv("all_66_alphas_audit.csv", index=False)
print("\n" + "=" * 130)
print(f"[*] TOP 15 STRATEGIES BY 120-DAY OUT-OF-SAMPLE (OOS) NET PNL:")
print("=" * 130)
print(df_res.head(15).to_string(index=False))
