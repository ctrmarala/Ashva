"""
Ashva Market Regime Profiler & Alpha DNA Generator
Segments market history into macro regimes (Trend, Volatility, Opening Gap Alignment),
tags every trade with its regime state at entry, and outputs an institutional Alpha Regime DNA Card.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.data_lake import DataLake


class MarketRegimeProfiler:
    """
    Classifies market macro regimes and profiles strategy performance per regime.
    """

    def __init__(self, data_lake: Optional[DataLake] = None):
        self.lake = data_lake or DataLake(read_only=True)
        self.benchmark_daily: pd.DataFrame = pd.DataFrame()
        self._load_market_benchmark()

    def _load_market_benchmark(self):
        """Constructs an aggregate equal-weight NIFTY 50 benchmark from top heavyweight equities."""
        heavyweights = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "ITC", "LT"]
        daily_dfs = []
        for sym in heavyweights:
            df = self.lake.load_bars(sym, "1d")
            if not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp").sort_index()
                daily_dfs.append(df[["close", "open", "high", "low", "volume"]])

        if not daily_dfs:
            return

        # Combine into equal-weight proxy
        all_dates = sorted(list(set.intersection(*[set(df.index.date) for df in daily_dfs])))
        agg_rows = []
        for d in all_dates:
            d_closes = [df.loc[df.index.date == d, "close"].iloc[-1] for df in daily_dfs if len(df.loc[df.index.date == d]) > 0]
            d_opens = [df.loc[df.index.date == d, "open"].iloc[0] for df in daily_dfs if len(df.loc[df.index.date == d]) > 0]
            d_highs = [df.loc[df.index.date == d, "high"].max() for df in daily_dfs if len(df.loc[df.index.date == d]) > 0]
            d_lows = [df.loc[df.index.date == d, "low"].min() for df in daily_dfs if len(df.loc[df.index.date == d]) > 0]

            agg_rows.append({
                "date": d,
                "open": np.mean(d_opens),
                "high": np.mean(d_highs),
                "low": np.mean(d_lows),
                "close": np.mean(d_closes),
            })

        bm = pd.DataFrame(agg_rows).set_index(pd.to_datetime([r["date"] for r in agg_rows]))
        bm["ema20"] = bm["close"].ewm(span=20, adjust=False).mean()
        bm["ema50"] = bm["close"].ewm(span=50, adjust=False).mean()
        
        # Volatility: 14-day normalized ATR
        tr = np.maximum(bm["high"] - bm["low"], np.abs(bm["high"] - bm["close"].shift(1)))
        tr = np.maximum(tr, np.abs(bm["low"] - bm["close"].shift(1)))
        bm["atr14"] = tr.rolling(14, min_periods=5).mean()
        bm["atr_pct"] = (bm["atr14"] / bm["close"]) * 100.0
        bm["vol_p75"] = bm["atr_pct"].rolling(60, min_periods=15).quantile(0.75)
        bm["vol_p25"] = bm["atr_pct"].rolling(60, min_periods=15).quantile(0.25)

        # Gap %
        bm["gap_pct"] = ((bm["open"] - bm["close"].shift(1)) / bm["close"].shift(1)) * 100.0

        self.benchmark_daily = bm

    def get_regime_for_date(self, trade_date: Any, stock_gap_pct: float = 0.0) -> Dict[str, str]:
        """
        Classifies the exact market regime for a given trade date.
        """
        if self.benchmark_daily.empty:
            return {"trend": "UNKNOWN", "volatility": "UNKNOWN", "gap_alignment": "UNKNOWN"}

        t_date = pd.to_datetime(trade_date).date()
        match = self.benchmark_daily.loc[self.benchmark_daily.index.date == t_date]
        if match.empty:
            # Fallback to closest prior date
            prior = self.benchmark_daily.loc[self.benchmark_daily.index.date <= t_date]
            if prior.empty:
                return {"trend": "UNKNOWN", "volatility": "UNKNOWN", "gap_alignment": "UNKNOWN"}
            row = prior.iloc[-1]
        else:
            row = match.iloc[0]

        # 1. Trend Regime
        close = row["close"]
        ema20 = row["ema20"]
        ema50 = row["ema50"]
        if close > ema20 and ema20 >= ema50:
            trend_regime = "BULL_TREND"
        elif close < ema20 and ema20 <= ema50:
            trend_regime = "BEAR_TREND"
        else:
            trend_regime = "SIDEWAYS_CHOP"

        # 2. Volatility Regime
        atr_pct = row["atr_pct"]
        p75 = row["vol_p75"] if not np.isnan(row["vol_p75"]) else 1.5
        p25 = row["vol_p25"] if not np.isnan(row["vol_p25"]) else 0.8
        if atr_pct >= p75:
            vol_regime = "HIGH_VOLATILITY"
        elif atr_pct <= p25:
            vol_regime = "LOW_VOLATILITY"
        else:
            vol_regime = "NORMAL_VOLATILITY"

        # 3. Gap Alignment
        mkt_gap = row["gap_pct"]
        if abs(mkt_gap) < 0.15:
            gap_alignment = "FLAT_OPEN"
        elif (mkt_gap > 0 and stock_gap_pct > 0) or (mkt_gap < 0 and stock_gap_pct < 0):
            gap_alignment = "ALIGNED_GAP"
        else:
            gap_alignment = "COUNTER_GAP"

        return {
            "trend": trend_regime,
            "volatility": vol_regime,
            "gap_alignment": gap_alignment,
            "composite": f"{trend_regime}_{vol_regime}",
        }

    def profile_trades(self, trades: List[Dict[str, Any]], alpha_id: str = "alpha_generic") -> Dict[str, Any]:
        """
        Takes a list of trade dictionaries, tags each with its regime, and computes regime decomposition.
        """
        if not trades:
            return {"status": "EMPTY", "alpha_id": alpha_id, "regimes": {}}

        df_trades = pd.DataFrame(trades)
        if "entry_time" not in df_trades.columns:
            return {"status": "INVALID_SCHEMA", "alpha_id": alpha_id}

        df_trades["entry_date"] = pd.to_datetime(df_trades["entry_time"]).dt.date
        df_trades["stock_gap_pct"] = df_trades.get("gap_pct", 0.0)

        # Tag each trade
        regime_tags = []
        for _, row in df_trades.iterrows():
            reg = self.get_regime_for_date(row["entry_date"], row["stock_gap_pct"])
            regime_tags.append(reg)

        df_reg = pd.DataFrame(regime_tags)
        df_merged = pd.concat([df_trades.reset_index(drop=True), df_reg.reset_index(drop=True)], axis=1)

        # Decompose across Trend, Volatility, and Composite
        regime_metrics = {}
        for category in ["trend", "volatility", "gap_alignment", "composite"]:
            regime_metrics[category] = {}
            for state, grp in df_merged.groupby(category):
                n_trades = len(grp)
                net_pnl = float(grp["pnl"].sum())
                win_rate = float((grp["pnl"] > 0).mean() * 100.0)
                gross_win = float(grp.loc[grp["pnl"] > 0, "pnl"].sum())
                gross_loss = float(abs(grp.loc[grp["pnl"] < 0, "pnl"].sum()))
                profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

                # Sharpe
                returns = grp["pnl"]
                sharpe = float((returns.mean() / (returns.std() + 1e-6)) * np.sqrt(252)) if len(returns) > 3 else 0.0

                is_approved = (net_pnl > 0 and profit_factor >= 1.1 and n_trades >= 5)
                status = "APPROVED" if is_approved else ("MARGINAL" if net_pnl > 0 else "DISABLED")

                regime_metrics[category][state] = {
                    "trades": n_trades,
                    "net_pnl": round(net_pnl, 2),
                    "win_rate": f"{round(win_rate, 1)}%",
                    "profit_factor": round(profit_factor, 2),
                    "sharpe": round(sharpe, 2),
                    "status": status,
                }

        # Generate DNA Card Summary
        dna_card = {
            "alpha_id": alpha_id,
            "total_trades": len(df_merged),
            "total_net_pnl": round(float(df_merged["pnl"].sum()), 2),
            "approved_regimes": [
                k for k, v in regime_metrics["composite"].items() if v["status"] == "APPROVED"
            ],
            "disabled_regimes": [
                k for k, v in regime_metrics["composite"].items() if v["status"] == "DISABLED"
            ],
            "regime_breakdown": regime_metrics,
        }

        # Save DNA Card to config/alpha_dna/
        out_dir = Path("config/alpha_dna")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{alpha_id.lower()}_dna.json"
        with open(out_file, "w") as f:
            json.dump(dna_card, f, indent=2)

        return dna_card
