"""
Ashva Market Regime Profiler & Alpha DNA Generator
Classifies historical market sessions into Macro Trend, Volatility, and Opening Gap states.
Generates Alpha Regime DNA Cards to dynamically filter hostile market environments
with strict ZERO-LOOKAHEAD bias (strictly using T-1 completed session for macro state).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd

from src.data.data_lake import DataLake


@dataclass
class MarketRegimeState:
    trend: str        # BULL_TREND, BEAR_TREND, SIDEWAYS_CHOP
    volatility: str   # HIGH_VOLATILITY, NORMAL_VOLATILITY, LOW_VOLATILITY
    gap_alignment: str  # ALIGNED_GAP, COUNTER_GAP, FLAT_OPEN
    composite: str    # e.g., BULL_TREND_HIGH_VOLATILITY_ALIGNED


class MarketRegimeProfiler:
    """
    Constructs an equal-weight NIFTY 50 macro benchmark and classifies
    market regimes for every historical session with strict T-1 lookahead protection.
    """

    HEAVYWEIGHTS = [
        "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
        "LT", "ITC", "BHARTIARTL", "SBIN", "KOTAKBANK",
    ]

    def __init__(self, data_lake: Optional[DataLake] = None):
        self.lake = data_lake or DataLake(read_only=True)
        self.benchmark_daily: pd.DataFrame = self._build_benchmark_timeline()

    def _build_benchmark_timeline(self) -> pd.DataFrame:
        """
        Builds a daily benchmark timeline using direct NIFTY 50 index data if present,
        or an equal-weighted composite from top market leaders.
        """
        # 1. Try direct NIFTY Index
        for idx_sym in ["NIFTY", "NIFTY50", "NIFTY 50", "^NSEI"]:
            df_idx = self.lake.load_bars(idx_sym, "1d")
            if not df_idx.empty and len(df_idx) > 50:
                if not isinstance(df_idx.index, pd.DatetimeIndex) and "timestamp" in df_idx.columns:
                    df_idx["timestamp"] = pd.to_datetime(df_idx["timestamp"])
                    df_idx = df_idx.set_index("timestamp").sort_index()
                
                bm = df_idx[["open", "high", "low", "close"]].copy()
                bm["ema20"] = bm["close"].ewm(span=20, adjust=False).mean()
                bm["ema50"] = bm["close"].ewm(span=50, adjust=False).mean()
                tr1 = bm["high"] - bm["low"]
                tr2 = (bm["high"] - bm["close"].shift(1)).abs()
                tr3 = (bm["low"] - bm["close"].shift(1)).abs()
                bm["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                bm["atr14"] = bm["tr"].rolling(14, min_periods=5).mean()
                bm["atr_pct"] = bm["atr14"] / bm["close"] * 100.0
                return bm.dropna()

        # 2. Equal-weight composite from heavyweight leaders
        daily_closes = []
        daily_highs = []
        daily_lows = []
        daily_opens = []

        for sym in self.HEAVYWEIGHTS:
            df = self.lake.load_bars(sym, "1d")
            if df.empty:
                continue
            if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            daily_closes.append(df["close"].rename(sym))
            daily_highs.append(df["high"].rename(sym))
            daily_lows.append(df["low"].rename(sym))
            daily_opens.append(df["open"].rename(sym))

        if not daily_closes:
            return pd.DataFrame()

        df_close = pd.concat(daily_closes, axis=1).ffill().dropna()
        df_high = pd.concat(daily_highs, axis=1).ffill().dropna()
        df_low = pd.concat(daily_lows, axis=1).ffill().dropna()
        df_open = pd.concat(daily_opens, axis=1).ffill().dropna()

        # Normalized equal-weight index starting at 10,000
        norm_close = df_close.pct_change().fillna(0.0).mean(axis=1)
        index_close = (1.0 + norm_close).cumprod() * 10000.0

        norm_open = df_open.pct_change().fillna(0.0).mean(axis=1)
        index_open = (1.0 + norm_open).cumprod() * 10000.0

        norm_high = df_high.pct_change().fillna(0.0).mean(axis=1)
        index_high = (1.0 + norm_high).cumprod() * 10000.0

        norm_low = df_low.pct_change().fillna(0.0).mean(axis=1)
        index_low = (1.0 + norm_low).cumprod() * 10000.0

        bm = pd.DataFrame({
            "open": index_open,
            "high": index_high,
            "low": index_low,
            "close": index_close,
        }, index=df_close.index)

        # Technical Regime Indicators on Daily Benchmark
        bm["ema20"] = bm["close"].ewm(span=20, adjust=False).mean()
        bm["ema50"] = bm["close"].ewm(span=50, adjust=False).mean()

        tr1 = bm["high"] - bm["low"]
        tr2 = (bm["high"] - bm["close"].shift(1)).abs()
        tr3 = (bm["low"] - bm["close"].shift(1)).abs()
        bm["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        bm["atr14"] = bm["tr"].rolling(14, min_periods=5).mean()
        bm["atr_pct"] = bm["atr14"] / bm["close"] * 100.0

        return bm.dropna()

    def get_regime_for_date(self, trade_timestamp: pd.Timestamp, gap_pct: float = 0.0) -> Dict[str, str]:
        """
        Classifies the market regime for a given date with STRICT ZERO-LOOKAHEAD:
        - Trend & Volatility are classified strictly using the PRIOR completed daily bar (T-1).
        - Opening Gap is classified strictly using Today's Open vs T-1 Close.
        """
        if self.benchmark_daily.empty:
            return {"trend": "NEUTRAL", "volatility": "NORMAL", "gap": "FLAT", "composite": "UNKNOWN"}

        t_date = pd.to_datetime(trade_timestamp).date()

        # 1. Look up PRIOR completed session (T-1)
        prior_sessions = self.benchmark_daily[self.benchmark_daily.index.date < t_date]
        if prior_sessions.empty:
            return {"trend": "NEUTRAL", "volatility": "NORMAL", "gap": "FLAT", "composite": "UNKNOWN"}

        prev_row = prior_sessions.iloc[-1]  # Strict T-1 completed session

        # Trend Regime (Strict T-1 Close vs T-1 EMAs)
        close_t1 = prev_row["close"]
        ema20_t1 = prev_row["ema20"]
        ema50_t1 = prev_row["ema50"]

        if close_t1 > ema20_t1 and ema20_t1 >= ema50_t1:
            trend_regime = "BULL_TREND"
        elif close_t1 < ema20_t1 and ema20_t1 <= ema50_t1:
            trend_regime = "BEAR_TREND"
        else:
            trend_regime = "SIDEWAYS_CHOP"

        # Volatility Regime (Strict T-1 ATR vs T-1 Rolling Distribution)
        atr_pct_t1 = prev_row["atr_pct"]
        rolling_atr = prior_sessions["atr_pct"].tail(60)
        p75 = rolling_atr.quantile(0.75) if len(rolling_atr) > 10 else 1.5
        p25 = rolling_atr.quantile(0.25) if len(rolling_atr) > 10 else 0.8

        if atr_pct_t1 >= p75:
            vol_regime = "HIGH_VOLATILITY"
        elif atr_pct_t1 <= p25:
            vol_regime = "LOW_VOLATILITY"
        else:
            vol_regime = "NORMAL_VOLATILITY"

        # Gap Alignment (Today's Open vs T-1 Close)
        if abs(gap_pct) < 0.0020:
            gap_regime = "FLAT_OPEN"
        elif (gap_pct > 0 and trend_regime == "BULL_TREND") or (gap_pct < 0 and trend_regime == "BEAR_TREND"):
            gap_regime = "ALIGNED_GAP"
        else:
            gap_regime = "COUNTER_GAP"

        composite = f"{trend_regime}_{vol_regime}"

        return {
            "trend": trend_regime,
            "volatility": vol_regime,
            "gap": gap_regime,
            "composite": composite,
        }

    def profile_trades(self, trade_list: List[Dict[str, Any]], alpha_id: str) -> Dict[str, Any]:
        """
        Profiles a list of trades against market regimes to construct an Alpha DNA card.
        """
        if not trade_list:
            return {"alpha_id": alpha_id, "approved_regimes": [], "disabled_regimes": []}

        regime_pnl: Dict[str, List[float]] = {}

        for tr in trade_list:
            t_time = tr["entry_time"]
            pnl = tr["pnl"]
            gap = tr.get("gap_pct", 0.0)

            state = self.get_regime_for_date(t_time, gap_pct=gap)
            comp = state["composite"]

            if comp not in regime_pnl:
                regime_pnl[comp] = []
            regime_pnl[comp].append(pnl)

        approved = []
        disabled = []
        regime_breakdown = {}

        for regime, pnls in regime_pnl.items():
            tot_pnl = sum(pnls)
            win_cnt = sum(1 for p in pnls if p > 0)
            wr = (win_cnt / len(pnls) * 100.0) if pnls else 0.0

            regime_breakdown[regime] = {
                "trades": len(pnls),
                "net_pnl": round(tot_pnl, 2),
                "win_rate": round(wr, 1),
            }

            if tot_pnl > 0 and wr >= 35.0:
                approved.append(regime)
            else:
                disabled.append(regime)

        dna_card = {
            "alpha_id": alpha_id,
            "approved_regimes": approved,
            "disabled_regimes": disabled,
            "regime_breakdown": regime_breakdown,
        }

        out_dir = Path("config/alpha_dna")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{alpha_id.lower()}_dna.json", "w") as f:
            json.dump(dna_card, f, indent=2)

        return dna_card
