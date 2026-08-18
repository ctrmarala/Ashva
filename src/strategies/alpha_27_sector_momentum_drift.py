"""
Ashva Quantitative Alpha 27: Cross-Sectional Sector Momentum Drift
Hypothesis:
    When an entire sector cluster (e.g., Banking, IT) demonstrates synchronized
    directional breadth (multiple peer stocks moving in tandem) during the morning session,
    the sector leader exhibiting the highest relative strength and volume expansion tends to
    experience persistent directional drift through the remainder of the session.

Mechanism:
    Institutional sector allocation and basket buying/selling creates persistent intra-day
    tailwinds that overpower single-stock mean-reverting market makers.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    StrategyHorizon,
    MarketMechanism,
)
from src.features.indicators import TechnicalIndicators
from src.data.data_lake import DataLake


SECTOR_MAP = {
    "IT": ["INFY", "TCS", "HCLTECH", "TECHM", "WIPRO", "LTIM"],
    "BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK"],
    "AUTO": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "M&M", "HEROMOTOCO", "EICHERMOT"],
    "METAL": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "FINANCE": ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE"],
    "ENERGY": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
    "CONGLOMERATE": ["LT", "BHARTIARTL", "GRASIM", "TITAN", "NESTLEIND", "ITC"],
}

SYMBOL_TO_SECTOR = {}
for sec, syms in SECTOR_MAP.items():
    for s in syms:
        SYMBOL_TO_SECTOR[s] = sec


class Alpha27SectorMomentumDrift(BaseHypothesis):
    """
    Alpha 27: Sector-Synchronized Momentum Drift Strategy.
    """

    _sector_dynamics_cache: Optional[Dict[str, Dict[str, pd.Series]]] = None
    _symbol_signatures_cache: Optional[Dict[str, float]] = None

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        meta = HypothesisMetadata(
            hypothesis_id="alpha_27",
            name="ALPHA_27_SECTOR_MOMENTUM_DRIFT",
            category="CROSS_SECTIONAL_MOMENTUM",
            economic_rationale=(
                "Institutional capital deploys via sector baskets (e.g. Bank Nifty or Nifty IT baskets). "
                "When sector peers move synchronously with high breadth, idiosyncratic noise is reduced, "
                "and directional flow persists until session close."
            ),
            target_instruments=["NIFTY50_LIQUID"],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
        )
        super().__init__(meta, parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_sector_breadth": [0.65, 0.75, 0.85],
            "min_sector_return_pct": [0.0030, 0.0050, 0.0075],
            "min_rvol": [1.10, 1.25, 1.40],
            "target_rr": [1.50, 2.00],
            "max_entry_hour": [11, 12, 13],
        }

    @classmethod
    def _initialize_caches(cls):
        """Loads and caches sector returns and symbol matching signatures."""
        if cls._sector_dynamics_cache is not None:
            return

        lake = DataLake(read_only=True)
        all_symbols = lake.list_symbols()
        
        sector_bars = {}
        signatures = {}

        for sym in all_symbols:
            sec = SYMBOL_TO_SECTOR.get(sym)
            df = lake.load_bars(sym, "15m")
            if not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Signature: last available close price
                signatures[sym] = float(df["close"].iloc[-1])

                if sec:
                    if sec not in sector_bars:
                        sector_bars[sec] = {}
                    dates = df.index.date
                    day_open = df.groupby(dates)["open"].transform("first")
                    cum_ret = (df["close"] - day_open) / day_open.replace(0, np.nan)
                    sector_bars[sec][sym] = cum_ret

        sector_summary = {}
        for sec, sym_dict in sector_bars.items():
            if len(sym_dict) >= 1:
                panel = pd.DataFrame(sym_dict)
                mean_ret = panel.mean(axis=1)
                pos_breadth = (panel > 0).sum(axis=1) / panel.count(axis=1)
                neg_breadth = (panel < 0).sum(axis=1) / panel.count(axis=1)
                sector_summary[sec] = {
                    "mean_return": mean_ret,
                    "pos_breadth": pos_breadth,
                    "neg_breadth": neg_breadth,
                }

        cls._sector_dynamics_cache = sector_summary
        cls._symbol_signatures_cache = signatures

    def _identify_symbol(self, df: pd.DataFrame) -> str:
        """Identifies symbol name from DataFrame."""
        if hasattr(self, "current_symbol") and self.current_symbol:
            return self.current_symbol
        
        self._initialize_caches()
        if not df.empty and self._symbol_signatures_cache:
            last_close = float(df["close"].iloc[-1])
            for sym, sig in self._symbol_signatures_cache.items():
                if abs(sig - last_close) < 1e-4:
                    return sym
        return "RELIANCE"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                out.index = pd.to_datetime(out.index)

        self._initialize_caches()
        sym = self._identify_symbol(out)
        sec = SYMBOL_TO_SECTOR.get(sym, "CONGLOMERATE")

        dates = out.index.date
        times = out.index.time

        # 1. Daily ATR (14-day) anchored to prior days
        daily_df = out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily_df) >= 14:
            daily_atr_df = TechnicalIndicators.add_atr(daily_df, period=14)
            daily_atr_prev = daily_atr_df["atr_14"].shift(1)
            atr_map = daily_atr_prev.to_dict()
            out["daily_atr"] = [atr_map.get(pd.Timestamp(d), np.nan) for d in dates]
        else:
            out["daily_atr"] = (out["high"] - out["low"]).rolling(14).mean()

        out["daily_atr"] = out["daily_atr"].ffill().bfill()

        # 2. Time-of-Day Mean Volume Baseline
        tod_rolling = out.groupby(times)["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        # Cumulative intraday stock return
        day_open = out.groupby(dates)["open"].transform("first")
        out["stock_cum_ret"] = (out["close"] - day_open) / day_open.replace(0, np.nan)

        # Retrieve Sector Dynamics
        sector_data = self._sector_dynamics_cache.get(sec, {}) if self._sector_dynamics_cache else {}

        if sector_data:
            out["sec_ret"] = sector_data["mean_return"].reindex(out.index).fillna(0.0)
            out["sec_pos_breadth"] = sector_data["pos_breadth"].reindex(out.index).fillna(0.5)
            out["sec_neg_breadth"] = sector_data["neg_breadth"].reindex(out.index).fillna(0.5)
        else:
            out["sec_ret"] = 0.0
            out["sec_pos_breadth"] = 0.5
            out["sec_neg_breadth"] = 0.5

        # Strategy Hyperparameters
        min_breadth = float(self.parameters.get("min_sector_breadth", 0.75))
        min_sec_ret = float(self.parameters.get("min_sector_return_pct", 0.0040))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_entry_hour = int(self.parameters.get("max_entry_hour", 12))

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        daily_atrs = out["daily_atr"].values
        stock_rets = out["stock_cum_ret"].values
        sec_rets = out["sec_ret"].values
        sec_pos_b = out["sec_pos_breadth"].values
        sec_neg_b = out["sec_neg_breadth"].values

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_0945 = pd.to_datetime("09:45:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(1, n):
            bar_date = dates[i]
            bar_time = times[i]

            # Reset on new trading day
            if bar_date != current_day:
                current_day = bar_date
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""

            # Intraday 15:15 EOD Square-Off
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 27 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across holding bars
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or bar_time < t_0945 or bar_time.hour > max_entry_hour:
                continue

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]
            c_sret = stock_rets[i]
            c_secret = sec_rets[i]
            c_pos_b = sec_pos_b[i]
            c_neg_b = sec_neg_b[i]

            if pd.isna(c_atr) or c_atr <= 0:
                continue

            rvol = c_vol / max(1.0, c_tod)
            if rvol < min_rvol:
                continue

            # Case 1: Bullish Sector Tailwinds
            if (c_pos_b >= min_breadth) and (c_secret >= min_sec_ret) and (c_sret >= c_secret) and (c_close > c_open):
                curr_state = 1.0
                stop_dist = max(c_close - c_low + 0.10 * c_atr, 0.30 * c_atr)
                curr_sl = c_close - stop_dist
                curr_tp = c_close + (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 27 SECTOR LONG: {sym} in {sec} | Breadth={c_pos_b*100:.0f}% | "
                    f"SectorRet={c_secret*100:+.2f}% | StockRet={c_sret*100:+.2f}% | RVOL={rvol:.2f}x"
                )
                signals[i] = 1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

            # Case 2: Bearish Sector Headwinds
            elif (c_neg_b >= min_breadth) and (c_secret <= -min_sec_ret) and (c_sret <= c_secret) and (c_close < c_open):
                curr_state = -1.0
                stop_dist = max(c_high - c_close + 0.10 * c_atr, 0.30 * c_atr)
                curr_sl = c_close + stop_dist
                curr_tp = c_close - (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 27 SECTOR SHORT: {sym} in {sec} | Breadth={c_neg_b*100:.0f}% | "
                    f"SectorRet={c_secret*100:+.2f}% | StockRet={c_sret*100:+.2f}% | RVOL={rvol:.2f}x"
                )
                signals[i] = -1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
