"""
Ashva Quantitative Strategy: Opening Relative-Strength Leadership (Alpha 09)
Captures cross-sectional momentum by identifying stock-specific institutional leadership
during the first 15 minutes (RS_i >= +0.30% vs universe median) followed by an opening range breakout.

Hypothesis:
When one stock significantly outperforms its peer universe during the first 15 minutes,
while the broader universe is not moving equally strongly, that relative strength represents
stock-specific institutional demand. If the leader subsequently breaks its own opening range,
the probability of continuation is significantly higher than for an ordinary breakout.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import os

from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus
from src.data.data_lake import DataLake


class Alpha09OpeningRelativeStrength(BaseHypothesis):
    """
    Opening Relative-Strength Leadership (Alpha 09):
    1. Cross-Sectional Opening Return: At 09:15-09:30, R_i = (Close_0915 - Open_0915) / Open_0915.
    2. Universe Median Benchmark: R_median = Median(R_universe) across 14 liquid blue chips.
    3. Relative Strength Gate:
       - Bullish Leader: RS_i = R_i - R_median >= +0.30% (+0.0030), Body >= 50%, RVOL >= 1.20x.
       - Bearish Laggard: RS_i = R_i - R_median <= -0.30% (-0.0030), Body >= 50%, RVOL >= 1.20x.
    4. Confirmation Breakout (09:30 to 11:00): Close outside Bar 1 High/Low + same-direction candle.
    5. Risk/Reward: Stop at opposite OR15 boundary, exact 1.5R target (1:1.50), EOD 15:15 exit.
    6. Frequency: Maximum 1 trade per stock per day.
    """

    # Class-level cache for universe median returns to ensure high-speed backtesting
    _universe_median_returns_cache: Optional[pd.Series] = None

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_09_OPENING_RELATIVE_STRENGTH",
            name="Alpha_09_Opening_Relative_Strength",
            category="CROSS_SECTIONAL_MOMENTUM_LEADERSHIP",
            economic_rationale=(
                "When one stock significantly outperforms its peer universe during the first 15 minutes, "
                "while the broader universe is not moving equally strongly, that relative strength represents "
                "stock-specific institutional demand. If the leader subsequently breaks its own opening range, "
                "the probability of continuation is significantly higher than for an ordinary breakout."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            author="AshvaQuantLab",
        )
        params = parameters or {
            "min_rs_threshold": 0.0030,        # Relative Strength >= +0.30% (or <= -0.30%)
            "min_body_ratio": 0.50,            # Bar 1 Body >= 50% of candle range
            "min_rvol": 1.20,                  # Volume >= 1.20x shifted TOD baseline
            "target_rr": 1.50,                 # Exactly 1:1.50 Risk-to-Reward ratio
            "max_breakout_hour": 11,           # Breakout window ends at 11:00 IST
            "max_breakout_minute": 0,
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rs_threshold": [0.0020, 0.0030, 0.0040],
            "min_body_ratio": [0.45, 0.50, 0.55],
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00],
        }

    @classmethod
    def _compute_universe_median_returns(cls) -> pd.Series:
        """
        Loads 15m opening bars across all universe stocks to compute daily median 09:15 return.
        Cached in class variable for high backtest performance.
        """
        if cls._universe_median_returns_cache is not None:
            return cls._universe_median_returns_cache

        lake = DataLake(read_only=True)
        universe = [
            "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
            "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
            "BAJFINANCE", "MARUTI", "SUNPHARMA"
        ]

        daily_stock_returns = {}
        for sym in universe:
            df = lake.load_bars(sym, "15m")
            if df.empty:
                continue
            ts = pd.to_datetime(df.index)
            # Filter 09:15 bars
            mask_0915 = (ts.time == pd.Timestamp("09:15").time())
            bars_0915 = df[mask_0915].copy()
            if not bars_0915.empty:
                ret_0915 = (bars_0915["close"] - bars_0915["open"]) / bars_0915["open"].replace(0, np.nan)
                ret_0915.index = pd.to_datetime(ret_0915.index).date
                daily_stock_returns[sym] = ret_0915

        if daily_stock_returns:
            panel_df = pd.DataFrame(daily_stock_returns)
            cls._universe_median_returns_cache = panel_df.median(axis=1)
        else:
            cls._universe_median_returns_cache = pd.Series(dtype=float)

        return cls._universe_median_returns_cache

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 09.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Compute 20-Session TOD Rolling Volume Baseline (Shifted 1 session)
        tod_rolling = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        # 2. Retrieve Daily Universe Median 09:15 Returns
        universe_median_series = self._compute_universe_median_returns()

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

        min_rs = float(self.parameters.get("min_rs_threshold", 0.0030))
        min_body = float(self.parameters.get("min_body_ratio", 0.50))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_hour = int(self.parameters.get("max_breakout_hour", 11))
        max_min = int(self.parameters.get("max_breakout_minute", 0))

        current_day = None
        or15_high = 0.0
        or15_low = 0.0
        is_bullish_leader = False
        is_bearish_laggard = False
        stock_rs_pct = 0.0
        traded_today = False

        for i in range(n):
            ts = timestamps[i]
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            # Reset on new daily session
            if bar_date != current_day:
                current_day = bar_date
                or15_high = 0.0
                or15_low = 0.0
                is_bullish_leader = False
                is_bearish_laggard = False
                stock_rs_pct = 0.0
                traded_today = False

            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]
            c_vol = volumes[i]
            c_tod_vol = tod_vols[i]

            # -------------------------------------------------------------
            # Phase 1 & 2: 09:15 Relative Strength & Quality Evaluation
            # -------------------------------------------------------------
            if hour == 9 and minute == 15:
                or15_high = c_high
                or15_low = c_low
                candle_range = c_high - c_low

                if candle_range > 0.01 and c_open > 0:
                    stock_ret = (c_close - c_open) / c_open
                    univ_median_ret = universe_median_series.get(bar_date, 0.0) if not universe_median_series.empty else 0.0
                    rel_strength = stock_ret - univ_median_ret
                    stock_rs_pct = rel_strength * 100.0

                    body_ratio = abs(c_close - c_open) / candle_range
                    rvol = c_vol / max(1.0, c_tod_vol)

                    # Bullish Leader: Outperformed universe median by >= +0.30% + Body >= 50% + RVOL >= 1.20x
                    if (rel_strength >= min_rs) and (c_close > c_open) and (body_ratio >= min_body) and (rvol >= min_rvol):
                        is_bullish_leader = True

                    # Bearish Laggard: Underperformed universe median by <= -0.30% + Body >= 50% + RVOL >= 1.20x
                    elif (rel_strength <= -min_rs) and (c_close < c_open) and (body_ratio >= min_body) and (rvol >= min_rvol):
                        is_bearish_laggard = True
                continue

            # Skip if not a qualified leader/laggard or already traded today
            if (not is_bullish_leader and not is_bearish_laggard) or traded_today:
                continue

            # -------------------------------------------------------------
            # Phase 3: Confirmation Breakout Window (09:30 to 11:00 IST)
            # -------------------------------------------------------------
            if (hour > max_hour) or (hour == max_hour and minute > max_min):
                continue

            # Bullish Leader Breakout (LONG)
            if is_bullish_leader and (c_close > or15_high) and (c_close > c_open):
                signals[i] = 1.0
                stop_dist = max(c_close - or15_low, 0.05)
                stop_loss[i] = or15_low
                take_profit[i] = c_close + (target_rr * stop_dist)
                rationales[i] = (
                    f"Alpha 09 LONG: Leader Breakout Close={c_close:.1f} > OR15_High={or15_high:.1f} | "
                    f"RS={stock_rs_pct:+.2f}% vs Universe | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                )
                traded_today = True

            # Bearish Laggard Breakdown (SHORT)
            elif is_bearish_laggard and (c_close < or15_low) and (c_close < c_open):
                signals[i] = -1.0
                stop_dist = max(or15_high - c_close, 0.05)
                stop_loss[i] = or15_high
                take_profit[i] = c_close - (target_rr * stop_dist)
                rationales[i] = (
                    f"Alpha 09 SHORT: Laggard Breakdown Close={c_close:.1f} < OR15_Low={or15_low:.1f} | "
                    f"RS={stock_rs_pct:+.2f}% vs Universe | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                )
                traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
