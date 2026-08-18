"""
Ashva Market Regime & Cross-Sectional Feature Extractor
Vectorized quantitative module for extracting market-derived regime features:
- Normalized Volatility (ATR% and Realized Volatility Percentile)
- Overnight Gap Structure & Intraday Gap Exhaustion Ratios
- Time-Of-Day (TOD) Relative Volume (RVOL)
- Multi-Session Structural Autocorrelation & Trend Legs
- Cross-Sectional Relative Strength vs Index/Benchmark

Strict Zero Look-Ahead Bias: All daily features use strictly completed prior session bars (Shifted 1 session).
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


class MarketRegimeFeatureExtractor:
    """
    Computes institutional market-derived regime signatures and asset characteristics from OHLCV data.
    """

    @staticmethod
    def compute_normalized_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculates ATR normalized by the closing price: ATR(t) / Close(t).
        Allows cross-asset comparison of volatility regardless of absolute nominal stock price.
        """
        high = df["high"]
        low = df["low"]
        close_prev = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        atr_pct = (atr / df["close"].replace(0, np.nan)) * 100.0
        return atr_pct.fillna(0.0)

    @staticmethod
    def compute_realized_volatility(df: pd.DataFrame, window: int = 20, periods_per_year: int = 252 * 25) -> pd.Series:
        """
        Calculates rolling realized volatility from log returns.
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        rolling_std = log_ret.rolling(window=window, min_periods=max(5, window // 2)).std()
        ann_vol = rolling_std * np.sqrt(periods_per_year)
        return ann_vol.fillna(0.0)

    @staticmethod
    def compute_tod_relative_volume(df: pd.DataFrame, lookback_sessions: int = 20) -> pd.Series:
        """
        Calculates Time-Of-Day (TOD) Relative Volume (RVOL):
        RVOL(t) = Volume(t) / Mean(Volume for this exact time-of-day over past N sessions)
        """
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        times = out.index.time
        out["_tod"] = times
        out["_date"] = out.index.date

        # Pivot to session-by-time grid
        pivot = out.pivot_table(index="_date", columns="_tod", values="volume", aggfunc="first")
        rolling_tod_mean = pivot.shift(1).rolling(window=lookback_sessions, min_periods=5).mean()

        # Unpivot and map back
        stacked = rolling_tod_mean.stack().rename("_baseline_vol")
        merged = out.join(stacked, on=["_date", "_tod"])
        
        rvol = (merged["volume"] / merged["_baseline_vol"].replace(0, np.nan)).fillna(1.0)
        return rvol

    @staticmethod
    def extract_session_gap_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts session-level overnight gap features for each intraday bar:
        - prev_day_close: Prior day final bar close (strictly T-1)
        - prev_day_atr: Prior day 14-session ATR
        - gap_pct: (Open_09:15 - prev_day_close) / prev_day_close * 100.0
        - gap_atr_ratio: |Open_09:15 - prev_day_close| / prev_day_atr
        - bar1_body_ratio: |Close_09:15 - Open_09:15| / (High_09:15 - Low_09:15)
        - is_gap_direction_bullish: True if Open_09:15 > prev_day_close
        """
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        dates = pd.Series(out.index.date, index=out.index)
        times = [t.strftime("%H:%M") for t in out.index.time]

        # Daily canvas strictly from completed sessions
        daily = out.groupby(dates).agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )

        prev_close = daily["day_close"].shift(1)
        prev_high = daily["day_high"].shift(1)
        prev_low = daily["day_low"].shift(1)
        prev_prev_close = daily["day_close"].shift(2)

        tr1 = prev_high - prev_low
        tr2 = (prev_high - prev_prev_close).abs()
        tr3 = (prev_low - prev_prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()

        gap_abs = daily["day_open"] - prev_close
        gap_pct = (gap_abs / prev_close.replace(0, np.nan)) * 100.0
        gap_atr_ratio = gap_abs.abs() / daily_atr.replace(0, np.nan)

        # Bar 1 (09:15-09:30) body metrics
        is_bar1 = pd.Series(times, index=out.index) == "09:15"
        bar1_df = out[is_bar1]
        bar1_dates = bar1_df.index.date
        bar1_range = (bar1_df["high"] - bar1_df["low"]).replace(0, np.nan)
        bar1_body = (bar1_df["close"] - bar1_df["open"]).abs()
        bar1_body_ratio_s = pd.Series((bar1_body / bar1_range).fillna(0.0).values, index=bar1_dates)
        bar1_is_bullish_s = pd.Series((bar1_df["close"] > bar1_df["open"]).values, index=bar1_dates)

        # Vectorized mapping onto intraday bars
        out["prev_close"] = dates.map(prev_close)
        out["prev_daily_atr"] = dates.map(daily_atr)
        out["gap_pct"] = dates.map(gap_pct)
        out["gap_atr_ratio"] = dates.map(gap_atr_ratio)
        out["bar1_body_ratio"] = dates.map(bar1_body_ratio_s).fillna(0.0)
        out["bar1_is_bullish"] = dates.map(bar1_is_bullish_s).fillna(False)

        return out

    @staticmethod
    def compute_relative_strength(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Calculates rolling Relative Strength (RS) of stock vs benchmark:
        RS = Stock_Return(window) - Benchmark_Return(window)
        Positive RS = Stock is outperforming benchmark.
        """
        s_close = stock_df["close"]
        b_close = benchmark_df["close"]

        # Align timestamps
        aligned_b = b_close.reindex(stock_df.index, method="ffill")

        s_ret = s_close.pct_change(window)
        b_ret = aligned_b.pct_change(window)

        rs = (s_ret - b_ret).fillna(0.0)
        return rs

    @staticmethod
    def compute_multiday_trend_structure(df: pd.DataFrame, lookback_days: int = 3) -> pd.DataFrame:
        """
        Identifies multi-session structural persistence (e.g. 3 consecutive higher lows or lower highs).
        Strictly shifted 1 session (zero look-ahead).
        """
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        dates = pd.Series(out.index.date, index=out.index)
        daily = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )

        prev_low = daily["day_low"].shift(1)
        prev_low_2 = daily["day_low"].shift(2)
        prev_low_3 = daily["day_low"].shift(3)

        prev_high = daily["day_high"].shift(1)
        prev_high_2 = daily["day_high"].shift(2)
        prev_high_3 = daily["day_high"].shift(3)

        is_bullish_3d = (prev_low > prev_low_2) & (prev_low_2 > prev_low_3)
        is_bearish_3d = (prev_high < prev_high_2) & (prev_high_2 < prev_high_3)

        out["is_3d_bullish_trend"] = dates.map(is_bullish_3d).fillna(False)
        out["is_3d_bearish_trend"] = dates.map(is_bearish_3d).fillna(False)

        return out
