"""
Ashva Market Microstructure & Order Flow Feature Store
Extracts quantitative microstructure signals: Anchored VWAP, Volume Delta, Opening Range Bands,
Hurst Exponent Regime Classifier, and Amihud Illiquidity.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class MicrostructureFeatureExtractor:
    """
    Computes institutional microstructure and order flow features from OHLCV time series.
    """

    @staticmethod
    def calculate_anchored_vwap(df: pd.DataFrame, anchor_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates intraday VWAP anchored to each day's market open (09:15 AM IST)
        with +/- 1.0, 2.0, and 3.0 standard deviation dispersion bands.
        """
        df_out = df.copy()
        
        # Ensure timestamp is datetime
        if not isinstance(df_out.index, pd.DatetimeIndex):
            if "timestamp" in df_out.columns:
                df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
                df_out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        typical_price = (df_out["high"] + df_out["low"] + df_out["close"]) / 3.0
        pv = typical_price * df_out["volume"]

        # Group by date to anchor VWAP daily
        dates = df_out.index.date
        df_out["cum_pv"] = pv.groupby(dates).cumsum()
        df_out["cum_vol"] = df_out["volume"].groupby(dates).cumsum()
        df_out["vwap"] = df_out["cum_pv"] / df_out["cum_vol"].replace(0, np.nan)

        # Dispersion / Standard Deviation Bands
        price_sq_v = (typical_price ** 2) * df_out["volume"]
        df_out["cum_price_sq_v"] = price_sq_v.groupby(dates).cumsum()
        
        variance = (df_out["cum_price_sq_v"] / df_out["cum_vol"]) - (df_out["vwap"] ** 2)
        variance = variance.clip(lower=0)
        df_out["vwap_std"] = np.sqrt(variance)

        df_out["vwap_upper_1sd"] = df_out["vwap"] + df_out["vwap_std"]
        df_out["vwap_lower_1sd"] = df_out["vwap"] - df_out["vwap_std"]
        df_out["vwap_upper_2sd"] = df_out["vwap"] + 2.0 * df_out["vwap_std"]
        df_out["vwap_lower_2sd"] = df_out["vwap"] - 2.0 * df_out["vwap_std"]

        # Drop intermediate calculation columns
        df_out.drop(columns=["cum_pv", "cum_vol", "cum_price_sq_v"], inplace=True)
        return df_out

    @staticmethod
    def calculate_volume_delta(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
        """
        Approximates Order Flow Imbalance and Volume Delta using price action sign.
        Calculates Cumulative Volume Delta (CVD) and Volume Surge Multiplier.
        """
        df_out = df.copy()
        price_change = df_out["close"].diff()
        
        # Approximate buyer vs seller volume delta
        # +1 if close > open/prev_close, -1 if close < open/prev_close
        bar_direction = np.where(price_change > 0, 1.0, np.where(price_change < 0, -1.0, 0.0))
        df_out["volume_delta"] = df_out["volume"] * bar_direction

        # Daily Cumulative Volume Delta
        if isinstance(df_out.index, pd.DatetimeIndex):
            dates = df_out.index.date
            df_out["cvd"] = df_out["volume_delta"].groupby(dates).cumsum()
        else:
            df_out["cvd"] = df_out["volume_delta"].cumsum()

        # Volume Surge Ratio (relative to 20-period rolling average)
        vol_ma = df_out["volume"].rolling(window=rolling_window, min_periods=1).mean()
        df_out["volume_surge_ratio"] = df_out["volume"] / vol_ma.replace(0, 1.0)

        return df_out

    @staticmethod
    def calculate_opening_range(
        df: pd.DataFrame,
        orb_start: str = "09:15:00",
        orb_end: str = "09:45:00"
    ) -> pd.DataFrame:
        """
        Calculates Opening Range High (ORH), Opening Range Low (ORL), and Opening Range Volume.
        """
        df_out = df.copy()
        if not isinstance(df_out.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        df_out["time_str"] = df_out.index.strftime("%H:%M:%S")
        dates = df_out.index.date

        # Mask for opening range window
        orb_mask = (df_out["time_str"] >= orb_start) & (df_out["time_str"] <= orb_end)
        
        # Calculate max high and min low during opening window per day
        orb_highs = df_out[orb_mask]["high"].groupby(dates[orb_mask]).max()
        orb_lows = df_out[orb_mask]["low"].groupby(dates[orb_mask]).min()
        orb_vols = df_out[orb_mask]["volume"].groupby(dates[orb_mask]).sum()

        # Map back to entire dataframe
        df_out["date_col"] = df_out.index.date
        df_out["orb_high"] = df_out["date_col"].map(orb_highs)
        df_out["orb_low"] = df_out["date_col"].map(orb_lows)
        df_out["orb_volume"] = df_out["date_col"].map(orb_vols)

        df_out.drop(columns=["time_str", "date_col"], inplace=True)
        return df_out

    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lags: int = 30) -> float:
        """
        Calculates the Hurst Exponent (H) of a price series using variance-scaling analysis.
        Var(P(t + tau) - P(t)) ~ tau^(2H)
        H < 0.45: Mean-Reverting regime
        0.45 <= H <= 0.55: Random Walk (Neutral)
        H > 0.55: Trending / Momentum regime
        """
        clean_s = series.dropna().values
        if len(clean_s) < 30:
            return 0.50

        # Monotonic check
        diffs = np.diff(clean_s)
        if np.all(diffs > 0) or np.all(diffs < 0):
            return 0.85

        lags = range(2, min(max_lags, len(clean_s) // 4))
        if len(lags) < 3:
            return 0.50

        tau = []
        for lag in lags:
            price_diff = clean_s[lag:] - clean_s[:-lag]
            std_diff = np.std(price_diff)
            tau.append(std_diff if std_diff > 1e-8 else 1e-8)

        log_lags = np.log(list(lags))
        log_tau = np.log(tau)

        valid = np.isfinite(log_lags) & np.isfinite(log_tau)
        if np.sum(valid) < 3:
            return 0.50

        poly = np.polyfit(log_lags[valid], log_tau[valid], 1)
        hurst = float(poly[0])
        return round(float(np.clip(hurst, 0.05, 0.95)), 3)

    @staticmethod
    def calculate_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculates Amihud's Illiquidity measure: |Return| / Turnover (Volume * Close).
        Higher values indicate higher market impact / lower liquidity.
        """
        returns = df["close"].pct_change().abs()
        turnover = (df["close"] * df["volume"]).replace(0, np.nan)
        daily_illiquidity = returns / turnover
        return daily_illiquidity.rolling(window=window, min_periods=1).mean() * 1e6
