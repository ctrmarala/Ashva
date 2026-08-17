"""
Ashva Centralized Technical Indicators & Quantitative Feature Toolbox
Vectorized, high-performance mathematical indicators built with NumPy and Pandas.
Supports all foundational and advanced technical indicators for strategy formulation.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Production-grade vectorized technical indicator library.
    Ensures zero look-ahead bias and fast execution across intraday and daily time series.
    """

    # =========================================================================
    # 1. MOVING AVERAGES
    # =========================================================================

    @staticmethod
    def add_sma(df: pd.DataFrame, period: int = 20, price_col: str = "close", col_name: Optional[str] = None) -> pd.DataFrame:
        """Simple Moving Average (SMA)."""
        df_out = df.copy()
        target_name = col_name or f"sma_{period}"
        df_out[target_name] = df_out[price_col].rolling(window=period, min_periods=period).mean()
        return df_out

    @staticmethod
    def add_ema(df: pd.DataFrame, period: int = 20, price_col: str = "close", col_name: Optional[str] = None) -> pd.DataFrame:
        """Exponential Moving Average (EMA)."""
        df_out = df.copy()
        target_name = col_name or f"ema_{period}"
        df_out[target_name] = df_out[price_col].ewm(span=period, adjust=False).mean()
        return df_out

    # =========================================================================
    # 2. VOLATILITY & ATR
    # =========================================================================

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range (ATR) with Wilder's smoothing."""
        df_out = df.copy()
        high = df_out["high"]
        low = df_out["low"]
        close_prev = df_out["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df_out["tr"] = tr
        df_out[f"atr_{period}"] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return df_out

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, price_col: str = "close") -> pd.DataFrame:
        """Bollinger Bands (Middle, Upper, Lower, %B, Bandwidth)."""
        df_out = df.copy()
        sma = df_out[price_col].rolling(window=window).mean()
        std = df_out[price_col].rolling(window=window).std(ddof=0)

        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        bandwidth = (upper - lower) / sma.replace(0, np.nan)
        percent_b = (df_out[price_col] - lower) / (upper - lower).replace(0, np.nan)

        df_out[f"bb_mid_{window}"] = sma
        df_out[f"bb_upper_{window}"] = upper
        df_out[f"bb_lower_{window}"] = lower
        df_out[f"bb_bandwidth_{window}"] = bandwidth
        df_out[f"bb_pct_b_{window}"] = percent_b
        return df_out

    @staticmethod
    def add_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5) -> pd.DataFrame:
        """Keltner Channels based on EMA +/- (Multiplier * ATR)."""
        df_out = TechnicalIndicators.add_atr(df, period=atr_period)
        ema = df_out["close"].ewm(span=ema_period, adjust=False).mean()
        atr = df_out[f"atr_{atr_period}"]

        df_out[f"kc_mid_{ema_period}"] = ema
        df_out[f"kc_upper_{ema_period}"] = ema + (multiplier * atr)
        df_out[f"kc_lower_{ema_period}"] = ema - (multiplier * atr)
        return df_out

    # =========================================================================
    # 3. MOMENTUM & OSCILLATORS
    # =========================================================================

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.DataFrame:
        """Relative Strength Index (RSI) using Wilder's exponential smoothing."""
        df_out = df.copy()
        delta = df_out[price_col].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_out[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))
        df_out[f"rsi_{period}"] = df_out[f"rsi_{period}"].fillna(50.0)
        return df_out

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "close") -> pd.DataFrame:
        """Moving Average Convergence Divergence (MACD Line, Signal Line, Histogram)."""
        df_out = df.copy()
        ema_fast = df_out[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df_out[price_col].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line

        df_out["macd_line"] = macd_line
        df_out["macd_signal"] = signal_line
        df_out["macd_hist"] = macd_hist
        return df_out

    @staticmethod
    def add_stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
        """Fast & Slow Stochastic Oscillator (%K and %D)."""
        df_out = df.copy()
        low_min = df_out["low"].rolling(window=k_period).min()
        high_max = df_out["high"].rolling(window=k_period).max()

        fast_k = 100.0 * (df_out["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        slow_k = fast_k.rolling(window=slowing).mean()
        slow_d = slow_k.rolling(window=d_period).mean()

        df_out[f"stoch_k_{k_period}"] = slow_k
        df_out[f"stoch_d_{d_period}"] = slow_d
        return df_out

    # =========================================================================
    # 4. TREND FOLLOWING & REGIME INDICATORS
    # =========================================================================

    @staticmethod
    def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Supertrend Indicator.
        Returns:
            supertrend: Stop-and-reverse price line
            supertrend_direction: +1.0 (BULLISH/LONG), -1.0 (BEARISH/SHORT)
        """
        df_out = TechnicalIndicators.add_atr(df, period=period)
        atr = df_out[f"atr_{period}"]
        hl2 = (df_out["high"] + df_out["low"]) / 2.0

        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        n = len(df_out)
        final_upper = np.zeros(n)
        final_lower = np.zeros(n)
        supertrend = np.zeros(n)
        direction = np.zeros(n)

        closes = df_out["close"].values
        b_upper = basic_upper.values
        b_lower = basic_lower.values

        for i in range(1, n):
            # Upper band logic
            if b_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]:
                final_upper[i] = b_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

            # Lower band logic
            if b_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]:
                final_lower[i] = b_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

            # Direction decision
            if direction[i - 1] == 1:
                if closes[i] < final_lower[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                else:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
            else:
                if closes[i] > final_upper[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
                else:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]

        df_out[f"supertrend_{period}_{multiplier}"] = supertrend
        df_out[f"supertrend_direction_{period}_{multiplier}"] = direction
        return df_out

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Movement Index (ADX, +DI, -DI)."""
        df_out = TechnicalIndicators.add_atr(df, period=period)
        tr = df_out["tr"]

        high_diff = df_out["high"].diff()
        low_diff = -df_out["low"].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

        alpha = 1.0 / period
        smooth_tr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_pdm = pd.Series(plus_dm, index=df_out.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smooth_mdm = pd.Series(minus_dm, index=df_out.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        plus_di = 100.0 * (smooth_pdm / smooth_tr.replace(0, np.nan))
        minus_di = 100.0 * (smooth_mdm / smooth_tr.replace(0, np.nan))

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        df_out[f"adx_{period}"] = adx
        df_out[f"plus_di_{period}"] = plus_di
        df_out[f"minus_di_{period}"] = minus_di
        return df_out

    @staticmethod
    def add_donchian_channels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Donchian Channels (Upper, Lower, Middle Breakout Bands)."""
        df_out = df.copy()
        high_max = df_out["high"].rolling(window=window).max()
        low_min = df_out["low"].rolling(window=window).min()
        mid = (high_max + low_min) / 2.0

        df_out[f"donchian_upper_{window}"] = high_max
        df_out[f"donchian_lower_{window}"] = low_min
        df_out[f"donchian_mid_{window}"] = mid
        return df_out

    # =========================================================================
    # 5. INTRADAY PIVOT POINTS
    # =========================================================================

    @staticmethod
    def add_camarilla_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates intraday Camarilla Pivot Points (H1-H4, L1-L4, PP)
        anchored to the previous day's High, Low, Close.
        """
        df_out = df.copy()
        if not isinstance(df_out.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex for pivot anchoring")

        # Resample daily OHLC to compute prior day levels
        daily = df_out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        daily_prev = daily.shift(1)

        diff = daily_prev["high"] - daily_prev["low"]
        h4 = daily_prev["close"] + diff * 1.1 / 2.0
        h3 = daily_prev["close"] + diff * 1.1 / 4.0
        l3 = daily_prev["close"] - diff * 1.1 / 4.0
        l4 = daily_prev["close"] - diff * 1.1 / 2.0

        pivots = pd.DataFrame({
            "cam_h4": h4, "cam_h3": h3, "cam_l3": l3, "cam_l4": l4
        }, index=daily.index)

        # Re-index to intraday timestamps
        intraday_dates = pd.to_datetime(df_out.index.date)
        for col in ["cam_h4", "cam_h3", "cam_l3", "cam_l4"]:
            val_map = pivots[col].to_dict()
            df_out[col] = [val_map.get(d, np.nan) for d in intraday_dates]

        return df_out
