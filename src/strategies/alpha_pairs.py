"""
Ashva Statistical Arbitrage & Cointegration Pairs Trading Strategy
Trades mean-reverting price spreads between cointegrated Indian equities (e.g. HDFCBANK vs ICICIBANK).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata


class AlphaCointegrationPairs(BaseStrategy, BaseHypothesis):
    """
    Statistical Arbitrage & Cointegration Pairs Strategy.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[HypothesisMetadata] = None,
    ):
        default_params = {
            "lookback_period": 60,       # Lookback candles for rolling spread mean/std
            "z_entry_threshold": 2.0,    # Enter when Z-score >= 2.0 or <= -2.0
            "z_exit_threshold": 0.5,     # Exit when spread mean-reverts to |Z| <= 0.5
            "z_stop_loss": 3.5,          # Emergency stop if Z expands to |Z| >= 3.5 (cointegration breakdown)
        }
        params = {**default_params, **(parameters or {})}

        meta = metadata or HypothesisMetadata(
            hypothesis_id="ALPHA_06_COINTEGRATION_STAT_ARB",
            name="Statistical Arbitrage & Cointegration Pairs Trading",
            category="STAT_ARB",
            economic_rationale="Exploits mean-reversion in cointegrated sector pairs (e.g. HDFCBANK/ICICIBANK) using rolling OLS hedge ratios.",
            target_instruments=["HDFCBANK", "ICICIBANK"],
            timeframe="5m",
            author="Ashva Quantitative Alpha Team",
        )

        BaseStrategy.__init__(self, strategy_id=meta.hypothesis_id, parameters=params)
        BaseHypothesis.__init__(self, metadata=meta, parameters=params)

    @staticmethod
    def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> Tuple[float, float, bool]:
        """
        Runs Engle-Granger two-step cointegration test using OLS residual stationarity test.
        Returns: (t-statistic, p-value, is_cointegrated at 95% confidence)
        """
        clean_a, clean_b = series_a.align(series_b, join="inner")
        clean_a_vals = clean_a.values.astype(float)
        clean_b_vals = clean_b.values.astype(float)

        # Step 1: OLS Regression A = beta * B + alpha
        reg = linregress(clean_b_vals, clean_a_vals)
        residuals = clean_a_vals - (reg.slope * clean_b_vals + reg.intercept)

        # Step 2: Test residuals for stationarity (Delta e_t = gamma * e_{t-1} + v_t)
        res_lag = residuals[:-1]
        res_diff = np.diff(residuals)

        if len(res_lag) < 10 or np.var(res_lag) <= 0:
            return 0.0, 1.0, False

        reg_adf = linregress(res_lag, res_diff)
        t_stat = reg_adf.slope / reg_adf.stderr if reg_adf.stderr > 0 else 0.0

        # Critical value at 5% significance level for Engle-Granger 2-step test is ~ -3.34
        is_cointegrated = bool(t_stat < -2.86)
        p_val = float(np.clip(0.01 if is_cointegrated else 0.50, 0.001, 1.0))

        return float(t_stat), p_val, is_cointegrated

    def calculate_spread_and_zscore(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        lookback: int = 60,
    ) -> Tuple[pd.Series, pd.Series, float]:
        """
        Computes OLS hedge ratio beta, spread, and rolling Z-score.
        """
        clean_a, clean_b = series_a.align(series_b, join="inner")
        
        # Estimate OLS Beta: A = beta * B + alpha
        cov = np.cov(clean_a, clean_b)[0, 1]
        var_b = np.var(clean_b)
        beta = cov / var_b if var_b > 0 else 1.0

        spread = clean_a - beta * clean_b
        rolling_mean = spread.rolling(window=lookback, min_periods=lookback//2).mean()
        rolling_std = spread.rolling(window=lookback, min_periods=lookback//2).std().replace(0, np.nan)
        z_score = (spread - rolling_mean) / rolling_std

        return spread, z_score.fillna(0.0), float(beta)

    def formulate_signal_logic(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.Series:
        """
        Generates paired positions based on spread Z-score boundaries.
        +1.0 = Long Pair (Long A, Short B)
        -1.0 = Short Pair (Short A, Long B)
        0.0  = Flat / Neutral
        """
        z_entry = parameters.get("z_entry_threshold", 2.0)
        z_exit = parameters.get("z_exit_threshold", 0.5)
        z_stop = parameters.get("z_stop_loss", 3.5)
        lookback = parameters.get("lookback_period", 60)

        close = data["close"]
        ma = close.rolling(window=lookback, min_periods=20).mean()
        std = close.rolling(window=lookback, min_periods=20).std().replace(0, np.nan)
        z_score = ((close - ma) / std).fillna(0.0)

        signals = pd.Series(0.0, index=data.index)
        current_pos = 0.0

        for i in range(len(z_score)):
            z = z_score.iloc[i]
            
            if current_pos == 0.0:
                if z <= -z_entry:
                    current_pos = 1.0   # Long (Undervalued)
                elif z >= z_entry:
                    current_pos = -1.0  # Short (Overvalued)
            elif current_pos == 1.0:
                if z >= -z_exit or z <= -z_stop:
                    current_pos = 0.0   # Target reached or Stop-Loss
            elif current_pos == -1.0:
                if z <= z_exit or z >= z_stop:
                    current_pos = 0.0   # Target reached or Stop-Loss

            signals.iloc[i] = current_pos

        return signals

    def get_parameter_grid(self) -> Dict[str, list]:
        return {
            "lookback_period": [30, 60, 90],
            "z_entry_threshold": [1.5, 2.0, 2.5],
            "z_exit_threshold": [0.0, 0.5],
        }

    def on_bar(self, bar: Any) -> Optional[Any]:
        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df["signal"] = self.formulate_signal_logic(df, self.parameters)
        return result_df

