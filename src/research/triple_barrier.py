"""
Ashva Triple-Barrier Labeling Method
Implements Marcos López de Prado's Triple-Barrier Method for dynamic path-dependent trade labeling.
"""

from typing import Optional
import numpy as np
import pandas as pd


class TripleBarrierLabeler:
    """
    Labels financial time series using dynamic volatility-adjusted upper, lower, and time barriers.
    """

    @staticmethod
    def calculate_daily_volatility(close: pd.Series, span: int = 50) -> pd.Series:
        """
        Computes exponentially weighted moving standard deviation of log returns.
        """
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.ewm(span=span).std()
        return vol.bfill()

    @classmethod
    def apply_triple_barrier(
        cls,
        df: pd.DataFrame,
        pt_mult: float = 2.0,      # Profit-taking multiplier on volatility
        sl_mult: float = 1.0,      # Stop-loss multiplier on volatility
        max_holding_bars: int = 24, # Maximum duration (e.g. 24 bars = 2 hours on 5m)
        vol_span: int = 50,
    ) -> pd.DataFrame:
        """
        Evaluates trades path-dependently across top, bottom, and time barriers.
        
        :param df: DataFrame with 'close', 'high', 'low', 'signal' (+1, -1, or 0)
        :return: DataFrame containing barrier outcomes: 'ret', 'label' (+1, -1, 0), 'touch_time'
        """
        df_out = df.copy()
        vol = cls.calculate_daily_volatility(df_out["close"], span=vol_span)
        df_out["volatility"] = vol

        outcomes = []
        close_vals = df_out["close"].values
        high_vals = df_out["high"].values
        low_vals = df_out["low"].values
        sig_vals = df_out["signal"].values if "signal" in df_out.columns else np.zeros(len(df_out))
        vol_vals = vol.values
        indices = df_out.index

        n = len(df_out)

        for i in range(n):
            signal = sig_vals[i]
            if signal == 0:
                continue

            entry_price = close_vals[i]
            curr_vol = vol_vals[i]
            
            # Determine dynamic barrier distances
            upper_barrier = entry_price * (1.0 + pt_mult * curr_vol)
            lower_barrier = entry_price * (1.0 - sl_mult * curr_vol)
            
            end_idx = min(i + max_holding_bars, n - 1)
            
            exit_price = entry_price
            exit_time = indices[end_idx]
            label = 0  # 0 = vertical time barrier hit

            # Path-dependent scan forward
            for j in range(i + 1, end_idx + 1):
                cur_high = high_vals[j]
                cur_low = low_vals[j]

                if signal > 0:  # LONG Trade
                    if cur_high >= upper_barrier:
                        exit_price = upper_barrier
                        exit_time = indices[j]
                        label = 1
                        break
                    elif cur_low <= lower_barrier:
                        exit_price = lower_barrier
                        exit_time = indices[j]
                        label = -1
                        break
                elif signal < 0:  # SHORT Trade
                    if cur_low <= lower_barrier:
                        exit_price = lower_barrier
                        exit_time = indices[j]
                        label = 1
                        break
                    elif cur_high >= upper_barrier:
                        exit_price = upper_barrier
                        exit_time = indices[j]
                        label = -1
                        break

            # If time barrier hit without touching horizontal barriers
            if label == 0:
                exit_price = close_vals[end_idx]
                exit_time = indices[end_idx]

            raw_return = (exit_price - entry_price) / entry_price if signal > 0 else (entry_price - exit_price) / entry_price

            outcomes.append({
                "entry_time": indices[i],
                "exit_time": exit_time,
                "signal": signal,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "raw_return": raw_return,
                "label": label,  # +1: Profit hit, -1: Stop hit, 0: Time expired
            })

        return pd.DataFrame(outcomes) if outcomes else pd.DataFrame(columns=["entry_time", "exit_time", "signal", "entry_price", "exit_price", "raw_return", "label"])
