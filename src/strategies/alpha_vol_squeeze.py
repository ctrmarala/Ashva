"""
Ashva Alpha Strategy: Institutional Volatility Squeeze & Momentum Expansion
Captures explosive directional markup moves when Bollinger Bands contract inside Keltner Channels
and expand outward with institutional volume and order flow confirmation.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata


class AlphaVolatilitySqueeze(BaseStrategy, BaseHypothesis):
    """
    Bollinger-Keltner Volatility Squeeze with Asymmetric Momentum Expansion.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[HypothesisMetadata] = None,
    ):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_mult": 1.5,
            "momentum_period": 12,
            "risk_reward_ratio": 3.0,    # 1:3 Asymmetric Payoff
            "volume_threshold": 1.15,
        }
        params = {**default_params, **(parameters or {})}

        meta = metadata or HypothesisMetadata(
            hypothesis_id="ALPHA_08_VOLATILITY_SQUEEZE_EXPANSION",
            name="Institutional Volatility Squeeze & Momentum Expansion",
            category="VOLATILITY_BREAKOUT_ASYMMETRIC",
            economic_rationale="Markets alternate between low-volatility compression and explosive expansion. When Bollinger Bands contract inside Keltner Channels, pressure builds up, leading to high-momentum continuation upon squeeze release.",
            target_instruments=["RELIANCE", "ICICIBANK", "INFY", "TCS"],
            timeframe="15m",
            author="Ashva Quantitative Alpha Team",
        )

        BaseStrategy.__init__(self, strategy_id=meta.hypothesis_id, parameters=params)
        BaseHypothesis.__init__(self, metadata=meta, parameters=params)

    def formulate_signal_logic(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.Series:
        bb_n = parameters.get("bb_period", 20)
        bb_std = parameters.get("bb_std", 2.0)
        kc_n = parameters.get("kc_period", 20)
        kc_mult = parameters.get("kc_mult", 1.5)
        mom_n = parameters.get("momentum_period", 12)
        rr_ratio = parameters.get("risk_reward_ratio", 3.0)
        vol_thresh = parameters.get("volume_threshold", 1.15)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # 1. Bollinger Bands
        sma = close.rolling(window=bb_n, min_periods=bb_n//2).mean()
        std = close.rolling(window=bb_n, min_periods=bb_n//2).std()
        bb_upper = sma + (std * bb_std)
        bb_lower = sma - (std * bb_std)

        # 2. Keltner Channels (ATR based)
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(window=kc_n, min_periods=kc_n//2).mean().fillna(close * 0.01)
        kc_upper = sma + (atr * kc_mult)
        kc_lower = sma - (atr * kc_mult)

        # Squeeze Condition: Bollinger Bands inside Keltner Channels
        is_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # 3. Momentum Oscillator (Linear Regression of Close vs Mean)
        momentum = close - ((sma + (high.rolling(bb_n).max() + low.rolling(bb_n).min()) / 2) / 2)

        # 4. Volume Surge
        vol_ma = volume.rolling(window=20, min_periods=5).mean()
        vol_surge = (volume / vol_ma.replace(0, np.nan)).fillna(1.0)

        signals = pd.Series(0.0, index=data.index)
        current_pos = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        for i in range(25, len(data)):
            curr_c = close.iloc[i]
            curr_high = high.iloc[i]
            curr_low = low.iloc[i]
            curr_atr = atr.iloc[i]
            curr_mom = momentum.iloc[i]
            prev_mom = momentum.iloc[i-1]
            prev_squeeze = is_squeeze.iloc[i-1]
            curr_squeeze = is_squeeze.iloc[i]

            # Squeeze Fired: Previous bar was in squeeze, current bar broke out
            squeeze_fired = prev_squeeze and not curr_squeeze

            if current_pos == 0.0:
                # LONG Squeeze Breakout: Squeeze fired with positive momentum & volume surge
                if (squeeze_fired or (curr_mom > 0 and curr_mom > prev_mom)) and (curr_c > bb_upper.iloc[i]) and (vol_surge.iloc[i] >= vol_thresh):
                    current_pos = 1.0
                    entry_price = curr_c
                    stop_loss = entry_price - (1.2 * curr_atr)
                    take_profit = entry_price + (1.2 * curr_atr * rr_ratio)

                # SHORT Squeeze Breakout: Squeeze fired with negative momentum & volume surge
                elif (squeeze_fired or (curr_mom < 0 and curr_mom < prev_mom)) and (curr_c < bb_lower.iloc[i]) and (vol_surge.iloc[i] >= vol_thresh):
                    current_pos = -1.0
                    entry_price = curr_c
                    stop_loss = entry_price + (1.2 * curr_atr)
                    take_profit = entry_price - (1.2 * curr_atr * rr_ratio)

            elif current_pos == 1.0:
                # Exit conditions
                if curr_high >= take_profit or curr_low <= stop_loss or (curr_mom < prev_mom and curr_c < sma.iloc[i]):
                    current_pos = 0.0
                    entry_price = 0.0

            elif current_pos == -1.0:
                # Exit conditions
                if curr_low <= take_profit or curr_high >= stop_loss or (curr_mom > prev_mom and curr_c > sma.iloc[i]):
                    current_pos = 0.0
                    entry_price = 0.0

            signals.iloc[i] = current_pos

        return signals

    def get_parameter_grid(self) -> Dict[str, list]:
        return {
            "bb_std": [1.8, 2.0, 2.2],
            "kc_mult": [1.3, 1.5, 1.7],
            "risk_reward_ratio": [2.5, 3.0, 3.5],
        }

    def on_bar(self, bar: Any) -> Optional[Any]:
        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df["signal"] = self.formulate_signal_logic(df, self.parameters)
        return result_df
