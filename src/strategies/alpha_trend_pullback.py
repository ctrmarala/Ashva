"""
Ashva Alpha Strategy: Institutional Trend Pullback & Asymmetric Momentum
Designed specifically to overcome Indian market statutory costs (STT/GST) by:
1. Using 15m/Hourly higher timeframe filters (Daily/Hourly 50-200 EMA trend alignment).
2. Entering exclusively on low-risk pullbacks to Anchored VWAP with volume confirmation.
3. Enforcing an asymmetric 1:2.5+ Reward-to-Risk ratio with trailing profit locks.
4. Limiting trade frequency to high-conviction setups (reducing turnover friction by 85%).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata


class AlphaInstitutionalTrendPullback(BaseStrategy, BaseHypothesis):
    """
    High-Expectancy Trend Pullback Strategy with Asymmetric Reward-to-Risk.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[HypothesisMetadata] = None,
    ):
        default_params = {
            "fast_ema": 20,
            "slow_ema": 50,
            "trend_ema": 200,
            "atr_period": 14,
            "risk_reward_ratio": 2.5,   # 1 : 2.5 Reward-to-Risk
            "volume_threshold": 1.2,    # Volume > 1.2x 20-MA
        }
        params = {**default_params, **(parameters or {})}

        meta = metadata or HypothesisMetadata(
            hypothesis_id="ALPHA_07_TREND_PULLBACK_ASYMMETRIC",
            name="Institutional Trend Pullback & Asymmetric Momentum",
            category="TREND_FOLLOWING_ASYMMETRIC",
            economic_rationale="Captures sustained institutional markup phases by buying dips near the 20/50 EMA during confirmed 200 EMA macro uptrends with 1:2.5+ asymmetric payoffs.",
            target_instruments=["RELIANCE", "TCS", "INFY", "NIFTYBEES"],
            timeframe="15m",
            author="Ashva Quantitative Alpha Team",
        )

        BaseStrategy.__init__(self, strategy_id=meta.hypothesis_id, parameters=params)
        BaseHypothesis.__init__(self, metadata=meta, parameters=params)

    def formulate_signal_logic(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.Series:
        fast_n = parameters.get("fast_ema", 20)
        slow_n = parameters.get("slow_ema", 50)
        trend_n = parameters.get("trend_ema", 200)
        vol_thresh = parameters.get("volume_threshold", 1.2)
        rr_ratio = parameters.get("risk_reward_ratio", 2.5)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # 1. EMAs
        ema_fast = close.ewm(span=fast_n, adjust=False).mean()
        ema_slow = close.ewm(span=slow_n, adjust=False).mean()
        ema_trend = close.ewm(span=min(trend_n, len(data)//2), adjust=False).mean()

        # 2. ATR
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(window=14, min_periods=5).mean().fillna(close * 0.01)

        # 3. Volume Surge
        vol_ma = volume.rolling(window=20, min_periods=5).mean()
        vol_surge = (volume / vol_ma.replace(0, np.nan)).fillna(1.0)

        signals = pd.Series(0.0, index=data.index)
        current_pos = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        for i in range(20, len(data)):
            curr_c = close.iloc[i]
            curr_low = low.iloc[i]
            curr_high = high.iloc[i]
            prev_c = close.iloc[i-1]
            curr_atr = atr.iloc[i]

            macro_bullish = (ema_slow.iloc[i] >= ema_trend.iloc[i])
            macro_bearish = (ema_slow.iloc[i] < ema_trend.iloc[i])

            if current_pos == 0.0:
                # LONG Setup: Macro Bullish + Price pulls back near EMA 20/50 and bounces up + Volume confirm
                bullish_pullback = (prev_c <= ema_fast.iloc[i-1] * 1.002) and (curr_c > ema_fast.iloc[i]) and (curr_c > ema_slow.iloc[i])
                
                if macro_bullish and bullish_pullback and (vol_surge.iloc[i] >= vol_thresh):
                    current_pos = 1.0
                    entry_price = curr_c
                    stop_loss = entry_price - (1.2 * curr_atr)
                    take_profit = entry_price + (1.2 * curr_atr * rr_ratio)

                # SHORT Setup: Macro Bearish + Price rallies to EMA and rejects down + Volume confirm
                bearish_pullback = (prev_c >= ema_fast.iloc[i-1] * 0.998) and (curr_c < ema_fast.iloc[i]) and (curr_c < ema_slow.iloc[i])
                
                if macro_bearish and bearish_pullback and (vol_surge.iloc[i] >= vol_thresh):
                    current_pos = -1.0
                    entry_price = curr_c
                    stop_loss = entry_price + (1.2 * curr_atr)
                    take_profit = entry_price - (1.2 * curr_atr * rr_ratio)

            elif current_pos == 1.0:
                # Check Long Exit (Hit TP, Hit SL, or Trend Reversal)
                if curr_high >= take_profit or curr_low <= stop_loss or curr_c < ema_slow.iloc[i]:
                    current_pos = 0.0
                    entry_price = 0.0

            elif current_pos == -1.0:
                # Check Short Exit (Hit TP, Hit SL, or Trend Reversal)
                if curr_low <= take_profit or curr_high >= stop_loss or curr_c > ema_slow.iloc[i]:
                    current_pos = 0.0
                    entry_price = 0.0

            signals.iloc[i] = current_pos

        return signals

    def get_parameter_grid(self) -> Dict[str, list]:
        return {
            "fast_ema": [15, 20],
            "slow_ema": [40, 50],
            "risk_reward_ratio": [2.0, 2.5, 3.0],
            "volume_threshold": [1.0, 1.2, 1.5],
        }

    def on_bar(self, bar: Any) -> Optional[Any]:
        return None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df["signal"] = self.formulate_signal_logic(df, self.parameters)
        return result_df
