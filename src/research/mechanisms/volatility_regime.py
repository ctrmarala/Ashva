"""
Ashva Market Mechanism: Volatility Regimes
Implements hypothesis families based on volatility contraction (squeeze) and expansion.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, MarketMechanism, StrategyHorizon

class BaseVolatilityRegimeHypothesis(BaseHypothesis):
    """
    Evaluates market behavior based on volatility regimes (e.g., Bollinger Bands inside Keltner Channels).
    Hypothesis: Low volatility contraction leads to high volatility breakout expansion.
    """

    def __init__(self, metadata: Optional[HypothesisMetadata] = None, parameters: Optional[Dict[str, Any]] = None):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="MCH_02_VOL_REGIME",
            name="Volatility Contraction Expansion Cycle",
            category="MARKET_MECHANISM",
            economic_rationale="Markets alternate between periods of high and low volatility. Prolonged periods of low volatility (squeeze) build up energy that eventually releases in a directional breakout.",
            target_instruments=["NIFTY", "BANKNIFTY", "RELIANCE"],
            timeframe="15m",
            horizon=StrategyHorizon.SWING,
            mechanism=MarketMechanism.BREAKOUT
        )
        super().__init__(metadata=meta, parameters=parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "lookback_period": [20, 50],
            "bb_std": [2.0],
            "kc_atr_mult": [1.5, 2.0],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Example Logic: Enter long/short on breakout from a volatility squeeze.
        """
        out = df.copy()
        out["signal"] = 0.0
        
        if out.empty or len(out) < 50:
            return out
            
        params = self.parameters or {k: v[0] for k, v in self.get_parameter_grid().items()}
        lookback = params.get("lookback_period", 20)
        bb_mult = params.get("bb_std", 2.0)
        kc_mult = params.get("kc_atr_mult", 1.5)
        
        close = out["close"]
        high = out["high"]
        low = out["low"]
        
        sma = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        
        bb_upper = sma + (std * bb_mult)
        bb_lower = sma - (std * bb_mult)
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(lookback).mean().fillna(close * 0.01)
        
        kc_upper = sma + (atr * kc_mult)
        kc_lower = sma - (atr * kc_mult)
        
        # Squeeze condition: BB is entirely inside KC
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        # Breakout signal: Price closes outside BB, and we were recently in a squeeze
        squeeze_recently = squeeze_on.rolling(5).max() > 0
        
        long_breakout = (close > bb_upper) & squeeze_recently & (close.shift(1) <= bb_upper.shift(1))
        short_breakout = (close < bb_lower) & squeeze_recently & (close.shift(1) >= bb_lower.shift(1))
        
        out.loc[long_breakout, "signal"] = 1.0
        out.loc[short_breakout, "signal"] = -1.0
                    
        return out
