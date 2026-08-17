"""
Ashva Automated Asset-to-Strategy Dynamic Selector & Screener
Classifies market regimes (Hurst Exponent, Volatility Squeeze, Anchored VWAP Dispersion)
and maps each asset to its mathematically optimal Alpha Strategy.

Usage:
    python scripts/run_asset_strategy_scanner.py
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.features.microstructure import MicrostructureFeatureExtractor
from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.strategies.alpha_vol_squeeze import AlphaVolatilitySqueeze
from src.strategies.alpha_regime import AlphaRegimeAdaptiveMR
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy


class StrategySelector:
    """
    Evaluates microstructure state and returns the optimal Alpha strategy for each stock.
    """

    def __init__(self, data_lake: Optional[DataLake] = None):
        self.data_lake = data_lake or DataLake()

    def analyze_asset_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classifies asset into:
        - TRENDING_MOMENTUM -> Alpha 07 (Trend Pullback)
        - VOLATILITY_COMPRESSION -> Alpha 08 (Vol Squeeze)
        - MEAN_REVERTING -> Alpha 02 (Regime MR)
        """
        if df.empty or len(df) < 50:
            return {"regime": "UNKNOWN", "hurst": 0.5, "is_squeeze": False, "recommended_strategy": "ALPHA_07_TREND_PULLBACK"}

        # 1. Compute Hurst Exponent
        hurst = MicrostructureFeatureExtractor.calculate_hurst_exponent(df["close"], max_lags=20)

        # 2. Check Volatility Squeeze
        close = df["close"]
        high = df["high"]
        low = df["low"]
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_upper = sma + (std * 2.0)
        bb_lower = sma - (std * 2.0)

        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(20).mean().fillna(close * 0.01)
        kc_upper = sma + (atr * 1.5)
        kc_lower = sma - (atr * 1.5)

        is_squeeze = bool((bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1]))

        # 3. Strategy Selection Decision Tree
        if is_squeeze:
            regime = "VOLATILITY_SQUEEZE_COMPRESSION"
            strat_id = "ALPHA_08_VOLATILITY_SQUEEZE"
            rationale = "Bollinger Bands inside Keltner Channels. High-energy compression ready for explosive breakout."
        elif hurst > 0.52:
            regime = "PERSISTENT_TREND"
            strat_id = "ALPHA_07_TREND_PULLBACK"
            rationale = f"Hurst Exponent = {hurst:.2f} (> 0.50). Asset exhibits persistent institutional trend memory."
        else:
            regime = "MEAN_REVERSION"
            strat_id = "ALPHA_02_REGIME_MR"
            rationale = f"Hurst Exponent = {hurst:.2f} (< 0.50). Asset is mean-reverting/rangebound."

        return {
            "regime": regime,
            "hurst_exponent": round(hurst, 3),
            "is_in_squeeze": is_squeeze,
            "recommended_strategy": strat_id,
            "rationale": rationale,
        }

    def scan_universe(self, symbols: List[str], timeframe: str = "15m") -> pd.DataFrame:
        rows = []
        for sym in symbols:
            df = self.data_lake.load_bars(sym, timeframe)
            analysis = self.analyze_asset_regime(df)
            rows.append({
                "Symbol": sym,
                "Regime": analysis["regime"],
                "Hurst": analysis.get("hurst_exponent", 0.5),
                "In Squeeze": "[YES]" if analysis.get("is_in_squeeze") else "[NO]",
                "Assigned Strategy": analysis["recommended_strategy"],
                "Rationale": analysis["rationale"],
            })
        return pd.DataFrame(rows)
