import pandas as pd
from typing import Dict, Any, Optional
from src.strategies.base import BaseStrategy
from src.core.events import BarEvent, SignalEvent
from src.research.hypothesis import HypothesisMetadata, StrategyHorizon, MarketMechanism

HYPOTHESIS = HypothesisMetadata(
    hypothesis_id="alpha_000_dummy",
    name="Dummy Alpha Baseline Test",
    category="STATISTICAL_REVERSION",
    economic_rationale="Baseline testing hypothesis to verify end-to-end Alpha Factory components without producing actual trading logic. Validates signal generation, ledger persistence, timeframe discovery, and canonical evidence logging.",
    target_instruments=["RELIANCE", "INFY", "HDFCBANK", "TCS", "ITC", "ICICIBANK", "SBIN"],
    timeframe="15m",
    horizon=StrategyHorizon.INTRADAY,
    mechanism=MarketMechanism.MEAN_REVERSION,
    author="AshvaQuantLab"
)

class Alpha000Dummy(BaseStrategy):
    """
    Dummy strategy for validating factory readiness.
    """
    def __init__(self, strategy_id: str = "alpha_000_dummy", parameters: Optional[Dict[str, Any]] = None):
        super().__init__(strategy_id, parameters)
        self.metadata = HYPOTHESIS

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0.0
        
        # Simple dummy logic just to generate trades for the lab to process
        # Using a slight lag to generate some artificial edges or just naive mean reversion
        returns = df["close"].pct_change()
        
        # Buy on drops, short on pops (naive mean reversion)
        df.loc[returns < -0.005, "signal"] = 1.0
        df.loc[returns > 0.005, "signal"] = -1.0
        
        return df

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        return None
