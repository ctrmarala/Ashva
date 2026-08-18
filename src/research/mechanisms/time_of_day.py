"""
Ashva Market Mechanism: Time of Day Regimes
Implements hypothesis families based on intraday institutional flow timing.
e.g., Morning Trend -> Afternoon Reversion, Morning Range -> Afternoon Breakout.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, MarketMechanism, StrategyHorizon

class BaseTimeOfDayHypothesis(BaseHypothesis):
    """
    Evaluates market behavior based on distinct intraday time regimes.
    Splits the day into Morning (09:15-11:00), Midday (11:00-13:30), and Afternoon (13:30-15:30).
    """

    def __init__(self, metadata: Optional[HypothesisMetadata] = None, parameters: Optional[Dict[str, Any]] = None):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="MCH_01_TOD_FLOWS",
            name="Institutional Time of Day Flow Dislocation",
            category="MARKET_MECHANISM",
            economic_rationale="Institutional order execution algorithms often have strict VWAP/TWAP schedules, creating distinct liquidity and trend regimes in the morning vs afternoon.",
            target_instruments=["NIFTY", "BANKNIFTY", "RELIANCE"],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION
        )
        super().__init__(metadata=meta, parameters=parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "morning_trend_threshold_pct": [0.5, 0.75, 1.0],
            "afternoon_start_hour": [13, 14],
            "afternoon_start_minute": [0, 30],
            "reversion_target_multiplier": [0.5, 1.0],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Example Logic: If morning moves > threshold, expect afternoon mean reversion.
        """
        out = df.copy()
        out["signal"] = 0.0
        
        if out.empty:
            return out
            
        params = self.parameters or {k: v[0] for k, v in self.get_parameter_grid().items()}
        thresh = params.get("morning_trend_threshold_pct", 0.75) / 100.0
        aft_hr = params.get("afternoon_start_hour", 13)
        aft_min = params.get("afternoon_start_minute", 30)
        
        # Calculate daily cumulative return up to the afternoon start
        out["date"] = out.index.date
        
        grouped = out.groupby("date")
        
        for date, group in grouped:
            # Find open price
            if group.empty:
                continue
                
            open_price = group.iloc[0]["open"]
            
            # Find state at afternoon start
            afternoon_mask = (group.index.hour > aft_hr) | ((group.index.hour == aft_hr) & (group.index.minute >= aft_min))
            morning_data = group[~afternoon_mask]
            
            if morning_data.empty:
                continue
                
            morning_close = morning_data.iloc[-1]["close"]
            morning_ret = (morning_close - open_price) / open_price
            
            # Generate afternoon signal
            if abs(morning_ret) > thresh:
                # Strong morning trend -> expect reversion
                signal_val = -1.0 if morning_ret > 0 else 1.0
                
                # Apply signal to the first bar of the afternoon
                afternoon_idx = group[afternoon_mask].index
                if len(afternoon_idx) > 0:
                    out.loc[afternoon_idx[0], "signal"] = signal_val
                    
        return out
