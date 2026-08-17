"""
Ashva Base Strategy Interface
Provides standardized signal generation, event hooks, and parameter schemas for production trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from src.core.events import BarEvent, SignalEvent, SignalType


class BaseStrategy(ABC):
    """
    Standard interface for all rule-based, statistical, and ML strategies.
    """

    def __init__(self, strategy_id: str, parameters: Optional[Dict[str, Any]] = None):
        self.strategy_id = strategy_id
        self.parameters = parameters or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes historical or streaming DataFrame and returns DataFrame with 'signal' column.
        Signal values: +1.0 (LONG), -1.0 (SHORT), 0.0 (FLAT / EXIT).
        """
        pass

    @abstractmethod
    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Real-time event handler called on each incoming BarEvent.
        """
        pass
