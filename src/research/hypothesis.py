"""
Ashva Quantitative Alpha Hypothesis Framework
Defines structured, scientific hypothesis contracts with explicit economic rationale and validation lifecycle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
import pandas as pd


class HypothesisStatus(str, Enum):
    DISCOVERY = "DISCOVERY"
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    FORWARD_PAPER = "FORWARD_PAPER"
    CAPITAL_CANDIDATE = "CAPITAL_CANDIDATE"
    LOW_FREQUENCY_WATCHLIST = "LOW_FREQUENCY_WATCHLIST"
    DECAYING_WATCHLIST = "DECAYING_WATCHLIST"
    REJECTED = "REJECTED"
    # Backward compatibility
    FORMULATED = "DISCOVERY"
    TESTING = "RESEARCH_CANDIDATE"
    ACCEPTED = "CAPITAL_CANDIDATE"


class StrategyHorizon(str, Enum):
    INTRADAY = "INTRADAY"     # Minutes to hours, 15:15 EOD square-off
    SWING = "SWING"           # 1 to 10 trading days, overnight delivery
    POSITIONAL = "POSITIONAL" # Weeks to months


class MarketMechanism(str, Enum):
    MOMENTUM = "MOMENTUM"
    RANGE = "RANGE"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    RELATIVE_VALUE = "RELATIVE_VALUE"
    VOLATILITY = "VOLATILITY"
    EVENT = "EVENT"


@dataclass
class HypothesisMetadata:
    hypothesis_id: str
    name: str
    category: str              # e.g., "MICROSTRUCTURE_FLOW", "REGIME_SWITCHING", "STAT_ARB"
    economic_rationale: str    # Why does this market inefficiency exist?
    target_instruments: List[str]
    timeframe: str
    horizon: StrategyHorizon = StrategyHorizon.INTRADAY
    mechanism: MarketMechanism = MarketMechanism.MOMENTUM
    author: str = "AshvaQuantLab"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HypothesisValidationReport:
    hypothesis_id: str
    status: HypothesisStatus
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    deflated_sharpe_p_value: float    # p <= 0.05 required
    cpcv_mean_sharpe: float
    cpcv_degradation_pct: float       # must be < 60%
    monte_carlo_95_max_dd_pct: float  # must be < 15%
    net_profit_factor_post_tax: float # dynamic hurdle 1.08 - 1.20
    rejection_reasons: List[str] = field(default_factory=list)
    tested_trials_count: int = 1
    evidence_tier: str = "PRELIMINARY"       # LOW_SAMPLE, PRELIMINARY, MODERATE, STRONG
    regime_stability_score: float = 0.0      # 0 to 100% across historical windows
    current_regime_score: float = 0.0        # 0 to 100% descriptive indicator of 60d trajectory
    recency_weighted_score: float = 0.0      # Sample-confidence weighted return metric (-1.0 to +1.0)
    window_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict) # 60d, 180d, 365d, 540d
    portfolio_correlation: float = 0.0       # Max absolute correlation with baseline portfolio
    validated_at: datetime = field(default_factory=datetime.now)

    def is_accepted(self) -> bool:
        return self.status in [HypothesisStatus.ACCEPTED, HypothesisStatus.CAPITAL_CANDIDATE, HypothesisStatus.FORWARD_PAPER]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "status": self.status.value,
            "evidence_tier": self.evidence_tier,
            "in_sample_sharpe": round(self.in_sample_sharpe, 3),
            "out_of_sample_sharpe": round(self.out_of_sample_sharpe, 3),
            "dsr_p_value": round(self.deflated_sharpe_p_value, 4),
            "cpcv_mean_sharpe": round(self.cpcv_mean_sharpe, 3),
            "cpcv_degradation_pct": round(self.cpcv_degradation_pct, 2),
            "monte_carlo_95_max_dd_pct": round(self.monte_carlo_95_max_dd_pct, 2),
            "net_profit_factor_post_tax": round(self.net_profit_factor_post_tax, 2),
            "regime_stability_score": round(self.regime_stability_score, 1),
            "current_regime_score": round(self.current_regime_score, 1),
            "recency_weighted_score": round(self.recency_weighted_score, 2),
            "portfolio_correlation": round(self.portfolio_correlation, 3),
            "window_metrics": self.window_metrics,
            "rejection_reasons": self.rejection_reasons,
            "trials_tested": self.tested_trials_count,
        }


class BaseHypothesis(ABC):
    """
    Abstract contract for all quantitative alpha hypotheses.
    """

    def __init__(self, metadata: HypothesisMetadata, parameters: Optional[Dict[str, Any]] = None):
        self.metadata = metadata
        self.parameters = parameters or {}
        self.status = HypothesisStatus.FORMULATED

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals and target positions from input DataFrame.
        DataFrame must contain OHLCV and feature store columns.
        Returns DataFrame with 'signal' column: +1.0 (LONG), -1.0 (SHORT), 0.0 (FLAT).
        """
        pass

    @abstractmethod
    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Returns the parameter search space for this hypothesis.
        """
        pass
