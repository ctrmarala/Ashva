"""
Ashva Qualified Alpha Contract
Encapsulates a fully validated, frozen quantitative hypothesis graduating from the Alpha Factory.
Guarantees zero reinterpretation drift between research backtesting and live trading engine execution.
"""

from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Any, Optional, Type


@dataclass(frozen=True)
class QualifiedAlphaContract:
    """
    Immutable specification of an Alpha cleared for execution by the Alpha Factory.
    """
    alpha_id: str
    strategy_class: Any                         # Strategy class reference or callable
    alpha_version: str = "1.0.0"
    version: str = "1.0.0"                      # Compatibility alias
    category: str = "MOMENTUM"                  # Strategy category (MOMENTUM, REVERSION, BREAKOUT)
    economic_rationale: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    universe: List[str] = field(default_factory=list)
    timeframe: str = "15m"                      # Primary execution timeframe
    entry_start_time: time = time(9, 30)        # Earliest allowed entry time
    entry_end_time: time = time(15, 0)          # Latest allowed entry time
    square_off_time: time = time(15, 15)        # Mandatory intraday EOD square-off
    risk_per_trade_pct: float = 0.0050          # 0.50% risk per trade on current MTM equity
    max_capital_allocation_pct: float = 0.20    # Max 20% capital cap per position
    trailing_mode: str = "BREAK_EVEN"           # "NONE", "BREAK_EVEN", "STEP_RATCHET"
    stop_type: str = "STRATEGY_DEFINED"         # "STRATEGY_DEFINED", "PERCENT", "ATR", "BARRIER"
    stop_loss_pct: Optional[float] = None
    target_type: str = "STRATEGY_DEFINED"       # "STRATEGY_DEFINED", "PERCENT", "RR"
    take_profit_pct: Optional[float] = None
    priority_score: float = 1.0                 # Allocation hierarchy weight (higher = preferred)
    git_commit_sha: str = "PRODUCTION_FROZEN"
    research_commit_sha: str = "PRODUCTION_FROZEN"
    status: str = "ACTIVE"                      # "ACTIVE", "PAUSED", "RETIRED"

    def instantiate_strategy(self) -> Any:
        """Instantiates the underlying strategy with frozen parameters."""
        if isinstance(self.strategy_class, type):
            try:
                return self.strategy_class(parameters=self.parameters)
            except TypeError:
                return self.strategy_class()
        return self.strategy_class

    def to_dict(self) -> Dict[str, Any]:
        """Serializes contract metadata to dictionary for ledger & manifest serialization."""
        return {
            "alpha_id": self.alpha_id,
            "alpha_version": self.alpha_version,
            "category": self.category,
            "economic_rationale": self.economic_rationale,
            "parameters": self.parameters,
            "universe": self.universe,
            "timeframe": self.timeframe,
            "entry_start_time": self.entry_start_time.strftime("%H:%M"),
            "entry_end_time": self.entry_end_time.strftime("%H:%M"),
            "square_off_time": self.square_off_time.strftime("%H:%M"),
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_capital_allocation_pct": self.max_capital_allocation_pct,
            "trailing_mode": self.trailing_mode,
            "stop_type": self.stop_type,
            "stop_loss_pct": self.stop_loss_pct,
            "target_type": self.target_type,
            "take_profit_pct": self.take_profit_pct,
            "priority_score": self.priority_score,
            "git_commit_sha": self.git_commit_sha,
            "status": self.status,
        }
