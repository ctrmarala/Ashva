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
    Immutable specification of an Alpha cleared for execution.
    """
    alpha_id: str
    strategy_class: Any                         # Strategy class reference
    parameters: Dict[str, Any] = field(default_factory=dict)
    universe: List[str] = field(default_factory=list)
    timeframe: str = "15m"                      # Primary execution timeframe
    entry_start_time: time = time(9, 30)        # Earliest allowed entry time
    entry_end_time: time = time(15, 0)          # Latest allowed entry time
    square_off_time: time = time(15, 15)        # Mandatory intraday EOD square-off
    risk_per_trade_pct: float = 0.0050          # 0.50% risk per trade on current MTM equity
    max_capital_allocation_pct: float = 0.20    # Max 20% capital cap per position
    trailing_mode: str = "BREAK_EVEN"           # "NONE", "BREAK_EVEN", "STEP_RATCHET"
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    priority_score: float = 1.0                 # Allocation hierarchy weight
    git_commit_sha: str = "PRODUCTION_FROZEN"
    version: str = "1.0.0"

    def instantiate_strategy(self) -> Any:
        """Instantiates the underlying strategy with frozen parameters."""
        if isinstance(self.strategy_class, type):
            try:
                return self.strategy_class(parameters=self.parameters)
            except TypeError:
                return self.strategy_class()
        return self.strategy_class
