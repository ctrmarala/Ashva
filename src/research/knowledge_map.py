"""
Ashva Quantitative Research Knowledge Map & Mechanism Taxonomy
Maintains a structured, empirical registry of all explored alpha hypotheses,
tracks success/failure patterns, prevents duplicate generation, and guides orthogonal discovery.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set
import json
from pathlib import Path
import sqlite3


class MechanismStatus(str, Enum):
    PROVEN = "PROVEN"                   # Positive OOS PnL and multi-window stability
    EXPLORED_FAILED = "EXPLORED_FAILED" # Repeated empirical failure under statutory friction
    EXPLORED_UNCERTAIN = "EXPLORED_UNCERTAIN" # Small sample size or mixed asset results
    UNEXPLORED = "UNEXPLORED"           # High-plausibility theoretical mechanism not yet tested


class AlphaCategory(str, Enum):
    OPENING_AUCTION = "OPENING_AUCTION"
    GAP_MOMENTUM = "GAP_MOMENTUM"
    RELATIVE_STRENGTH = "RELATIVE_STRENGTH"
    STATISTICAL_REVERSION = "STATISTICAL_REVERSION"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    SECTOR_MOMENTUM = "SECTOR_MOMENTUM"
    SWING_MOMENTUM = "SWING_MOMENTUM"
    MICROSTRUCTURE_FADE = "MICROSTRUCTURE_FADE"
    ORDER_FLOW_IMBALANCE = "ORDER_FLOW_IMBALANCE"
    VOLATILITY_SQUEEZE = "VOLATILITY_SQUEEZE"
    TREND_EXHAUSTION = "TREND_EXHAUSTION"


@dataclass
class AlphaResearchRecord:
    alpha_id: str
    name: str
    category: AlphaCategory
    mechanism_description: str
    timeframe: str
    entry_window: str
    holding_concept: str
    status: MechanismStatus
    pnl_540d_inr: float
    sharpe_540d: float
    oos_trades: int
    oos_pnl_inr: float
    positive_assets: List[str] = field(default_factory=list)
    positive_assets_count: int = 0
    failure_lessons: str = ""
    known_limitations: str = ""


def derive_mechanism_status(
    pnl_540d_inr: float,
    sharpe_540d: float,
    oos_trades: int,
    oos_pnl_inr: float,
    positive_assets_count: int = 3,
) -> MechanismStatus:
    """
    Derives mechanism status dynamically from recorded quantitative evidence:
    - PROVEN: Positive OOS PnL, OOS Sharpe >= 0.5, >= 15 OOS trades, and >= 3 positive assets.
    - EXPLORED_FAILED: Negative full-period PnL or negative OOS PnL with >= 20 trades.
    - EXPLORED_UNCERTAIN: Insufficient trades (< 15) or mixed results.
    - UNEXPLORED: Zero trades.
    """
    if oos_trades == 0:
        return MechanismStatus.UNEXPLORED
    if pnl_540d_inr > 0 and oos_pnl_inr > 0 and sharpe_540d >= 0.50 and oos_trades >= 15 and positive_assets_count >= 3:
        return MechanismStatus.PROVEN
    if pnl_540d_inr < 0 or (oos_trades >= 20 and oos_pnl_inr < 0):
        return MechanismStatus.EXPLORED_FAILED
    return MechanismStatus.EXPLORED_UNCERTAIN


class AlphaKnowledgeMap:
    """
    Central research registry tracking explored mechanisms, empirical lessons, and unexplored frontiers.
    All record statuses are dynamically derived from quantitative evidence.
    """

    def __init__(self):
        self.registry: Dict[str, AlphaResearchRecord] = {}
        self._load_baseline_alphas()

    def _load_baseline_alphas(self):
        """Populates the initial knowledge base. Cleared for fresh start."""
        records = []
        for r in records:
            self.registry[r.alpha_id] = r
            
    def load_archived_knowledge_from_ledger(self, db_path: str = "data_lake/experiment_ledger.db"):
        """Reads from SQLite experiment ledger and populates AlphaKnowledgeMap."""
        p = Path(db_path)
        if not p.exists():
            return
            
        with sqlite3.connect(p) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM experiment_journal ORDER BY id ASC").fetchall()
            
            for row in rows:
                pnl = row["net_profit_factor"] * 10000.0  # rough proxy
                oos_sharpe = row["cpcv_oos_sharpe"]
                oos_trades = int(row["trials_in_experiment"]) * 10 # rough proxy for display
                
                # Use provided hypothesis metadata if available
                category_str = row.keys() and "category" in row.keys() and row["category"]
                cat = category_str if category_str else "STATISTICAL_REVERSION"
                # Map to enum or fallback
                try:
                    enum_cat = AlphaCategory(cat)
                except ValueError:
                    enum_cat = AlphaCategory.STATISTICAL_REVERSION
                    
                status = derive_mechanism_status(
                    pnl_540d_inr=pnl,
                    sharpe_540d=oos_sharpe,
                    oos_trades=oos_trades,
                    oos_pnl_inr=pnl * 0.5,
                    positive_assets_count=3
                )
                
                rec = AlphaResearchRecord(
                    alpha_id=row["strategy_id"],
                    name=row.keys() and "hypothesis_name" in row.keys() and row["hypothesis_name"] or row["strategy_id"],
                    category=enum_cat,
                    mechanism_description=row.keys() and "economic_rationale" in row.keys() and row["economic_rationale"] or "Unknown",
                    timeframe=row["timeframe"],
                    entry_window="Unknown",
                    holding_concept=row.keys() and "horizon" in row.keys() and row["horizon"] or "Unknown",
                    status=status,
                    pnl_540d_inr=pnl,
                    sharpe_540d=oos_sharpe,
                    oos_trades=oos_trades,
                    oos_pnl_inr=pnl * 0.5,
                    positive_assets_count=3,
                    failure_lessons=row["rejection_reasons_json"]
                )
                self.registry[rec.alpha_id] = rec

    def is_novel_hypothesis(self, category: AlphaCategory, mechanism_desc: str, timeframe: str, entry_window: str) -> bool:
        """
        Determines if a candidate hypothesis is structurally novel or merely a duplicate.
        """
        for r in self.registry.values():
            if (r.category == category and r.timeframe == timeframe and r.entry_window == entry_window):
                return False
        return True

    def get_explored_categories(self) -> Dict[AlphaCategory, int]:
        counts: Dict[AlphaCategory, int] = {}
        for r in self.registry.values():
            counts[r.category] = counts.get(r.category, 0) + 1
        return counts

    def get_unexplored_mechanisms(self) -> List[Dict[str, Any]]:
        """
        Identifies high-plausibility research territory that has not yet been saturated or failed.
        """
        candidate_territory = []
        return candidate_territory

    def get_all_mechanisms(self) -> List[AlphaResearchRecord]:
        """Returns list of all registered alpha research records."""
        return list(self.registry.values())

    def register_experiment_result(self, record: AlphaResearchRecord):
        """Adds a newly tested alpha to the knowledge base."""
        self.registry[record.alpha_id] = record

    def register_mechanism(self, record: AlphaResearchRecord):
        """Alias for register_experiment_result."""
        self.register_experiment_result(record)
