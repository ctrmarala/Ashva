"""
Tests for Ashva Factory v2 Autonomous Discovery Controller & Knowledge Map
"""

import pytest
from src.research.knowledge_map import AlphaKnowledgeMap, AlphaCategory, MechanismStatus, AlphaResearchRecord
from src.research.discovery_controller import AutonomousDiscoveryController, ResearchBudget


def test_knowledge_map_baseline_loading():
    km = AlphaKnowledgeMap()
    assert isinstance(km.registry, dict)
    # Register sample record
    rec = AlphaResearchRecord(
        alpha_id="alpha_14",
        name="Gap Momentum Drift",
        category=AlphaCategory.GAP_MOMENTUM,
        mechanism_description="Gap continuation on 15m timeframe",
        timeframe="15m",
        entry_window="09:15-09:30",
        holding_concept="Intraday",
        status=MechanismStatus.PROVEN,
        pnl_540d_inr=150000.0,
        sharpe_540d=1.8,
        oos_trades=45,
        oos_pnl_inr=80000.0,
        positive_assets=["INFY", "TCS", "RELIANCE"]
    )
    km.register_mechanism(rec)
    assert "alpha_14" in km.registry
    assert km.registry["alpha_14"].status == MechanismStatus.PROVEN


def test_knowledge_map_duplication_rejection():
    km = AlphaKnowledgeMap()
    rec = AlphaResearchRecord(
        alpha_id="alpha_14",
        name="Gap Momentum Drift",
        category=AlphaCategory.GAP_MOMENTUM,
        mechanism_description="Gap continuation on 15m timeframe",
        timeframe="15m",
        entry_window="09:15-09:30",
        holding_concept="Intraday",
        status=MechanismStatus.PROVEN,
        pnl_540d_inr=150000.0,
        sharpe_540d=1.8,
        oos_trades=45,
        oos_pnl_inr=80000.0,
        positive_assets=["INFY", "TCS", "RELIANCE"]
    )
    km.register_mechanism(rec)
    
    # Candidate identical to Alpha 14 (GAP_MOMENTUM, 15m, 09:15-09:30)
    is_novel = km.is_novel_hypothesis(
        category=AlphaCategory.GAP_MOMENTUM,
        mechanism_desc="Gap continuation on 15m timeframe",
        timeframe="15m",
        entry_window="09:15-09:30",
    )
    assert is_novel is False, "Duplicate mechanism must be rejected."


def test_knowledge_map_novel_territory_acceptance():
    km = AlphaKnowledgeMap()
    # Candidate in unexplored territory (60m multi-day)
    is_novel = km.is_novel_hypothesis(
        category=AlphaCategory.OPENING_AUCTION,
        mechanism_desc="First pullback to session VWAP after strong opening drive",
        timeframe="15m",
        entry_window="09:45-10:30",
    )
    assert is_novel is True, "Novel unexplored mechanism should be accepted."


def test_discovery_controller_budget_enforcement():
    budget = ResearchBudget(max_hypotheses=1, max_dev_runs=1, max_runtime_seconds=60)
    controller = AutonomousDiscoveryController(budget=budget)
    reports = controller.execute_discovery_cycle()
    assert len(reports) <= 1, "Controller must halt after reaching hypothesis budget."
