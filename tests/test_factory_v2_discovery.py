"""
Tests for Ashva Factory v2 Autonomous Discovery Controller & Knowledge Map
"""

import pytest
from src.research.knowledge_map import AlphaKnowledgeMap, AlphaCategory, MechanismStatus
from src.research.discovery_controller import AutonomousDiscoveryController, ResearchBudget


def test_knowledge_map_baseline_loading():
    km = AlphaKnowledgeMap()
    assert len(km.registry) >= 15
    assert "alpha_14" in km.registry
    assert km.registry["alpha_14"].status == MechanismStatus.PROVEN
    assert "alpha_24" in km.registry
    assert km.registry["alpha_24"].status == MechanismStatus.EXPLORED_FAILED


def test_knowledge_map_duplication_rejection():
    km = AlphaKnowledgeMap()
    # Candidate identical to Alpha 14 (GAP_MOMENTUM, 15m, 09:15-09:30)
    is_novel = km.is_novel_hypothesis(
        category=AlphaCategory.GAP_MOMENTUM,
        mechanism_desc="Duplicate gap continuation",
        timeframe="15m",
        entry_window="09:15-09:30",
    )
    assert is_novel is False, "Duplicate mechanism must be rejected."


def test_knowledge_map_novel_territory_acceptance():
    km = AlphaKnowledgeMap()
    # Candidate in unexplored territory (60m multi-day)
    is_novel = km.is_novel_hypothesis(
        category=AlphaCategory.SWING_MOMENTUM,
        mechanism_desc="Multi-day consolidation breakout",
        timeframe="60m",
        entry_window="14:00-15:00",
    )
    assert is_novel is True, "Novel unexplored mechanism should be accepted."


def test_discovery_controller_budget_enforcement():
    budget = ResearchBudget(max_hypotheses=1, max_dev_runs=1, max_runtime_seconds=60)
    controller = AutonomousDiscoveryController(budget=budget)
    reports = controller.execute_discovery_cycle()
    assert len(reports) <= 1, "Controller must halt after reaching hypothesis budget."
