"""
Unit Tests for AlphaLinter Static and Runtime Contract Verifier
"""

import pytest
from src.research.alpha_linter import AlphaLinter
from src.strategies.alpha_001_opening_gap_continuation import Alpha1OpeningGapContinuation


def test_alpha_linter_source_file():
    violations = AlphaLinter.lint_strategy_source_file("src/strategies/alpha_001_opening_gap_continuation.py")
    assert len(violations) == 0, f"Expected 0 violations, got: {violations}"


def test_alpha_linter_instance_valid():
    strat = Alpha1OpeningGapContinuation()
    violations = AlphaLinter.lint_strategy_instance(strat)
    assert len(violations) == 0, f"Expected 0 violations, got: {violations}"


def test_alpha_linter_detects_missing_metadata():
    class DummyInvalidStrat:
        pass

    violations = AlphaLinter.lint_strategy_instance(DummyInvalidStrat())
    assert len(violations) > 0
    assert any("Must be instance of BaseHypothesis" in v for v in violations)